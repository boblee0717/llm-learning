"""
第 4 课附加实验：在真实 GPU 上测量推理优化

主课 04_inference_optimization.py 跑在 CPU 上，用玩具模型讲清了 KV Cache 的原理。
有几个问题只有真实 GPU 才能回答，而且答案大多和直觉相反：

  实验 1  GPU 基准测试的两个陷阱：忘记同步、忽视测量顺序
  实验 2  玩具模型搬到 GPU 后，为什么比 CPU 还慢
  实验 3  决定 KV Cache 收益的不是模型大小，是序列长度
  实验 4  KV Cache 显存的理论公式 vs 实测，以及 cat 追加的代价

模型实现直接复用主课文件，避免两处代码各自漂移。
完整跑一遍约 1 分钟，其中实验 3 最费时（无缓存长序列生成很慢）。
"""

import contextlib
import io
import runpy
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch


def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


if not torch.cuda.is_available():
    print("本文件需要 CUDA GPU。当前 torch 检测不到可用 GPU：")
    print(f"  torch 版本: {torch.__version__}")
    print(f"  CUDA 版本:  {torch.version.cuda}")
    print()
    print("如果机器上确实有 NVIDIA 卡，通常是装成了 CPU 版 torch（版本号带 +cpu）。")
    print("按 pytorch.org 的指引装对应 CUDA 版本的 wheel 即可。")
    sys.exit(0)


# 主课文件是脚本式的，import 会把它的全部输出打一遍。
# 这里静默执行一次只为取出模型类——好处是永远和主课实现保持一致。
_LESSON = Path(__file__).with_name("04_inference_optimization.py")
with contextlib.redirect_stdout(io.StringIO()):
    _ns = runpy.run_path(str(_LESSON))
CachedGPT = _ns["CachedGPT"]

device = torch.device("cuda")
props = torch.cuda.get_device_properties(0)


def cpu_runtime(fn, repeats=5):
    fn()
    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - start)
    return result, statistics.median(durations)


def cuda_runtime(fn, repeats=5):
    """GPU 版计时：每次测量后同步，确保测到的是真实计算耗时而非提交耗时。"""
    fn()
    torch.cuda.synchronize()

    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    return result, statistics.median(durations)


section("环境")
print(f"GPU:    {props.name}")
print(f"算力:   sm_{props.major}{props.minor}")
print(f"显存:   {props.total_memory / 1024**3:.1f} GB")
print(f"torch:  {torch.__version__}  (CUDA {torch.version.cuda})")


# ============================================================
# 先测干净的 CPU 基准
# ============================================================
# 必须放在所有 GPU 操作之前。实验 1 会证明：一旦 GPU 满载过，
# 同一进程后续的 CPU 计时会整体变慢好几倍，再也测不准了。

TINY_CFG = dict(vocab_size=200, n_layers=2)
TINY_PROMPT_LEN = 20
TINY_NEW_TOKENS = 50

torch.manual_seed(0)
tiny_cpu = CachedGPT(**TINY_CFG).eval()
tiny_prompt_cpu = torch.randint(0, TINY_CFG["vocab_size"], (1, TINY_PROMPT_LEN))

_, cpu_no_clean = cpu_runtime(
    lambda: tiny_cpu.generate_no_cache(tiny_prompt_cpu, max_new_tokens=TINY_NEW_TOKENS)
)
_, cpu_yes_clean = cpu_runtime(
    lambda: tiny_cpu.generate_with_cache(tiny_prompt_cpu, max_new_tokens=TINY_NEW_TOKENS)
)


# ============================================================
# 实验 1: GPU 基准测试的两个陷阱
# ============================================================

section("实验 1: GPU 基准测试的两个陷阱")

print("陷阱一：忘记 synchronize")
print("CUDA 把 kernel 提交到队列后立刻返回，GPU 可能还在算。")
print("perf_counter 测到的是「提交耗时」，不是「计算耗时」。\n")

a = torch.randn(4096, 4096, device=device)

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(50):
    a @ a
t_submit = time.perf_counter() - start
torch.cuda.synchronize()
t_real = time.perf_counter() - start

flops = 2 * 4096**3 * 50
print("50 次 4096×4096 矩阵乘：")
print(f"  不同步（只测提交）: {t_submit * 1000:8.2f} ms"
      f"  → 反推 {flops / t_submit / 1e12:6.1f} TFLOPS")
print(f"  同步后（真实耗时）: {t_real * 1000:8.2f} ms"
      f"  → 反推 {flops / t_real / 1e12:6.1f} TFLOPS")
print(f"  低估了 {t_real / t_submit:.1f} 倍")
print()
print("自检技巧：用耗时反推算力，超过硬件标称值就说明漏了同步。")
print("（RTX 4060 的 fp32 峰值约 15 TFLOPS，同步后的数字才在合理范围内。）")

del a
torch.cuda.empty_cache()

print()
print("-" * 60)
print("陷阱二：测量顺序会污染结果")
print()
print("刚才那 50 次矩阵乘让 GPU 满载了一秒。现在重测一遍开头那组 CPU 基准：\n")

_, cpu_no_dirty = cpu_runtime(
    lambda: tiny_cpu.generate_no_cache(tiny_prompt_cpu, max_new_tokens=TINY_NEW_TOKENS)
)
_, cpu_yes_dirty = cpu_runtime(
    lambda: tiny_cpu.generate_with_cache(tiny_prompt_cpu, max_new_tokens=TINY_NEW_TOKENS)
)

print(f"{'':22} {'无缓存':>10} {'有缓存':>10}")
print(f"{'GPU 满载前 (干净)':22} {cpu_no_clean * 1000:8.2f}ms {cpu_yes_clean * 1000:8.2f}ms")
print(f"{'GPU 满载后':22} {cpu_no_dirty * 1000:8.2f}ms {cpu_yes_dirty * 1000:8.2f}ms")
print(f"{'倍数':22} {cpu_no_dirty / cpu_no_clean:8.2f}x {cpu_yes_dirty / cpu_yes_clean:8.2f}x")
print()
print("同一份代码、同一个进程，CPU 耗时凭空涨了几倍。可能的原因：")
print("  - GPU 升到高功耗状态后不会立刻降回，整机功耗预算被挤占，CPU 被限频")
print("  - Windows WDDM 下 GPU 由系统调度，CUDA 上下文活跃期间有持续的 CPU 开销")
print("实测静置几秒、释放显存都不会恢复，所以这不是临时抖动。")
print()
print("→ 教训：混合 CPU/GPU 的基准测试，必须先把 CPU 基准测完再碰 GPU；")
print("  更稳妥的做法是分成两个独立进程各测各的。")
print("  本文件后面凡是引用 CPU 数字，都用的是开头那份干净基准。")


# ============================================================
# 实验 2: 玩具模型在 GPU 上反而更慢
# ============================================================

section("实验 2: 玩具模型在 GPU 上比 CPU 还慢")

torch.manual_seed(0)
tiny_gpu = CachedGPT(**TINY_CFG).eval().to(device)
tiny_gpu.load_state_dict(tiny_cpu.state_dict())
tiny_prompt_gpu = tiny_prompt_cpu.to(device)

out_no, gpu_no = cuda_runtime(
    lambda: tiny_gpu.generate_no_cache(tiny_prompt_gpu, max_new_tokens=TINY_NEW_TOKENS)
)
out_yes, gpu_yes = cuda_runtime(
    lambda: tiny_gpu.generate_with_cache(tiny_prompt_gpu, max_new_tokens=TINY_NEW_TOKENS)
)

assert torch.equal(out_no, out_yes), "GPU 上两条路径的 greedy 结果也必须一致"

print(f"模型: d_model=64, n_layers=2, prompt={TINY_PROMPT_LEN}, 生成={TINY_NEW_TOKENS}\n")
print(f"{'':8} {'无缓存':>12} {'有缓存':>12} {'加速比':>10}")
print(f"{'CPU':8} {cpu_no_clean * 1000:10.2f}ms {cpu_yes_clean * 1000:10.2f}ms "
      f"{cpu_no_clean / cpu_yes_clean:9.2f}x")
print(f"{'GPU':8} {gpu_no * 1000:10.2f}ms {gpu_yes * 1000:10.2f}ms "
      f"{gpu_no / gpu_yes:9.2f}x")
print()
print(f"GPU 比 CPU 慢了 {gpu_yes / cpu_yes_clean:.1f} 倍——这不是配置错误，是规模太小。")
print()
print("decode 每步只算 1 个 token，几千次浮点运算。RTX 4060 有 3072 个 CUDA 核心，")
print("算力利用率不到 1%，时间全花在「把活派给 GPU」这件事本身上：")
print(f"  每步耗时 {gpu_yes / TINY_NEW_TOKENS * 1000:.2f} ms，一步要走 2 层 × 十几个算子")
print("  每个算子都有 Python dispatch（约 10~20μs）+ kernel 启动（约 5~10μs）")
print("  这些固定开销和「算多少」无关，却占了绝大部分时间 → launch-bound")
print()
print("所以无缓存版在 GPU 上也没慢多少：它多出来的计算量对闲着的核心几乎是免费的。")
print("KV Cache 省的是计算量，而这里的瓶颈根本不是计算量。")
print()
print("→ 优化前先判断瓶颈是算力、带宽还是启动开销，否则会优化错方向。")
print("  真实引擎用 CUDA Graph、算子融合、continuous batching 对付的正是这一类开销。")

del tiny_gpu, tiny_prompt_gpu, out_no, out_yes
torch.cuda.empty_cache()


# ============================================================
# 实验 3: 序列长度才是决定变量
# ============================================================

section("实验 3: 决定 KV Cache 收益的是序列长度")

BIG_CFG = dict(vocab_size=10000, d_model=768, n_heads=12, n_layers=12, max_seq_len=2048)
BIG_NEW_TOKENS = 32

torch.manual_seed(0)
big = CachedGPT(**BIG_CFG).eval().to(device)
n_params = sum(p.numel() for p in big.parameters())

print(f"模型: d_model=768, n_layers=12, n_heads=12, 参数 {n_params / 1e6:.1f}M "
      f"({n_params * 4 / 1024**2:.0f} MB fp32)")
print(f"固定生成 {BIG_NEW_TOKENS} 个 token，只改 prompt 长度：\n")
print(f"{'prompt':>8} {'无缓存':>12} {'有缓存':>12} {'加速比':>10}")

for prompt_len in (128, 384, 768, 1536):
    p = torch.randint(0, BIG_CFG["vocab_size"], (1, prompt_len), device=device)
    o_no, t_no = cuda_runtime(
        lambda: big.generate_no_cache(p, max_new_tokens=BIG_NEW_TOKENS), repeats=3
    )
    o_yes, t_yes = cuda_runtime(
        lambda: big.generate_with_cache(p, max_new_tokens=BIG_NEW_TOKENS), repeats=3
    )
    assert torch.equal(o_no, o_yes), f"prompt={prompt_len} 时两条路径结果不一致"
    print(f"{prompt_len:>8} {t_no * 1000:10.1f}ms {t_yes * 1000:10.1f}ms "
          f"{t_no / t_yes:9.2f}x")
    del p, o_no, o_yes
    torch.cuda.empty_cache()

print()
print("看两列的走势，比看加速比更有启发：")
print("  有缓存列几乎是一条水平线——每步只处理 1 个 token，成本主要是")
print("    「读一遍权重 + 走一遍算子」，和已经生成了多长几乎无关")
print("  无缓存列随 prompt 长度快速上涨——每步都要重算整个序列，attention 是 O(L²)")
print()
print("这才是 KV Cache 的本质价值：把每步成本从「随上下文增长」压成「近乎恒定」。")
print("短 prompt 时两者都被固定开销主导，看不出差别；上下文越长，差距越悬殊。")
print("主课练习 1 让你验证的规律，就是这张表。")
print()
print("反过来说：上一版实验里我们放大模型宽度和层数，加速比并没有起色——")
print("因为那只是让每步的固定开销一起变大，没有改变 O(L²) 那一项的占比。")

del big
torch.cuda.empty_cache()


# ============================================================
# 实验 4: KV Cache 显存，理论 vs 实测
# ============================================================

section("实验 4: KV Cache 显存实测")

MEM_CFG = dict(vocab_size=10000, d_model=768, n_heads=12, n_layers=12, max_seq_len=1024)
MEM_SEQ_LEN = 512

torch.manual_seed(0)
mem_model = CachedGPT(**MEM_CFG).eval().to(device)
head_dim = MEM_CFG["d_model"] // MEM_CFG["n_heads"]


def cache_bytes_of(caches):
    return sum(
        k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in caches
    )


print("主课 Part 6 的公式：")
print("  2 × n_layers × batch × n_kv_heads × seq_len × head_dim × bytes\n")
print(f"配置: n_layers={MEM_CFG['n_layers']}, n_heads={MEM_CFG['n_heads']}, "
      f"head_dim={head_dim}, seq_len={MEM_SEQ_LEN}, fp32\n")
print(f"{'batch':>6} {'理论 cache':>12} {'实测 cache':>12} "
      f"{'prefill 峰值':>14} {'峰值/cache':>11}")

for batch in (1, 4, 16):
    tokens = torch.randint(0, MEM_CFG["vocab_size"], (batch, MEM_SEQ_LEN), device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        _, caches = mem_model(tokens)
    peak = torch.cuda.max_memory_allocated() - base

    measured = cache_bytes_of(caches)
    theory = 2 * MEM_CFG["n_layers"] * batch * MEM_CFG["n_heads"] * MEM_SEQ_LEN * head_dim * 4
    print(f"{batch:>6} {theory / 1024**2:>10.1f} MB {measured / 1024**2:>10.1f} MB "
          f"{peak / 1024**2:>12.1f} MB {peak / measured:>10.1f}x")

    del tokens, caches

print()
print("两点观察：")
print("  1. 理论值和实测完全吻合——公式没有隐藏项，就是 K/V 两份张量的字节数。")
print("  2. prefill 峰值是 cache 的两倍多。多出来的是 attention scores")
print(f"     (batch × n_heads × L²，这里单层就有 {MEM_SEQ_LEN**2 * MEM_CFG['n_heads'] * 4 / 1024**2:.0f} MB)、")
print("     FFN 中间激活、logits 等临时张量。它们用完即弃，")
print("     但决定了「这张卡跑不跑得下来」。")
print()
print("→ 显存预算要按峰值算，不能只算 KV Cache。")
print("  这也是 vLLM 之类的引擎要做精细显存管理的原因。")

print()
print("-" * 60)
print("再看教学版用 cat 追加 cache 的代价（主课 block 里那行注释提到过）：\n")

tokens = torch.randint(0, MEM_CFG["vocab_size"], (1, MEM_SEQ_LEN), device=device)
with torch.no_grad():
    _, caches = mem_model(tokens)
cache_size = cache_bytes_of(caches)
next_token = torch.randint(0, MEM_CFG["vocab_size"], (1, 1), device=device)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
before = torch.cuda.memory_allocated()
with torch.no_grad():
    _, new_caches = mem_model(next_token, caches, start_pos=MEM_SEQ_LEN)
peak_delta = torch.cuda.max_memory_allocated() - before

print(f"  decode 前 cache 大小:        {cache_size / 1024**2:8.1f} MB")
print(f"  decode 一步的显存增量峰值:   {peak_delta / 1024**2:8.1f} MB")
print(f"  增量约等于 {peak_delta / cache_size:.2f} 份 cache")
print()
print("  torch.cat 必须为拼接结果分配一整块新内存，而旧 cache 在赋值完成前")
print("  还不能释放——峰值时刻新旧两份同时驻留，总共占了 2 份 cache 的显存。")
print("  序列越长、层数越多，这份浪费越可观。")
print()
print("  生产实现的两条出路：预分配到 max_seq_len 后原地写入（省掉复制，")
print("  但要为最长情况预留），或用 PagedAttention 按页追加（按需分配、减少碎片）。")

del mem_model, tokens, caches, new_caches, next_token
torch.cuda.empty_cache()


# ============================================================
# 小结
# ============================================================

section("小结")

print("把主课的 CPU 结论搬到真实 GPU 上，四件事值得记住：\n")
print("1. GPU 计时必须 synchronize，且要留意测量顺序。")
print("   反推 TFLOPS 是个好用的自检；CPU 基准要在碰 GPU 之前测完。")
print()
print("2. 小模型在 GPU 上是 launch-bound，甚至比 CPU 更慢。")
print("   省计算量不等于省时间——先确认瓶颈在哪，再谈优化。")
print()
print("3. KV Cache 的收益由序列长度决定，不由模型大小决定。")
print("   它把每步成本从「随上下文增长」压成「近乎恒定」。")
print()
print("4. 显存要按峰值预算。教学版 cat 追加会短暂占用两份 cache，")
print("   生产实现靠预分配或分页规避。")

section("动手练习")

print("练习 1：改成 fp16 / bf16")
print("  把实验 3、4 的模型和输入 .half() 后重跑")
print("  预期：显存减半，耗时下降（Tensor Core 生效）")
print("  观察 greedy 结果是否还和 fp32 完全一致，并解释为什么")
print()
print("练习 2：把实验 3 的表格画出来")
print("  横轴 prompt 长度（128 到 1536），两条曲线分别是有/无缓存耗时")
print("  验证「有缓存近似水平线、无缓存呈二次增长」")
print()
print("练习 3：batch 扩展与显存墙")
print("  固定模型，把 batch 从 1 加到 OOM 为止，记录每档显存与吞吐")
print("  先用 Part 6 的公式预测 8 GB 卡在 seq_len=1024 时能开多大 batch")
print("  再实测对比，解释误差来自哪里（提示：看实验 4 的峰值/cache 比）")
print()
print("练习 4：CUDA Graph")
print("  用 torch.cuda.CUDAGraph 捕获 decode 单步，消除 launch 开销")
print("  在实验 2 的玩具模型上测收益，验证 launch-bound 的判断")
