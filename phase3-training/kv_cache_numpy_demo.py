"""
======================================================
KV Cache 最小演示（纯 NumPy 手写）
======================================================

这是 `05_inference_optimization.py`（PyTorch 版第 5 课）的轻量对照版。
目的：剥掉框架，用最少的代码把「KV Cache 到底省了什么」看透。

一句话回顾：
  自回归生成时，旧 token 的 Key / Value 算过一次后就永远不变
  （因为有 causal mask，旧 token 看不到新 token）。
  把它们缓存下来，每生成一个新 token 就只算「这一个 token」的 Q/K/V，
  而不是把整段历史重算一遍。

本 demo 做三件事：
  Part 1: 同一段权重，分别用「无缓存整段重算」和「有缓存逐 token」两种方式
          算注意力输出，验证两者数值完全一致（缓存不改变结果，只省计算）。
  Part 2: 统计两种方式的「Q/K/V 投影次数」，直观看到 O(n^2) vs O(n) 的差距。
  Part 3: 用公式估算真实模型的 KV Cache 显存占用。

运行方式：python3 kv_cache_numpy_demo.py
"""

import sys

import numpy as np

# 作者在 Windows / PowerShell 环境，避免中文输出乱码
sys.stdout.reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定 softmax：先减最大值再 exp，防止溢出。"""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ============================================================
# 共享的单头注意力权重
# 两种实现（无缓存 / 有缓存）必须用同一套权重，结果才可比。
# ============================================================
rng = np.random.default_rng(0)
d_model = 8
W_q = rng.standard_normal((d_model, d_model)) * 0.5
W_k = rng.standard_normal((d_model, d_model)) * 0.5
W_v = rng.standard_normal((d_model, d_model)) * 0.5
scale = np.sqrt(d_model)


def attention_no_cache(X: np.ndarray) -> np.ndarray:
    """无缓存版：每次都把整段序列的 Q/K/V 全部重算一遍。

    X: (T, d_model)，返回 (T, d_model)。
    这正是「不做 KV Cache」时，每生成一个 token 要重复干的活。
    """
    Q = X @ W_q              # (T, d)
    K = X @ W_k              # (T, d)
    V = X @ W_v              # (T, d)

    scores = Q @ K.T / scale  # (T, T)
    # causal mask：第 i 个 token 不能看第 j>i 个（未来）token
    T = X.shape[0]
    future = np.triu(np.ones((T, T), dtype=bool), k=1)  # 上三角(不含对角线)=未来
    scores[future] = -np.inf

    A = softmax(scores, axis=-1)  # (T, T)
    return A @ V                   # (T, d)


class KVCache:
    """最朴素的 KV Cache：把历史 token 的 K、V 按行堆叠存起来。"""

    def __init__(self) -> None:
        self.K = None  # (t, d)
        self.V = None  # (t, d)

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> None:
        # k_new, v_new: (d,) —— 当前这一个新 token 的 K、V
        if self.K is None:
            self.K = k_new[None, :]
            self.V = v_new[None, :]
        else:
            self.K = np.concatenate([self.K, k_new[None, :]], axis=0)
            self.V = np.concatenate([self.V, v_new[None, :]], axis=0)


def attention_step_with_cache(x_t: np.ndarray, cache: KVCache) -> np.ndarray:
    """有缓存版：只为「当前这一个新 token」算 Q/K/V，K、V 追加进缓存后复用。

    x_t: (d_model,) 当前新 token，返回 (d_model,)。
    注意：这里不需要 causal mask —— 缓存里只有「过去 + 自己」，
    本来就看不到未来，因果性天然成立。
    """
    q = x_t @ W_q  # (d,)  只有新 token 需要 Q
    k = x_t @ W_k  # (d,)
    v = x_t @ W_v  # (d,)
    cache.append(k, v)  # 新 token 的 K、V 进缓存

    scores = (q @ cache.K.T) / scale  # (t,) 当前 Q 对所有历史 K
    a = softmax(scores, axis=-1)       # (t,)
    return a @ cache.V                  # (d,)


# ============================================================
# Part 1: 验证「有缓存」和「无缓存」结果完全一致
# ============================================================
section("Part 1: 缓存不改变结果，只省计算")

T = 6
X = rng.standard_normal((T, d_model))  # 假装是 T 个 token 的隐藏向量

out_full = attention_no_cache(X)

cache = KVCache()
outs = []
for t in range(T):
    outs.append(attention_step_with_cache(X[t], cache))
out_cached = np.stack(outs, axis=0)

max_diff = np.abs(out_full - out_cached).max()
print(f"序列长度 T = {T}, d_model = {d_model}")
print(f"无缓存整段输出 shape : {out_full.shape}")
print(f"有缓存逐 token 输出 shape: {out_cached.shape}")
print(f"缓存中 K 的最终 shape : {cache.K.shape}  (每个历史 token 一行)")
print(f"两种方式最大差异     : {max_diff:.2e}  (≈0 说明结果一致)")
print()


# ============================================================
# Part 2: 计算量对比 —— O(n^2) vs O(n)
# ============================================================
section("Part 2: 投影计算量 O(n^2) vs O(n)")

print("""
场景：已有 P 个 prompt token，再自回归生成 G 个 token。
关注「K/V 投影」这个最能体现重复计算的操作。

  无缓存：生成第 i 个新 token 时，要把前面所有 token 的 K、V 全部重算一遍。
  有缓存：每步只算新 token 自己的 K、V 投影（1 次），历史直接复用。
""")

P, G = 10, 50
# 无缓存：生成第 i 个新 token（序列已有 P+i-1 个）时，重算全部 P+i 个 token 的 K、V
no_cache_proj = sum(P + i for i in range(1, G + 1))
# 有缓存：prefill 阶段 P 个，decode 阶段每步 1 个，共 P + G 次
with_cache_proj = P + G

print(f"prompt 长度 P = {P}, 生成 token 数 G = {G}")
print(f"  无缓存 K/V 投影总次数: {no_cache_proj:>6}  (∝ (P+G)^2，重复计算爆炸)")
print(f"  有缓存 K/V 投影总次数: {with_cache_proj:>6}  (∝ P+G，每个 token 只算一次)")
print(f"  节省倍数            : {no_cache_proj / with_cache_proj:>6.1f}x")
print("  → 生成越长，KV Cache 省得越多；代价是要拿显存存下所有历史 K、V。")
print()


# ============================================================
# Part 3: KV Cache 显存估算
# ============================================================
section("Part 3: KV Cache 显存占用估算")

print("""
KV Cache 用显存换算力。每一层都要存 K 和 V 两份：
  显存 = 2(K,V) × n_layers × n_heads × seq_len × head_dim × batch × bytes
""")

configs = [
    # name, n_layers, n_heads, head_dim, seq_len
    ("GPT-2 (124M)", 12, 12, 64, 1024),
    ("LLaMA-7B", 32, 32, 128, 4096),
    ("LLaMA-70B", 80, 64, 128, 4096),
]
batch = 1
bytes_per_elem = 2  # FP16

for name, n_layers, n_heads, head_dim, seq_len in configs:
    nbytes = 2 * n_layers * n_heads * seq_len * head_dim * batch * bytes_per_elem
    gb = nbytes / (1024 ** 3)
    print(f"  {name:14s}: seq_len={seq_len:>5}, batch=1 → {gb:6.2f} GB"
          f"   (batch=32 → {gb * 32:6.1f} GB)")

print("""
所以长上下文 / 大 batch 时 KV Cache 可能比模型权重还吃显存，催生了：
  - MQA / GQA : 多个注意力头共享 K、V，直接砍掉大部分缓存
  - MLA       : 把 K、V 压成低维潜向量再缓存（DeepSeek-V2/V3）
  - 量化       : KV Cache 用 INT8 / FP8 存储
  - PagedAttention(vLLM): 像操作系统分页一样管理显存，减少碎片
配套论文见 papers/kv-cache/ 与 papers/deepseek/DeepSeek-V2。
""")

section("完成：你已经用纯 NumPy 跑通了 KV Cache 的核心机制")
