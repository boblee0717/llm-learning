"""
第 6 课：工程化训练循环（torch 版）⭐
=====================================
前面 5 课你已经备齐零件：tensor、autograd、Module、损失/优化器、DataLoader。
这节课把它们组装成一个【工业级训练脚本】——phase3 第 1 课一上来就默认你会的那套：
五步曲 + train/eval 切换 + 复现性 + 梯度裁剪/累积 + checkpoint 断点续训 +
early stopping + AMP 混合精度。学完这课，你就能独立写出一个不丢人的训练 loop。

核心问题：
- 训练循环到底有哪「五步」？为什么评估时要 model.eval() + torch.no_grad()？
- 怎么让一次训练【可复现】（同样的种子跑出同样的 loss）？
- 显存不够想要大 batch 怎么办（梯度累积）？梯度爆炸怎么办（梯度裁剪）？
- 训练中途断电了怎么续训（checkpoint 存 model/opt/scheduler/step）？
- 什么时候该提前停（early stopping 看 val loss）？AMP 混合精度为什么能省显存提速？

与大模型的关系：
- 所有 LLM 训练脚本（nanoGPT / GPT-NeoX / Megatron）都是这套骨架：
  梯度累积撑大有效 batch、梯度裁剪防 loss 爆、AMP 上 bf16/fp16、
  定期 checkpoint 断点续训、复现性靠固定 seed + generator。

前置：本专项第 4 课（loss/optim）、第 5 课（DataLoader），以及 phase3 第 1 课预习
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 文件锚定脚本自身目录


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# 一个全程用到的小回归模型 + 数据：拟合 y = sin(x)
def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


def make_data(n=200):
    xs = torch.linspace(-3, 3, n).unsqueeze(1)
    ys = torch.sin(xs)
    return xs, ys


# ============================================================
section("Part 1: 训练五步曲 + train/eval 切换 + @torch.no_grad")
# ============================================================
# 第 4 课你已经见过五步曲，这里再钉死一遍，并补上「评估」该怎么写。
#   1. pred = model(x)        前向
#   2. loss = loss_fn(pred,y) 算损失
#   3. opt.zero_grad()        清空上一步残留梯度（phase1 你手动清零的那步）
#   4. loss.backward()        反向求梯度（autograd）
#   5. opt.step()             按优化算法更新参数
#
# 训练(train) 与 评估(eval) 的两个关键区别（很多人栽在这）：
#   * model.train() / model.eval()：切换 Dropout、BatchNorm 等层的行为。
#       train 下 Dropout 会随机丢、BN 用 batch 统计；eval 下 Dropout 关闭、BN 用 running 统计。
#   * torch.no_grad()：评估不需要梯度，关掉计算图能省显存、提速。
#       （对照第 2 课：no_grad 下前向不再记录 grad_fn。）

torch.manual_seed(0)
xs, ys = make_data()
model = make_model()
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)


@torch.no_grad()  # 装饰器写法：整个函数体都在 no_grad 上下文里跑
def evaluate(model, xs, ys):
    was_training = model.training      # 记下进来前的状态，结束好恢复
    model.eval()                       # 切到评估模式（关 Dropout/BN 更新）
    loss = F.mse_loss(model(xs), ys).item()
    if was_training:
        model.train()                  # 恢复原状态，别污染后续训练
    return loss


print("训练 5 步（每步打印 train loss），并在前后各评估一次：")
print(f"  训练前 eval loss = {evaluate(model, xs, ys):.5f}")
model.train()
for step in range(5):
    pred = model(xs)                   # 1. 前向
    loss = F.mse_loss(pred, ys)        # 2. 损失
    opt.zero_grad()                    # 3. 清梯度
    loss.backward()                    # 4. 反向
    opt.step()                         # 5. 更新
    print(f"  step {step}  train loss={loss.item():.5f}")
print(f"  训练后 eval loss = {evaluate(model, xs, ys):.5f}")
print("evaluate 跑完后 model.training =", model.training, "（已恢复 train 状态）")


# ============================================================
section("Part 2: 复现性 —— manual_seed 与固定 generator")
# ============================================================
# 「同样的代码同样的种子，跑出同样的结果」是调试的前提。否则你改一行不知道
# loss 变化是因为改对了还是因为随机性。复现性靠两件事：
#   1. torch.manual_seed(s)：固定全局随机源（影响参数初始化、Dropout、全局 randn 等）。
#   2. 给会用到随机的地方传一个【固定 seed 的 generator】（如 DataLoader 的 shuffle、
#      torch.randn(..., generator=g)），避免被别处的随机调用打乱顺序。

def train_once(seed):
    torch.manual_seed(seed)            # 固定全局种子 → 参数初始化可复现
    m = make_model()
    o = torch.optim.SGD(m.parameters(), lr=0.05)
    x, y = make_data(64)
    for _ in range(10):
        l = F.mse_loss(m(x), y)
        o.zero_grad(); l.backward(); o.step()
    return l.item()

a = train_once(123)
b = train_once(123)                    # 同种子 → 完全一样
c = train_once(456)                    # 不同种子 → 不一样
print(f"seed=123 第一次 loss = {a:.8f}")
print(f"seed=123 第二次 loss = {b:.8f}  →  与第一次相同: {a == b}")
print(f"seed=456       loss = {c:.8f}  →  与上面不同: {a != c}")

# 固定 generator：局部随机源，不受外界全局随机调用干扰
g = torch.Generator().manual_seed(2024)
r1 = torch.randn(3, generator=g)
g = torch.Generator().manual_seed(2024)
r2 = torch.randn(3, generator=g)
print(f"\n固定 generator 两次采样一致: {torch.allclose(r1, r2)}")
print("→ LLM 训练里 DataLoader(shuffle=True) 也应传 generator，保证 epoch 顺序可复现")


# ============================================================
section("Part 3: 梯度裁剪 clip_grad_norm_ 与梯度累积")
# ============================================================
# 【梯度裁剪】防止梯度爆炸：把所有参数梯度看成一个大向量，若其 L2 范数超过 max_norm，
# 就整体等比例缩小到 max_norm。LLM 训练几乎必开（典型 max_norm=1.0），防 loss 突然 NaN。
#   total_norm = clip_grad_norm_(model.parameters(), max_norm)
#   它【就地】缩放梯度，返回【裁剪前】的总范数（可监控梯度健康度）。
#
# 【梯度累积】显存不够想要大 batch：把一个大 batch 拆成 N 个小 batch，逐个 backward
# 让梯度【累加】（PyTorch 默认就是累加，不 zero 就会叠加），攒够 N 个再 step 一次。
# 关键坑：每个小 batch 的 loss 要【除以 accum_steps】，否则等效放大了 lr。

torch.manual_seed(0)
model = make_model()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
xs, ys = make_data()

# --- 梯度裁剪演示 ---
loss = F.mse_loss(model(xs), ys)
opt.zero_grad()
loss.backward()
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
# 裁剪后重新量一下范数
after = math.sqrt(sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None))
print(f"裁剪前梯度总范数 = {total_norm.item():.4f}（clip_grad_norm_ 的返回值）")
print(f"裁剪后梯度总范数 = {after:.4f}（被压到 max_norm=0.5 附近）")
opt.step()

# --- 梯度累积演示：把 200 条数据拆成 4 个 micro-batch ---
torch.manual_seed(0)
model = make_model()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
accum_steps = 4
batches = list(torch.chunk(xs, accum_steps)), list(torch.chunk(ys, accum_steps))

opt.zero_grad()
running = 0.0
for xb, yb in zip(*batches):
    micro_loss = F.mse_loss(model(xb), yb) / accum_steps  # 除以 accum！
    micro_loss.backward()                                 # 梯度累加，不 zero
    running += micro_loss.item()
opt.step()                                                # 攒够 4 个才更新一次
opt.zero_grad()
print(f"\n梯度累积：{accum_steps} 个 micro-batch 累加后 step 一次，"
      f"平均 loss = {running:.5f}")
print("→ 有效 batch = micro_batch_size × accum_steps，省显存换大 batch")
print("  注：此等价以各 micro-batch 等大小为前提；末批不满时 mean 归约会给样本不同权重，")
print("      故工程上常用 drop_last 或按样本数加权，避免梯度被尾批轻微带偏。")


# ============================================================
section("Part 4: checkpoint —— 保存/加载实现断点续训")
# ============================================================
# 训练几天的大模型不能断电就白干。checkpoint = 把恢复训练所需的【全部状态】
# 存成一个 .pt 文件：model / optimizer / scheduler 的 state_dict + 当前 step。
#   保存：torch.save({...}, path)
#   加载：ckpt = torch.load(path); model.load_state_dict(ckpt["model"]); ...
# 只存 model 不够！optimizer（Adam 的动量/方差）和 scheduler（lr 进度）也得存，
# 否则续训时优化器从零开始、lr 跳回起点，等于没续好。

torch.manual_seed(0)
model = make_model()
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
xs, ys = make_data()

ckpt_path = os.path.join(SCRIPT_DIR, "_ckpt_demo.pt")

# 先训练 30 步，在第 30 步存盘
for step in range(30):
    loss = F.mse_loss(model(xs), ys)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
torch.save({
    "step": 30,
    "model": model.state_dict(),
    "opt": opt.state_dict(),
    "sched": sched.state_dict(),
}, ckpt_path)
print(f"已在 step 30 保存 checkpoint → {os.path.basename(ckpt_path)}")
print(f"  存盘时 lr = {opt.param_groups[0]['lr']:.5f}，loss = {loss.item():.5f}")

# 模拟「断电重启」：新建一套全 0 进度的对象，从盘里恢复
model2 = make_model()
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-2)
sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=20, gamma=0.5)
ckpt = torch.load(ckpt_path)              # CPU 上加载；GPU 时可传 map_location
model2.load_state_dict(ckpt["model"])
opt2.load_state_dict(ckpt["opt"])
sched2.load_state_dict(ckpt["sched"])
start_step = ckpt["step"]
print(f"\n断电重启：从 step {start_step} 恢复，"
      f"lr 续上为 {opt2.param_groups[0]['lr']:.5f}（没跳回起点）")

# 续训 10 步
for step in range(start_step, start_step + 10):
    loss = F.mse_loss(model2(xs), ys)
    opt2.zero_grad(); loss.backward(); opt2.step(); sched2.step()
print(f"续训到 step {start_step + 10}，loss = {loss.item():.5f}")

os.remove(ckpt_path)                      # 跑完删文件，保持目录干净
print(f"已删除 checkpoint 文件，目录保持干净")


# ============================================================
section("Part 5: early stopping —— patience 监控 val loss")
# ============================================================
# 训练久了会过拟合：train loss 还在降，但 val loss 开始升。early stopping 的逻辑：
# 监控 val loss，若连续 patience 个评估点都没有改善（没刷新历史最低），就提前停。
# 这样既省算力，又拿到泛化最好的那个点（通常配合「保存最优 checkpoint」）。

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta            # 至少要改善这么多才算「有改善」
        self.best = float("inf")
        self.bad_epochs = 0                   # 连续没改善的次数

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss              # 刷新历史最低 → 计数清零
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1                  # 没改善 → 计数 +1
        return self.bad_epochs >= self.patience  # 攒够 patience → 该停了


# 模拟一段「先降后升」的 val loss 曲线
stopper = EarlyStopper(patience=3)
val_curve = [1.0, 0.8, 0.6, 0.55, 0.6, 0.65, 0.7, 0.8]
print("模拟 val loss 曲线，patience=3：")
for i, v in enumerate(val_curve):
    stop = stopper.step(v)
    flag = "  ← 触发 early stop！" if stop else ""
    print(f"  epoch {i}  val_loss={v:.2f}  best={stopper.best:.2f}  "
          f"连续未改善={stopper.bad_epochs}{flag}")
    if stop:
        print(f"在 epoch {i} 提前停止（best val_loss={stopper.best:.2f}）")
        break


# ============================================================
section("Part 6: AMP 混合精度 —— autocast + GradScaler")
# ============================================================
# AMP（Automatic Mixed Precision）：前向用低精度（fp16/bf16）算，省显存 + 提速，
# 关键数值（如 loss 累加、参数主副本）仍用 fp32 保精度。两个主角：
#   * torch.autocast(device_type, dtype)：上下文里自动把算子切到低精度。
#   * torch.amp.GradScaler("cuda")：fp16 梯度太小会下溢成 0，先把 loss 放大(scale)
#       再 backward，step 前再 unscale，防止小梯度消失。（bf16 动态范围大，可不用 scaler。）
# GPU 上的标准写法（注释展示，CPU 上不真正执行）：
#     scaler = torch.amp.GradScaler("cuda")   # ← 旧入口 torch.cuda.amp.GradScaler() 自 2.x 起已弃用
#     with torch.autocast("cuda", dtype=torch.float16):
#         loss = loss_fn(model(x), y)
#     scaler.scale(loss).backward()      # 放大 loss 再反向
#     scaler.unscale_(opt)               # 还原梯度尺度（裁剪前要先 unscale）
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     scaler.step(opt)                   # 内部检查 inf/nan，安全才更新
#     scaler.update()                    # 动态调整 scale 因子

if torch.cuda.is_available():
    print("检测到 CUDA → 走真正的 fp16 autocast + GradScaler 路径")
    device = "cuda"
    m = make_model().to(device)
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x, y = make_data()
    x, y = x.to(device), y.to(device)
    scaler = torch.amp.GradScaler("cuda")   # torch 2.x 现代写法（旧 torch.cuda.amp.* 已弃用）
    with torch.autocast("cuda", dtype=torch.float16):
        loss = F.mse_loss(m(x), y)
    o.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(o)
    scaler.update()
    print(f"  AMP 一步完成，loss = {loss.item():.5f}")
else:
    print("当前是 CPU（torch 2.12.0+cpu，无 GPU）→ 跳过真实 fp16/GradScaler。")
    print("CPU 上 AMP 行为说明：")
    print("  * GradScaler 主要服务 GPU 上的 fp16，CPU 上用处不大（也无需 scaler）。")
    print("  * CPU 仍可用 torch.autocast('cpu', dtype=torch.bfloat16) 跑 bf16，")
    print("    但 CPU 收益小，这里只演示 API 形态，不依赖它出数。")
    # 演示 CPU bf16 autocast 的 API 形态（能跑，但不是性能重点）
    m = make_model()
    x, y = make_data()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(x)
    print(f"  CPU autocast(bf16) 前向输出 dtype = {out.dtype}（autocast 内算子降精度）")
    print("  → 真正的提速/省显存收益在 GPU 上的 fp16/bf16，到 phase3 上卡再体会。")


# ============================================================
section("小结")
# ============================================================
print("""
本课关键结论：
  1. 训练五步曲 forward→loss→zero_grad→backward→step；评估必须 model.eval() +
     torch.no_grad()，且评估完记得恢复 model.train()，别污染后续训练。
  2. 复现性 = torch.manual_seed 固定全局随机源 + 给随机处传固定 generator
     （DataLoader shuffle 同理），才能「同种子同结果」。
  3. 梯度裁剪 clip_grad_norm_ 防爆炸（返回裁剪前总范数，LLM 常用 max_norm=1.0）；
     梯度累积撑大有效 batch，记住 micro loss 要除以 accum_steps。
  4. checkpoint 要存 model + optimizer + scheduler + step 四件套，断电续训才不丢
     动量与 lr 进度；用 map_location 处理跨设备加载。
  5. early stopping 用 patience 监控 val loss，连续不改善就停，省算力又防过拟合；
     AMP（autocast + GradScaler）在 GPU 上用 fp16/bf16 省显存提速，CPU 上收益小。

下一课：调试 / Profile / 显存 —— 看懂报错、查 NaN、估显存与算力。
""")
