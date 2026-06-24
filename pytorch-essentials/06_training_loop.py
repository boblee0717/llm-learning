"""
第 6 课：工程化训练循环 ⭐
=========================
前 5 课的零件（tensor / autograd / Module / 优化器 / DataLoader）在这里
组装成一个真正能用的训练脚本。这节课的内容几乎就是 phase3 第 1 课
（training pipeline）的预习——学完后那一课你会觉得「全是熟面孔」。

核心问题：
- 怎么让训练可复现（seed / determinism）？
- 一个健壮的训练 step 除了五步曲还要做什么？（梯度裁剪）
- 怎么正确做验证（eval 模式 + no_grad + 平均 loss）？
- checkpoint 要存哪些东西才能「断点续训」？（不只是模型权重！）
- 早停（early stopping）和梯度累积怎么实现？
- 混合精度（AMP）是什么，CPU / GPU 上分别怎么用？

与大模型的关系：
- 这就是预训练 / 微调 LLM 的训练主循环骨架。断点续训对动辄训练几天的大模型是刚需；
  AMP 和梯度累积是在有限显存上训大模型的标配。

前置：本专项第 1~5 课
"""

import sys
import os
import tempfile

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
section("Part 1: 可复现性 —— 固定随机种子")
# ============================================================
# 训练涉及多个随机源：Python random、numpy、torch（CPU/GPU）。
# 想复现实验（同样的初始化、同样的 shuffle 顺序），要把它们一起固定。

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # 即使没 GPU 调用也安全（无操作）


set_seed(0)
a = torch.randn(3)
set_seed(0)
b = torch.randn(3)
print("两次固定同样 seed 后结果相同:", torch.equal(a, b))
print("→ 完全确定性还需 torch.use_deterministic_algorithms(True) + 环境变量，"
      "但通常 set_seed 已够复现实验")


# ============================================================
section("Part 2: 数据与模型")
# ============================================================
# 造一个回归任务：y = sin(3x) + 噪声，用 MLP 拟合。

set_seed(42)
N = 1000
X = torch.linspace(-3, 3, N).unsqueeze(1)
Y = torch.sin(3 * X) + 0.1 * torch.randn(N, 1)

n_val = int(N * 0.2)
perm = torch.randperm(N)
train_idx, val_idx = perm[n_val:], perm[:n_val]
train_loader = DataLoader(TensorDataset(X[train_idx], Y[train_idx]),
                          batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[val_idx], Y[val_idx]),
                        batch_size=64, shuffle=False)


def make_model():
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    ).to(DEVICE)


print(f"device={DEVICE}，训练 {len(train_idx)} 条，验证 {len(val_idx)} 条")


# ============================================================
section("Part 3: 健壮的训练 step + 梯度裁剪")
# ============================================================
# 五步曲之外加一步梯度裁剪：把梯度的整体范数限制在 max_norm 内，防止偶发的
# 梯度爆炸把参数冲飞（训练 LLM / RNN 几乎必加）。要放在 backward 之后、step 之前。

def train_step(model, xb, yb, opt, max_norm=1.0):
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    pred = model(xb)
    loss = F.mse_loss(pred, yb)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # 梯度裁剪
    opt.step()
    return loss.item()


# ============================================================
section("Part 4: 验证 —— eval 模式 + no_grad + 平均 loss")
# ============================================================
# 验证要点：model.eval()（关 Dropout/BN）、torch.no_grad()（省内存、不建图）、
# 按样本数加权平均（最后一个 batch 可能更小）。

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss = F.mse_loss(model(xb), yb, reduction="sum")  # 求和，最后再除总数
        total_loss += loss.item()
        total_n += xb.shape[0]
    model.train()
    return total_loss / total_n


# ============================================================
section("Part 5: checkpoint —— 断点续训要存哪些东西")
# ============================================================
# 常见误区：只存 model.state_dict()。要真正「续训」，还得存优化器状态
#（Adam 的动量 m/v）、当前 step/epoch、最好的指标，甚至 RNG 状态。
# 否则续训等于「换了个优化器从头预热」，曲线会有个台阶。

def save_checkpoint(path, model, opt, step, best_val):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),   # ← 关键：别漏了优化器状态
        "step": step,
        "best_val": best_val,
    }, path)


def load_checkpoint(path, model, opt):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    return ckpt["step"], ckpt["best_val"]


# 演示：训一半 → 存 → 模拟崩溃 → 新建模型加载 → 接着训，验证「无缝衔接」
set_seed(1)
model = make_model()
opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

ckpt_dir = tempfile.mkdtemp()
ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")

step = 0
for epoch in range(10):
    for xb, yb in train_loader:
        train_step(model, xb, yb, opt)
        step += 1
val_mid = evaluate(model, val_loader)
save_checkpoint(ckpt_path, model, opt, step, val_mid)
print(f"训练 10 epoch 后 step={step}，val={val_mid:.4f}，已存 checkpoint")

# —— 模拟程序崩溃重启 ——
del model, opt
model = make_model()                       # 全新随机初始化
opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
val_fresh = evaluate(model, val_loader)
print(f"新建模型（未加载）val={val_fresh:.4f}（随机水平，明显更差）")

resume_step, best_val = load_checkpoint(ckpt_path, model, opt)
val_resumed = evaluate(model, val_loader)
print(f"加载 checkpoint 后 step={resume_step}，val={val_resumed:.4f}（恢复到崩溃前水平）")
print("→ 存了 optimizer 状态，续训才能无缝衔接，不会出现 loss 台阶")


# ============================================================
section("Part 6: 早停（early stopping）+ 完整训练")
# ============================================================
# 早停：验证集 loss 连续 patience 个 epoch 没改善就停，避免过拟合 + 省时间。
# 同时只保存「目前最好」的权重（best checkpoint）。

set_seed(2)
model = make_model()
opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

best_val = float("inf")
patience, bad_epochs = 5, 0
best_path = os.path.join(ckpt_dir, "best.pt")

for epoch in range(100):
    for xb, yb in train_loader:
        train_step(model, xb, yb, opt)
    val = evaluate(model, val_loader)
    if val < best_val - 1e-5:
        best_val = val
        bad_epochs = 0
        torch.save(model.state_dict(), best_path)   # 只存最好的
    else:
        bad_epochs += 1
    if epoch % 10 == 0 or bad_epochs >= patience:
        print(f"  epoch {epoch:>3d}  val={val:.5f}  best={best_val:.5f}  bad={bad_epochs}")
    if bad_epochs >= patience:
        print(f"  → 连续 {patience} 个 epoch 没改善，早停于 epoch {epoch}")
        break

print(f"训练结束，最佳 val={best_val:.5f}")


# ============================================================
section("Part 7: 梯度累积 —— 用小显存模拟大 batch")
# ============================================================
# 显存装不下大 batch 时：把一个大 batch 拆成 accum_steps 个小 batch，
# 各自 backward（梯度累加），累够了再 step 一次 + 清零。
# 等效 batch_size = micro_batch * accum_steps。loss 要除以 accum_steps 保持尺度一致。

set_seed(3)
model = make_model()
opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
accum_steps = 4

micro_batches = [next(iter(train_loader)) for _ in range(accum_steps)]
opt.zero_grad()
for i, (xb, yb) in enumerate(micro_batches):
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    loss = F.mse_loss(model(xb), yb) / accum_steps   # 除以累积步数
    loss.backward()                                   # 梯度累加（第 2 课讲过累加特性）
print(f"累积了 {accum_steps} 个 micro-batch 的梯度后，step 一次")
opt.step()
opt.zero_grad()
print("→ 等效 batch_size = micro_batch × accum_steps，显存只占一个 micro-batch")
print("   （对照 papers/notes 的 critical batch size：目标是凑到 B_crit，不是越大越好）")


# ============================================================
section("Part 8: 混合精度 AMP —— CPU / GPU 的差异")
# ============================================================
# AMP（autocast）让前向用低精度（GPU 上 float16、CPU 上 bfloat16）算，
# 省显存、提速；GradScaler 在 GPU float16 下放大 loss 防止梯度下溢（CPU 不需要）。

print("AMP 标准写法（GPU float16）:")
print("""
    scaler = torch.amp.GradScaler('cuda')
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = F.mse_loss(model(xb), yb)
    opt.zero_grad()
    scaler.scale(loss).backward()   # 放大 loss 再 backward
    scaler.step(opt)                # 内部 unscale 后再 step
    scaler.update()                 # 调整缩放因子
""")

# 本机用 CPU + bfloat16 演示 autocast（CPU 上 bfloat16 不需要 GradScaler）
set_seed(4)
model = make_model()
opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
xb, yb = next(iter(train_loader))
xb, yb = xb.to(DEVICE), yb.to(DEVICE)
with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
    pred = model(xb)
    loss = F.mse_loss(pred, yb)
print(f"autocast 内前向输出 dtype: {pred.dtype}（bfloat16，省内存）")
opt.zero_grad()
loss.backward()       # CPU bfloat16 不需要 GradScaler
opt.step()
print(f"反向 + 更新完成，loss={loss.item():.4f}")


# ============================================================
section("小结")
# ============================================================
print("""
本课关键结论：
  1. set_seed 固定 random/numpy/torch 三个随机源，让实验可复现。
  2. 训练 step 在五步曲基础上加梯度裁剪（clip_grad_norm_）防梯度爆炸。
  3. 验证三件套：model.eval() + torch.no_grad() + 按样本数加权平均 loss。
  4. checkpoint 要存 model + optimizer + step + best_val，才能无缝断点续训。
  5. 早停：val 连续 patience 个 epoch 不改善就停，并只保存 best 权重。
  6. 梯度累积：拆 micro-batch 累加梯度再 step，loss 除以 accum_steps；等效大 batch。
  7. AMP：GPU float16 配 GradScaler，CPU bfloat16 不用 scaler；都用 autocast 包前向。

→ 这套骨架就是 phase3 第 1 课的 training pipeline。下一课收尾：调试、性能与显存。
""")
