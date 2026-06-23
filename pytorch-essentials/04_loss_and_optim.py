"""
第 4 课：损失函数与优化器（torch 版）
=====================================
phase1 第 2/4 课你用 NumPy 手写过 MSE、cross entropy，也从零实现了
SGD / Momentum / Adam / AdamW。这节课把它们换成 PyTorch 的标准件：
nn 里的损失函数、torch.optim 里的优化器，再补上训练大模型必备的
两件事——param groups（哪些参数不做 weight decay）和 lr 调度
（warmup + cosine，GPT 训练同款）。

核心问题：
- F.cross_entropy 到底等于什么？为什么它直接吃 logits，不用先 softmax？
- torch.optim 怎么用？optimizer.step() / zero_grad() 替你做了 phase1 手写的哪几步？
- 为什么 bias 和 LayerNorm 参数通常【不做】weight decay？怎么用 param groups 实现？
- warmup + cosine 学习率曲线长什么样？为什么要 warmup？

与大模型的关系：
- 几乎所有 LLM 预训练都用 AdamW + warmup + cosine decay；
  param groups 区分 decay/no-decay 是 GPT-2/nanoGPT 的标准写法。

前置：phase1 第 2/4 课（损失 / 优化器），本专项第 2/3 课（autograd / Module）
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # 无界面后端，直接存图不弹窗
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 图片锚定脚本自身目录


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
section("Part 1: CrossEntropyLoss —— 直接吃 logits")
# ============================================================
# F.cross_entropy(logits, target) = NLLLoss(log_softmax(logits), target)
# 即：先对 logits 做 log_softmax，再取目标类的负对数似然，最后平均。
# 它【内部】帮你做了 softmax，所以模型最后一层只输出 raw logits，不要自己 softmax！
#（自己 softmax 再喂进去会算两次，数值还不稳定。）

logits = torch.tensor([[2.0, 0.5, 0.1],     # 样本 0，真实类别 0
                       [0.1, 0.2, 3.0]])     # 样本 1，真实类别 2
target = torch.tensor([0, 2])

loss_torch = F.cross_entropy(logits, target)

# 手写对拍：log_softmax 后取目标类、取负、平均
log_probs = F.log_softmax(logits, dim=-1)
loss_manual = -(log_probs[0, 0] + log_probs[1, 2]) / 2

print(f"F.cross_entropy   = {loss_torch.item():.6f}")
print(f"手写 NLL(logsoftmax) = {loss_manual.item():.6f}")
print("两者一致:", torch.allclose(loss_torch, loss_manual))

# ignore_index：序列任务里用来跳过 padding 位置（target 设成 -100 默认被忽略）
target_with_pad = torch.tensor([0, -100])    # 第 2 个样本是 padding，不计入 loss
loss_ignore = F.cross_entropy(logits, target_with_pad, ignore_index=-100)
print(f"\nignore_index=-100 跳过 padding 后的 loss = {loss_ignore.item():.6f}"
      f"（只剩样本 0：{-log_probs[0,0].item():.6f}）")


# ============================================================
section("Part 2: torch.optim —— 把 phase1 手写优化器换成标准件")
# ============================================================
# phase1 你手写过：param -= lr * (动量/自适应处理后的梯度)，还要手动 zero。
# torch.optim 把这些封装好了。训练五步曲：
#   pred = model(x)        # 前向
#   loss = loss_fn(...)    # 算损失
#   optimizer.zero_grad()  # 清空上一步的梯度（对应 phase1 的手动清零）
#   loss.backward()        # 反向求梯度（autograd）
#   optimizer.step()       # 按优化算法更新参数（对应 phase1 手写的更新公式）

torch.manual_seed(0)
model = nn.Linear(4, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.randn(16, 4)
y = torch.randn(16, 1)

print("用 SGD 训练 5 步：")
for step in range(5):
    pred = model(x)
    loss = F.mse_loss(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"  step {step}  loss={loss.item():.4f}")

# AdamW 是 LLM 训练的默认选择（Adam + 解耦权重衰减，phase1 第 4 课讲过原理）
print("\n常用优化器一览：")
print("  SGD(momentum=0.9)  —— 经典，配好 lr/schedule 也能很强")
print("  Adam               —— 自适应学习率，省心")
print("  AdamW              —— Adam + 解耦 weight decay，LLM 预训练默认")


# ============================================================
section("Part 3: param groups —— 哪些参数不做 weight decay")
# ============================================================
# weight decay（L2 正则）通常只加在【矩阵权重】上，不加在 bias 和 LayerNorm 上
# （这些一维参数做 decay 反而有害）。用 param groups 给不同参数设不同超参。

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.ln = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(self.ln(torch.relu(self.fc1(x))))


net = TinyNet()
decay, no_decay = [], []
for name, p in net.named_parameters():
    if p.ndim >= 2:           # 矩阵权重（Linear.weight 等）→ 做 decay
        decay.append(name)
    else:                     # 一维参数（bias、LayerNorm 的 γ/β）→ 不做 decay
        no_decay.append(name)

print("做 weight decay 的参数:", decay)
print("不做 weight decay 的参数:", no_decay)

# 用这两组构造优化器：第一组 wd=0.1，第二组 wd=0
param_groups = [
    {"params": [p for n, p in net.named_parameters() if p.ndim >= 2], "weight_decay": 0.1},
    {"params": [p for n, p in net.named_parameters() if p.ndim < 2], "weight_decay": 0.0},
]
opt2 = torch.optim.AdamW(param_groups, lr=1e-3)
print(f"\nAdamW 两组 param groups：组0 wd={opt2.param_groups[0]['weight_decay']}，"
      f"组1 wd={opt2.param_groups[1]['weight_decay']}")
print("→ 这正是 nanoGPT / GPT-2 训练里区分 decay/no-decay 的标准写法")


# ============================================================
section("Part 4: 学习率调度 —— warmup + cosine（GPT 同款）")
# ============================================================
# 几乎所有 LLM 都用这条曲线：
#   warmup 阶段：lr 从 0 线性升到峰值（让初期不稳定的梯度别一上来就炸）
#   cosine 阶段：lr 按余弦曲线平滑降到接近 0（后期小步精调）
# 我们手写这条 lr 倍率函数，再用 LambdaLR 套上去。

def lr_lambda(step, warmup_steps, total_steps, min_ratio=0.1):
    if step < warmup_steps:                       # 线性 warmup
        return step / max(1, warmup_steps)
    # cosine 衰减到 min_ratio
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_ratio + (1 - min_ratio) * cosine


total_steps, warmup_steps, base_lr = 1000, 100, 3e-4
dummy = torch.nn.Parameter(torch.zeros(1))
opt3 = torch.optim.AdamW([dummy], lr=base_lr)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt3, lr_lambda=lambda s: lr_lambda(s, warmup_steps, total_steps)
)

lrs = []
for step in range(total_steps):
    lrs.append(opt3.param_groups[0]["lr"])
    opt3.step()       # 这里没真训练，只是推动 scheduler
    sched.step()

print(f"峰值 lr = {max(lrs):.2e}（在 warmup 结束处），末值 lr = {lrs[-1]:.2e}")

plt.figure(figsize=(8, 4))
plt.plot(lrs)
plt.axvline(warmup_steps, color="r", ls="--", label=f"warmup 结束 (step {warmup_steps})")
plt.xlabel("step"); plt.ylabel("learning rate")
plt.title("Warmup + Cosine LR Schedule")
plt.legend(); plt.grid(alpha=0.3)
out_path = os.path.join(SCRIPT_DIR, "lr_schedule.png")
plt.savefig(out_path, dpi=100, bbox_inches="tight")
plt.close()
print(f"学习率曲线已保存: {out_path}")


# ============================================================
section("Part 5: 完整训练 —— optim + scheduler 串起来")
# ============================================================
# 一个能跑的小训练：拟合 y = sin(x)，用 AdamW + warmup/cosine。

torch.manual_seed(1)
xs = torch.linspace(-3, 3, 200).unsqueeze(1)
ys = torch.sin(xs)

reg = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
opt = torch.optim.AdamW(reg.parameters(), lr=1e-2, weight_decay=1e-4)
T = 500
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lr_lambda=lambda s: lr_lambda(s, warmup_steps=50, total_steps=T)
)

for step in range(T):
    pred = reg(xs)
    loss = F.mse_loss(pred, ys)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 100 == 0 or step == T - 1:
        print(f"  step {step:>3d}  loss={loss.item():.5f}  lr={opt.param_groups[0]['lr']:.2e}")

print(f"\n最终拟合 sin(x) 的 MSE = {loss.item():.5f}")


# ============================================================
section("小结")
# ============================================================
print("""
本课关键结论：
  1. F.cross_entropy = NLLLoss(log_softmax(logits))，直接吃 logits，模型别自己 softmax。
     ignore_index（默认 -100）用来跳过 padding 位置。
  2. 训练五步曲：forward → loss → zero_grad → backward → step。
     optimizer.step() 替你做了 phase1 手写的参数更新公式。
  3. AdamW 是 LLM 默认优化器；weight decay 只加在矩阵权重上，bias/LayerNorm 用 param group 排除。
  4. warmup + cosine 是 LLM 标准 lr 曲线：先线性升（稳住初期），再余弦降（后期精调）。
  5. 完整训练 = 优化器 + 调度器 + 五步曲，每步 step 后记得 sched.step()。

下一课：Dataset / DataLoader —— 把数据喂给模型的标准管道。
""")
