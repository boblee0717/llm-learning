"""
第 2 课：Autograd 自动微分 ⭐
============================
这是 PyTorch 的灵魂。你在 phase1 用 NumPy 一行行手写过反向传播——
求 loss 对每个参数的偏导。PyTorch 的 autograd 帮你把这件事【自动】做了：
你只管写前向（forward），它在背后记录一张计算图，调用 .backward() 时
沿图反向用链式法则求出所有梯度。

本课的目标不是「会调 .backward()」，而是确认一件事：
**autograd 算出来的梯度，和你 phase1 手写的解析梯度，逐位相等。**
一旦你亲眼看到它们对拍一致，autograd 就不再是黑盒。

核心问题：
- requires_grad / 计算图 / 叶子张量 / .grad 到底是什么关系？
- .backward() 在做什么？为什么梯度会「累加」而不是覆盖？（所以要 zero_grad）
- detach() 和 torch.no_grad() 有什么区别，分别什么时候用？
- 怎么用有限差分（数值梯度）验证 autograd 算对了？

与大模型的关系：
- 训练循环里的 loss.backward() 就是这套机制。理解它，才能 debug 梯度爆炸 /
  消失 / nan、才能看懂梯度裁剪、梯度累积、detach 在 RL/蒸馏里的用法。

前置：phase1 第 2/3 课（梯度下降、手写反向传播）
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码

import torch


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
section("Part 1: requires_grad、计算图、叶子张量")
# ============================================================
# requires_grad=True 的张量会被 autograd 追踪。由它参与运算得到的新张量，
# 会记住「我是怎么算出来的」（grad_fn），从而连成一张计算图。

x = torch.tensor([2.0], requires_grad=True)   # 叶子张量（我们要对它求导）
y = x ** 2 + 3 * x + 1                          # y = x² + 3x + 1
print("x:", x, "→ 叶子张量 is_leaf =", x.is_leaf)
print("y:", y, "→ y 记得自己是怎么算的 grad_fn =", y.grad_fn)

# 手算导数：dy/dx = 2x + 3，在 x=2 处 = 7
y.backward()                                    # 反向传播，把梯度填进各叶子的 .grad
print("autograd 求得 dy/dx =", x.grad.item(), "（手算 2x+3 = 7）")


# ============================================================
section("Part 2: 多元函数的梯度 —— 和 phase1 手写对拍")
# ============================================================
# 线性层 + MSE：loss = sum((W·x - target)²)
# phase1 手写过它的梯度：d_loss/d_W = 2 * (W·x - target) ⊗ x（外积）。
# 这里让 autograd 算，再和手写解析式对比。

torch.manual_seed(0)
W = torch.randn(3, 4, requires_grad=True)   # (out=3, in=4)
xin = torch.randn(4)                         # 输入 (4,)
target = torch.randn(3)                      # 目标 (3,)

pred = W @ xin                               # (3,)
loss = ((pred - target) ** 2).sum()
loss.backward()

# 手写解析梯度：d_loss/d_W[i,j] = 2*(pred_i - target_i) * x_j  → 外积
grad_manual = 2 * (pred - target).unsqueeze(1) * xin.unsqueeze(0)  # (3,4)
print("autograd 梯度 W.grad:\n", W.grad)
print("\n手写解析梯度:\n", grad_manual.detach())
print("\n两者是否逐位相等:", torch.allclose(W.grad, grad_manual.detach()))


# ============================================================
section("Part 3: 梯度会累加 —— 这就是要 zero_grad 的原因")
# ============================================================
# .backward() 是把新算的梯度【加】到 .grad 上，不是覆盖。
# 好处：支持梯度累积（phase3 会用）。代价：每个训练 step 前必须清零，否则梯度会越滚越大。

p = torch.tensor([1.0], requires_grad=True)
(p * 2).backward()
print("第 1 次 backward 后 grad:", p.grad.item())   # 2
(p * 2).backward()
print("第 2 次 backward 后 grad:", p.grad.item(), "← 累加成 4，不是覆盖")

p.grad.zero_()                                       # 清零
(p * 2).backward()
print("zero_() 之后再 backward:", p.grad.item(), "← 回到 2")


# ============================================================
section("Part 4: detach 与 no_grad —— 怎么「关掉」追踪")
# ============================================================
# 两种常见需求：
#  (a) 推理 / 评估时不需要梯度，想省内存又加速 → torch.no_grad() 包住整段。
#  (b) 想把某个张量当成「常数」截断梯度回传 → detach()（RLHF/蒸馏里常见）。

a = torch.tensor([3.0], requires_grad=True)

with torch.no_grad():
    b = a * 2                          # 这段里所有运算都不建图
print("no_grad 内的结果 requires_grad:", b.requires_grad, "（不再追踪）")

c = a * 2
c_detached = c.detach()               # 摘下来，得到不带梯度、共享数据的张量
print("detach 后 requires_grad:", c_detached.requires_grad)
print("→ no_grad 管「一整段」，detach 管「单个张量」")


# ============================================================
section("Part 5: 用有限差分验证 autograd（数值梯度校验）")
# ============================================================
# phase0 第 4 课用过这招：把参数偷偷加减一个很小的 eps，用 (f(x+e)-f(x-e))/(2e)
# 近似导数，和 autograd 的结果对比。这是验证「梯度有没有写对」的黄金方法。

def f(t):
    return (t ** 3 + 2 * t).sum()      # f = Σ(t³ + 2t)，f' = 3t² + 2

t = torch.randn(5, requires_grad=True)
f(t).backward()
analytic = t.grad.clone()

eps = 1e-4
numeric = torch.zeros_like(t)
with torch.no_grad():
    for i in range(t.numel()):
        plus = t.clone();  plus[i] += eps
        minus = t.clone(); minus[i] -= eps
        numeric[i] = (f(plus) - f(minus)) / (2 * eps)

print("autograd 梯度:", analytic)
print("数值梯度    :", numeric)
print("最大差异:", (analytic - numeric).abs().max().item(), "（应非常接近 0）")


# ============================================================
section("Part 6: 用 autograd 跑一个从零的线性回归（无 optimizer）")
# ============================================================
# 把前 5 部分串起来：纯手写训练循环，但梯度交给 autograd。
# 注意参数更新要放在 no_grad 里（更新这一步本身不该被建图），并手动 zero。

torch.manual_seed(1)
true_W, true_b = 2.5, -1.0
xs = torch.linspace(-3, 3, 100)
ys = true_W * xs + true_b + 0.1 * torch.randn(100)   # 带噪声的直线

w = torch.zeros(1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)
lr = 0.05

for step in range(200):
    pred = w * xs + bias
    loss = ((pred - ys) ** 2).mean()
    loss.backward()
    with torch.no_grad():          # 更新参数不建图
        w -= lr * w.grad
        bias -= lr * bias.grad
        w.grad.zero_()             # 手动清零（下一课起交给 optimizer.zero_grad）
        bias.grad.zero_()
    if step % 50 == 0 or step == 199:
        print(f"  step {step:>3d}  loss={loss.item():.4f}  "
              f"w={w.item():.3f}  b={bias.item():.3f}")

print(f"\n拟合结果 w={w.item():.3f} (真值 {true_W})，b={bias.item():.3f} (真值 {true_b})")


# ============================================================
section("小结")
# ============================================================
print("""
本课关键结论：
  1. requires_grad=True 的叶子张量被追踪；前向时自动建计算图（grad_fn 记录来路）。
  2. loss.backward() 沿图用链式法则求导，把梯度填进各叶子的 .grad。
  3. 梯度是【累加】的 → 每个 step 前必须 zero_grad（也正因如此能做梯度累积）。
  4. no_grad 关掉「一整段」的追踪（推理/更新参数）；detach 截断「单个张量」（当常数）。
  5. autograd 的梯度 == 你 phase1 手写的解析梯度 == 有限差分数值梯度，三者对拍一致。

→ 你以前手写反向传播是为了「懂」；以后用 autograd 是为了「快」。两者算的是同一件事。
下一课：nn.Module —— 把参数、前向、状态打包成可复用、可保存的模块。
""")
