"""
======================================================
第 4 课：优化器 —— 比朴素 SGD 更聪明的"下山方式"
======================================================

承上启下：
  第 2 课我们学了最朴素的更新规则 `w = w - lr * grad`（SGD）。
  它能用，但在真实训练里有两个毛病，于是有了 Momentum / RMSprop / Adam / Adafactor。
  这一课就把这些"优化器"从零用 NumPy 实现一遍，搞懂它们各自在解决什么问题。

核心问题：
  既然梯度已经告诉了方向，"更新规则"还能怎么改进？
  答：① 利用历史梯度积累"惯性"（动量）；
      ② 给每个参数单独定步长（自适应学习率）；
      ③ 二者结合 + 偏差修正 = Adam；
      ④ 当模型大到优化器状态都存不下时，用 Adafactor 把状态压缩。

学习目标：
1. 理解 SGD 的两个毛病（各向异性 + 震荡/缓慢）
2. 手写 Momentum（动量）
3. 手写 RMSprop（自适应学习率）
4. 手写 Adam（动量 + 自适应 + 偏差修正）与 AdamW（解耦权重衰减）
5. 理解优化器的"内存代价"，以及为什么最大的模型改用 Adafactor

运行方式：python3 04_optimizers.py
"""

import sys

# 作者在 Windows / PowerShell 环境，重配置 stdout 防中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 实验场景：一个"各向异性"的碗（ill-conditioned loss）
# ============================================================
#
# 我们要最小化的损失：
#   L(w) = 0.5 * sum( curvature * (w - target)^2 )
#
# 其中 curvature = [1, 100]：第一个方向很平缓、第二个方向很陡（曲率差 100 倍）。
# 梯度：grad = curvature * (w - target)
#
# 为什么用这个例子？
#   真实神经网络的 loss 地形几乎都是"各向异性"的——有的方向陡、有的方向平。
#   这个 2 维碗是最小的能复现该现象的玩具，能清楚看出各优化器的差异。
#   两个方向曲率差得越大（条件数越大），朴素 SGD 就越吃亏。

CURVATURE = np.array([1.0, 100.0])  # 条件数 = 100，刻意做得病态一点
TARGET = np.array([0.0, 0.0])       # 最小值在原点
START = np.array([-5.0, -5.0])      # 统一的起点
STEPS = 300                         # 统一的迭代步数


def loss_fn(w):
    return 0.5 * np.sum(CURVATURE * (w - TARGET) ** 2)


def grad_fn(w):
    return CURVATURE * (w - TARGET)


def steps_to_converge(history, thresh=1e-3):
    for i, l in enumerate(history):
        if l < thresh:
            return i
    return None


def report(name, lr, history):
    n = steps_to_converge(history)
    n_str = f"{n} 步收敛" if n is not None else f">{STEPS} 步(未达 1e-3)"
    print(f"{name} (lr={lr})  {STEPS} 步后 loss = {history[-1]:.3e}   ({n_str})")


# ============================================================
# 第一部分：回顾 SGD —— 以及它的两个毛病
# ============================================================

section("第一部分：朴素 SGD 的两个毛病")

print("""
更新规则（第 2 课学的）：  w = w - lr * grad

毛病 1：各方向曲率不同时，一个学习率两头不讨好。
  - lr 调大 → 在陡方向（curvature=100）上来回震荡甚至发散；
  - lr 调小 → 在平方向（curvature=1）上挪得太慢。
毛病 2：在狭长山谷里会"之"字形反复横跳，收敛很慢。
""")


def run_sgd(lr, steps=STEPS):
    w = START.copy()
    history = [loss_fn(w)]
    for _ in range(steps):
        g = grad_fn(w)
        w = w - lr * g          # ← 第 2 课的核心公式
        history.append(loss_fn(w))
    return w, history


# lr 必须 < 2/max_curvature = 2/100 = 0.02 才稳定；这里取 0.018 已接近上限
_, hist_sgd = run_sgd(lr=0.018)
report("SGD     ", 0.018, hist_sgd)
print("  → 陡方向早就到位，但平方向只能以 (1-0.018) 的龟速收缩，被'最慢的方向'拖累。")

# 想把 lr 调大去加速平方向 → 陡方向直接炸了（0.025 > 稳定上限 0.02）
_, hist_div = run_sgd(lr=0.025)
print(f"SGD      (lr=0.025)  {STEPS} 步后 loss = {hist_div[-1]:.3e}   ← 发散了（陡方向震荡放大）")
print()


# ============================================================
# 第二部分：Momentum（动量）—— 给下山加"惯性"
# ============================================================

section("第二部分：Momentum 动量")

print("""
思路：不要只看当前这一步的梯度，把历史梯度也累积进来，像下坡的小球带惯性。

公式（PyTorch SGD 的 momentum 约定）：
  v = beta * v + grad          # 速度 = 衰减的历史速度 + 当前梯度
  w = w - lr * v               # 用速度而不是裸梯度去更新

效果：
  - 在持续同向的平方向上，速度不断累积 → 越走越快（加速收敛）；
  - 在来回正负的陡方向上，正负梯度相互抵消 → 抑制震荡。
beta 常取 0.9，相当于"记住"最近约 10 步的梯度。
""")


def run_momentum(lr, beta=0.9, steps=STEPS):
    w = START.copy()
    v = np.zeros_like(w)
    history = [loss_fn(w)]
    for _ in range(steps):
        g = grad_fn(w)
        v = beta * v + g
        w = w - lr * v
        history.append(loss_fn(w))
    return w, history


_, hist_mom = run_momentum(lr=0.004, beta=0.9)
report("Momentum", 0.004, hist_mom)
print("  → 即使 lr 比 SGD 还小，靠惯性在平方向上也远远追了上来。")
print()


# ============================================================
# 第三部分：RMSprop —— 给每个参数单独定步长（自适应学习率）
# ============================================================

section("第三部分：RMSprop 自适应学习率")

print("""
思路：动量改的是"方向"，RMSprop 改的是"每个参数各自的步长"。
  谁的梯度一直很大，就把谁的有效步长调小；谁很久没动，就放大。

公式：
  s = beta * s + (1 - beta) * grad^2        # 梯度平方的滑动平均（每个参数各一份）
  w = w - lr * grad / (sqrt(s) + eps)        # 除以 sqrt(s) 做归一化

关键直觉：
  陡方向梯度大 → s 大 → 除以大数 → 步子自动变小（不再震荡）；
  平方向梯度小 → s 小 → 步子相对变大（不再龟速）。
  eps（如 1e-8）防止除以 0。

注意：RMSprop 单独用在"无噪声的确定性问题"上对学习率比较敏感
  （它本是为带噪声的在线/小批量训练设计的）。这里它能收敛，
  但更重要的是把它理解成 Adam 的"自适应步长"组件。
""")


def run_rmsprop(lr, beta=0.9, eps=1e-8, steps=STEPS):
    w = START.copy()
    s = np.zeros_like(w)
    history = [loss_fn(w)]
    for _ in range(steps):
        g = grad_fn(w)
        s = beta * s + (1 - beta) * g ** 2
        w = w - lr * g / (np.sqrt(s) + eps)
        history.append(loss_fn(w))
    return w, history


_, hist_rms = run_rmsprop(lr=0.02, beta=0.9)
report("RMSprop ", 0.02, hist_rms)
print("  → 两个方向被'归一化'到相近的步长，但靠纯归一化没有动量，收得不算快。")
print()


# ============================================================
# 第四部分：Adam = Momentum + RMSprop + 偏差修正
# ============================================================

section("第四部分：Adam（现代训练的默认选择）")

print("""
Adam 把前两者合在一起：既用动量(m)平滑方向，又用二阶矩(v)归一化步长。

公式：
  m = beta1 * m + (1 - beta1) * grad        # 一阶矩（动量）
  v = beta2 * v + (1 - beta2) * grad^2      # 二阶矩（RMSprop）
  m_hat = m / (1 - beta1^t)                 # 偏差修正（t 是第几步，从 1 开始）
  v_hat = v / (1 - beta2^t)
  w = w - lr * m_hat / (sqrt(v_hat) + eps)

为什么要偏差修正？
  m、v 初始都是 0，前几步会被"拉"得偏小（偏向 0）。
  除以 (1 - beta^t)：t 小的时候这个分母很小，把估计放大补偿回来；
  t 变大后分母趋近 1，修正自动消失。
默认超参：beta1=0.9, beta2=0.999, eps=1e-8。
""")


def run_adam(lr, beta1=0.9, beta2=0.999, eps=1e-8, steps=STEPS):
    w = START.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    history = [loss_fn(w)]
    for t in range(1, steps + 1):
        g = grad_fn(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(loss_fn(w))
    return w, history


_, hist_adam = run_adam(lr=0.2)
report("Adam    ", 0.2, hist_adam)
print("  → 方向稳、步长齐，几乎不用为各向异性操心，这就是它成为默认选择的原因。")
print()


# ============================================================
# 第五部分：AdamW —— 解耦的权重衰减（你在 phase3 真正用的那个）
# ============================================================

section("第五部分：AdamW（解耦权重衰减）")

print("""
权重衰减(weight decay)：训练时让参数轻微地往 0 收缩，是一种正则化，防止过拟合。

传统做法 Adam + L2：把 wd*w 加进梯度里 → 但它会被 Adam 的 sqrt(v) 一起缩放，
  导致"梯度大的参数被衰减得更少"，正则化效果被扭曲。

AdamW 的做法：把权重衰减从梯度里"解耦"出来，直接作用在参数上：
  m, v, m_hat, v_hat 同 Adam
  w = w - lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * w )
                └── Adam 的自适应更新 ──┘   └── 独立的、不被缩放的衰减 ──┘

这就是 pytorch-essentials/06_training_loop.py 里
  torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
背后真正在做的事——现在你能解释那行黑盒了。
""")


def run_adamw(lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, steps=STEPS):
    w = START.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    history = [loss_fn(w)]
    for t in range(1, steps + 1):
        g = grad_fn(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * w)
        history.append(loss_fn(w))
    return w, history


_, hist_adamw = run_adamw(lr=0.2, weight_decay=0.01)
report("AdamW   ", 0.2, hist_adamw)
print("  → 这里没有过拟合可言，weight_decay 只是让 w 多收缩一点；它的价值在真实大模型上才显著。")
print()


# ============================================================
# 第六部分：横向对比 + 可视化
# ============================================================

section("第六部分：四种优化器收敛对比（同一个碗、同一个起点）")

print(f"{'优化器':<12}{'最终 loss':>14}{'到 loss<1e-3 的步数':>22}")
print("-" * 50)

for name, hist in [
    ("SGD", hist_sgd),
    ("Momentum", hist_mom),
    ("RMSprop", hist_rms),
    ("Adam", hist_adam),
]:
    n = steps_to_converge(hist)
    n_str = f"{n}" if n is not None else f">{STEPS}(未达到)"
    print(f"{name:<12}{hist[-1]:>14.2e}{n_str:>22}")
print()
print("结论：在病态(各向异性)地形上，朴素 SGD 被最慢的方向拖累、收敛最慢；")
print("      Momentum 靠惯性、Adam 靠'动量+自适应'都明显更快，且对超参更不挑。")
print()

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for name, hist in [
        ("SGD (lr=0.018)", hist_sgd),
        ("Momentum (lr=0.004)", hist_mom),
        ("RMSprop (lr=0.02)", hist_rms),
        ("Adam (lr=0.2)", hist_adam),
    ]:
        plt.plot(hist, label=name, linewidth=2)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss (log scale)")
    plt.title("Optimizers on an ill-conditioned bowl (condition number=100)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    out_path = "optimizers_convergence.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"已保存收敛曲线图：{out_path}")
except Exception as err:  # noqa: BLE001
    print(f"(跳过画图：{err})")
print()


# ============================================================
# 第七部分：内存代价，以及为什么最大模型用 Adafactor
# ============================================================

section("第七部分：优化器的内存代价 与 Adafactor")

print("""
优化器不是免费的——它要为"每个参数"额外存状态：
  - SGD：           0 份额外状态
  - SGD+Momentum：  1 份（速度 v）
  - Adam / AdamW：  2 份（一阶矩 m + 二阶矩 v）

训练时显存的经典拆解（混合精度 + Adam，常说的"每参数约 16 字节"）：
  fp16 权重 2  +  fp16 梯度 2  +  fp32 主权重 4  +  fp32 m 4  +  fp32 v 4  = 16 字节/参数
  → 其中优化器状态(m,v)就占了 8 字节，是权重本身的好几倍。

这正是"为什么训大模型这么吃显存"的核心原因之一，也直接关系到：
  - 第二阶段第 6 课 Scaling Law 的显存估算；
  - 第三阶段 LoRA：只训练极少量参数 → 优化器状态(m,v)随之暴跌，这是它省显存的关键。
""")


def adam_state_bytes(num_params, bytes_per_value=4):
    """Adam 的优化器状态(m + v)所占字节数（fp32）。"""
    return 2 * num_params * bytes_per_value


for n_params, label in [(1.25e8, "GPT-2 (125M)"), (7e9, "7B"), (7e10, "70B")]:
    gb = adam_state_bytes(n_params) / 1024 ** 3
    print(f"  {label:<14} 仅 Adam 状态(m+v, fp32) ≈ {gb:8.1f} GB")
print()

print("""
Adafactor 解决的就是这个内存问题：
  对一个 m×n 的权重矩阵，Adam 要存满 m×n 的二阶矩 v；
  Adafactor 把 v 用"行方向一个向量 + 列方向一个向量"近似（外积重构），
  存储从 O(m×n) 降到 O(m+n)；还可以省掉一阶动量 m。
  代价是估计更粗糙、需配套技巧（相对步长等），但对超大模型这点内存省得非常值。
  代表作：T5、PaLM 用的就是 Adafactor。
所以"最大模型用 Adafactor"本质是一句：
  显存预算被优化器状态卡死了，只能换一个更省内存的优化器。
""")


# ============================================================
# 总结：与你学过内容的对应关系
# ============================================================

section("总结：优化器在整条学习线里的位置")

print("""
┌───────────────────────────┬──────────────────────────────────────────┐
│ 本课从零实现的             │ 对应到你学过 / 将学的                       │
├───────────────────────────┼──────────────────────────────────────────┤
│ w = w - lr*grad (SGD)      │ 第 2 课梯度下降的核心公式                    │
│ Momentum 速度 v            │ "更聪明的梯度下降变种"（第 2 课结尾那句话）   │
│ Adam 的 m, v 状态          │ phase3 checkpoint 里存的 optimizer.state_dict│
│ AdamW 解耦权重衰减         │ phase3 torch.optim.AdamW(..., weight_decay)  │
│ 优化器状态的内存代价       │ Scaling Law 显存估算 / LoRA 省显存的原理     │
│ Adafactor 状态压缩         │ 超大模型训练的工程权衡                        │
└───────────────────────────┴──────────────────────────────────────────┘

下一步：打开 04_optimizers_self_write.py，亲手把这些优化器写一遍。
""")

print("=" * 60)
print("恭喜完成第 4 课！现在你能解释 PyTorch 那行 AdamW 黑盒了。")
print("=" * 60)
