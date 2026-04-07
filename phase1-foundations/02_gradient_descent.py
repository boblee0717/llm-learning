"""
======================================================
第 2 课：梯度下降 —— 模型如何"学习"
======================================================

核心问题：模型怎么从"一无所知"变成"能回答问题"？
答案：通过不断调整参数，让预测误差越来越小。这个过程就是梯度下降。

类比：
  想象你站在山上，蒙着眼睛要走到山谷最低点。
  你能做的就是：摸一下脚下的坡度（梯度），然后往下坡方向走一步。
  重复这个过程，你就能逐渐到达谷底（最优解）。

学习目标：
1. 理解损失函数（Loss Function）
2. 理解梯度的含义
3. 从零实现梯度下降
4. 理解学习率的影响

运行方式：python3 02_gradient_descent.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 第一部分：什么是损失函数
# ============================================================

print("=" * 60)
print("第一部分：损失函数 —— 衡量模型有多'差'")
print("=" * 60)

# 假设我们要学习一个简单的关系：y = 2x + 1
# 模型一开始并不知道参数是 w=2, b=1，它需要自己学出来

np.random.seed(42)
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_true = 2 * X + 1  # 真实值：3, 5, 7, 9, 11

w, b = 0.0, 0.0  # 模型初始参数（瞎猜的）

y_pred = w * X + b
print(f"真实值:   {y_true}")
print(f"初始预测: {y_pred}  (w={w}, b={b}，全是0，完全不对)")

# 均方误差（MSE）：预测值与真实值差距的平方的平均值
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

loss = mse_loss(y_pred, y_true)
print(f"初始损失: {loss:.4f}  (越大说明模型越差)")
print()


# ============================================================
# 第二部分：什么是梯度
# ============================================================

print("=" * 60)
print("第二部分：梯度 —— 告诉你往哪个方向调整参数")
print("=" * 60)

print("""
梯度 = 损失函数对每个参数的偏导数

对于 MSE Loss = mean((wx + b - y)²)：
  ∂Loss/∂w = mean(2(wx + b - y) · x)  → w 应该怎么调
  ∂Loss/∂b = mean(2(wx + b - y))       → b 应该怎么调

梯度的含义：
  - 梯度为正 → 参数应该减小
  - 梯度为负 → 参数应该增大
  - 梯度为零 → 参数已经最优（到达谷底）
""")

def compute_gradients(X, y_true, w, b):
    """手动计算 w 和 b 的梯度"""
    n = len(X)
    y_pred = w * X + b
    error = y_pred - y_true

    dw = (2 / n) * np.sum(error * X)
    db = (2 / n) * np.sum(error)
    return dw, db

dw, db = compute_gradients(X, y_true, w, b)
print(f"当前 w={w}, b={b}")
print(f"w 的梯度: {dw:.4f}  (负数 → w 应该增大)")
print(f"b 的梯度: {db:.4f}  (负数 → b 应该增大)")
print()

# ---- 可视化：梯度到底在说什么？ ----

plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ========== 图1：损失函数的"山谷"曲线 ==========
w_range = np.linspace(-1, 5, 200)
losses = [mse_loss(wi * X + 0, y_true) for wi in w_range]  # 固定 b=0，只看 w 的影响

axes[0].plot(w_range, losses, 'b-', linewidth=2)
axes[0].set_xlabel('w (weight)')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss vs w\n(Goal: find the valley bottom)')

w_demo_points = [0.0, 1.0, 2.0, 3.0, 4.0]
for wp in w_demo_points:
    lp = mse_loss(wp * X + 0, y_true)
    gp, _ = compute_gradients(X, y_true, wp, 0)
    axes[0].plot(wp, lp, 'ro', markersize=8)
    axes[0].annotate(f'grad={gp:.1f}', (wp, lp),
                     textcoords="offset points", xytext=(5, 10), fontsize=9,
                     color='red')
    arrow_len = -gp * 0.08
    axes[0].annotate('', xy=(wp + arrow_len, lp - abs(arrow_len) * 5),
                     xytext=(wp, lp),
                     arrowprops=dict(arrowstyle='->', color='green', lw=2))

axes[0].axvline(x=2.0, color='gray', linestyle='--', alpha=0.5, label='optimal w=2')
axes[0].legend(fontsize=10)

# ========== 图2：梯度下降的轨迹 ==========
w_path, b_path, loss_path = [0.0], [0.0], [mse_loss(0, y_true)]
w_t, b_t = 0.0, 0.0
lr = 0.02
for _ in range(80):
    dw_t, db_t = compute_gradients(X, y_true, w_t, b_t)
    w_t = w_t - lr * dw_t
    b_t = b_t - lr * db_t
    w_path.append(w_t)
    b_path.append(b_t)
    loss_path.append(mse_loss(w_t * X + b_t, y_true))

axes[1].plot(loss_path, 'g-o', markersize=3, linewidth=1.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Gradient Descent Process\n(Loss decreasing over time)')
axes[1].set_yscale('log')

axes[1].annotate('Start: loss={:.1f}'.format(loss_path[0]),
                 xy=(0, loss_path[0]),
                 textcoords="offset points", xytext=(30, -5), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))
axes[1].annotate('End: loss={:.4f}'.format(loss_path[-1]),
                 xy=(len(loss_path)-1, loss_path[-1]),
                 textcoords="offset points", xytext=(-80, 20), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='blue'))

# ========== 图3：参数 w 在"山谷"上的移动轨迹 ==========
w_range2 = np.linspace(-0.5, 3.5, 200)
b_at_end = b_path[-1]
losses2 = [mse_loss(wi * X + b_at_end, y_true) for wi in w_range2]

axes[2].plot(w_range2, losses2, 'b-', linewidth=2, alpha=0.5, label='Loss curve')
w_sub = w_path[::5]
loss_sub = [mse_loss(ws * X + b_at_end, y_true) for ws in w_sub]
axes[2].plot(w_sub, loss_sub, 'r-o', markersize=6, linewidth=1.5, label='w trajectory')
axes[2].plot(w_sub[0], loss_sub[0], 'r^', markersize=14, label='Start')
axes[2].plot(w_sub[-1], loss_sub[-1], 'g*', markersize=14, label='End')
axes[2].set_xlabel('w (weight)')
axes[2].set_ylabel('Loss')
axes[2].set_title('w Moving Down the Valley\n(Gradient points uphill, we go downhill)')
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('phase1-foundations/gradient_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print(">> Gradient visualization saved to: phase1-foundations/gradient_visualization.png")
print()


# ============================================================
# 第三部分：梯度下降完整实现
# ============================================================

print("=" * 60)
print("第三部分：梯度下降 —— 一步一步逼近最优解")
print("=" * 60)

def train_linear_model(X, y_true, learning_rate=0.01, epochs=100, print_every=10):
    """
    用梯度下降训练一个线性模型 y = wx + b

    参数:
        learning_rate: 学习率，每一步走多大（太大会震荡，太小会太慢）
        epochs: 训练轮数
    """
    w, b = 0.0, 0.0

    for epoch in range(epochs):
        y_pred = w * X + b
        loss = mse_loss(y_pred, y_true)
        dw, db = compute_gradients(X, y_true, w, b)

        # 核心：参数更新 = 参数 - 学习率 × 梯度
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if epoch % print_every == 0:
            print(f"  Epoch {epoch:3d}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

    return w, b

print(f"目标：学习 y = 2x + 1 (即 w=2, b=1)")
print(f"训练数据: X={X}, y={y_true}\n")

w_final, b_final = train_linear_model(X, y_true, learning_rate=0.02, epochs=200, print_every=20)

print(f"\n学到的参数: w={w_final:.4f}, b={b_final:.4f}")
print(f"目标参数:   w=2.0000, b=1.0000")
print(f"→ 模型成功地通过梯度下降'学到'了正确的参数！")
print()


# ============================================================
# 第四部分：学习率的影响
# ============================================================

print("=" * 60)
print("第四部分：学习率的影响")
print("=" * 60)

print("\n--- 学习率太小 (0.001) ---")
print("  步子太小，收敛很慢：")
w1, b1 = train_linear_model(X, y_true, learning_rate=0.001, epochs=200, print_every=40)
print(f"  200轮后: w={w1:.4f}, b={b1:.4f} → 还没收敛")

print("\n--- 学习率合适 (0.02) ---")
print("  步子适中，收敛较快：")
w2, b2 = train_linear_model(X, y_true, learning_rate=0.02, epochs=200, print_every=40)
print(f"  200轮后: w={w2:.4f}, b={b2:.4f} → 接近目标")

print("\n--- 学习率太大 (0.1) ---")
print("  步子太大，可能震荡甚至发散：")
try:
    w3, b3 = train_linear_model(X, y_true, learning_rate=0.1, epochs=200, print_every=40)
    print(f"  200轮后: w={w3:.4f}, b={b3:.4f}")
except:
    print("  数值溢出了！学习率太大导致参数爆炸")
print()


# ============================================================
# 第五部分：与大模型的联系
# ============================================================

print("=" * 60)
print("与大模型的联系")
print("=" * 60)

print("""
我们刚才做的事情，和训练 GPT / LLaMA 本质上完全一样：

1. 定义模型:     我们用 y = wx + b
                  GPT 用 Transformer（参数多得多，但本质一样）

2. 定义损失函数:  我们用 MSE（均方误差）
                  GPT 用 Cross-Entropy Loss（交叉熵，预测下一个 token 的概率）

3. 计算梯度:      我们手动推导偏导数
                  PyTorch 用 Autograd 自动计算（反向传播）

4. 更新参数:      我们用 w = w - lr * grad
                  GPT 用 Adam 优化器（更智能的梯度下降变种）

唯一的区别是**规模**：
  - 我们的模型: 2 个参数 (w, b)
  - GPT-3:     1750 亿个参数
  - 但学习的原理是一模一样的！
""")

print("=" * 60)
print("恭喜完成第 2 课！")
print("下一课我们将从零搭建一个神经网络，实现'万能函数逼近器'")
print("=" * 60)
