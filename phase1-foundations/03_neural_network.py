"""
======================================================
第 3 课：从零搭建神经网络
======================================================

上一课我们学了线性模型 y = wx + b，但它只能学线性关系。
现实中的问题（比如理解语言）远比线性关系复杂。

神经网络 = 多层线性变换 + 非线性激活函数
  → 理论上可以逼近任意复杂的函数！

学习目标：
1. 理解神经网络的前向传播
2. 理解反向传播（链式法则）
3. 从零实现一个2层神经网络
4. 用它解决一个非线性分类问题

运行方式：python3 03_neural_network.py
"""

import numpy as np

np.random.seed(42)

# ============================================================
# 第一部分：激活函数 —— 引入非线性
# ============================================================

print("=" * 60)
print("第一部分：激活函数")
print("=" * 60)

def relu(x):
    """ReLU: 小于0的变成0，大于0的保持不变。最常用的激活函数。"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU 的导数：x>0 时为1，x<=0 时为0"""
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid: 把任意实数压缩到 (0, 1) 之间，可以理解为'概率'"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Sigmoid 的导数：s(x) * (1 - s(x))"""
    s = sigmoid(x)
    return s * (1 - s)

x = np.array([-2, -1, 0, 1, 2, 3], dtype=float)
print(f"输入 x:       {x}")
print(f"ReLU(x):      {relu(x)}")
print(f"Sigmoid(x):   {np.round(sigmoid(x), 4)}")
print()
print("为什么需要激活函数？")
print("  没有激活函数：多层线性变换 = 一层线性变换（矩阵乘法的结合律）")
print("  有了激活函数：每一层都引入非线性，网络能表达复杂模式")
print()


# ============================================================
# 第二部分：生成非线性数据
# ============================================================

print("=" * 60)
print("第二部分：生成训练数据（异或问题）")
print("=" * 60)

# XOR 问题：线性模型无法解决的经典问题
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
y = np.array([[0], [1], [1], [0]])  # XOR：相同为0，不同为1

print("XOR 问题 (异或):")
print("  输入      → 输出")
for i in range(len(X)):
    print(f"  {X[i]}  →  {y[i][0]}")
print()
print("这个问题无法用一条直线分开（试试在纸上画画看）")
print("所以线性模型解决不了，但神经网络可以！")
print()


# ============================================================
# 第三部分：从零实现2层神经网络
# ============================================================

print("=" * 60)
print("第三部分：2层神经网络 —— 完整实现")
print("=" * 60)

print("""
网络结构：
  输入层 (2个神经元) → 隐藏层 (8个神经元, ReLU) → 输出层 (1个神经元, Sigmoid)

  x ──→ [W1, b1] ──→ ReLU ──→ [W2, b2] ──→ Sigmoid ──→ 预测值
  (2,)    (2,8)       (8,)      (8,1)        (1,)

前向传播（Forward）：从输入算到输出
  z1 = x @ W1 + b1
  a1 = ReLU(z1)
  z2 = a1 @ W2 + b2
  a2 = Sigmoid(z2)  ← 这就是预测结果

反向传播（Backward）：从输出往回算梯度，用链式法则
  Loss 对 W2 的梯度 = a1.T @ δ2
  Loss 对 W1 的梯度 = x.T @ δ1
  其中 δ 是误差信号，从输出层往输入层传递
""")


class NeuralNetwork:
    """一个2层全连接神经网络，纯 NumPy 实现"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        # Xavier 初始化：让每层输出的方差保持一致，避免梯度消失/爆炸
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        """前向传播：输入 → 预测"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y_true, y_pred, learning_rate):
        """反向传播：计算梯度并更新参数"""
        n = X.shape[0]

        # 输出层的误差
        dz2 = y_pred - y_true  # (n, 1) — Binary Cross-Entropy + Sigmoid 的导数简化

        # W2, b2 的梯度
        dW2 = (1 / n) * self.a1.T @ dz2
        db2 = (1 / n) * np.sum(dz2, axis=0, keepdims=True)

        # 误差传回隐藏层
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)

        # W1, b1 的梯度
        dW1 = (1 / n) * X.T @ dz1
        db1 = (1 / n) * np.sum(dz1, axis=0, keepdims=True)

        # 更新参数（梯度下降）
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def compute_loss(self, y_pred, y_true):
        """二元交叉熵损失"""
        eps = 1e-8
        return -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )


# ============================================================
# 第四部分：训练！
# ============================================================

print("=" * 60)
print("第四部分：训练神经网络")
print("=" * 60)

nn = NeuralNetwork(input_dim=2, hidden_dim=8, output_dim=1)

epochs = 5000
learning_rate = 1.0

print(f"网络结构: 2 → 8 → 1")
print(f"学习率: {learning_rate}")
print(f"训练轮数: {epochs}")
print(f"总参数量: {2*8 + 8 + 8*1 + 1} = {2*8 + 8 + 8*1 + 1}个")
print(f"  (对比: GPT-3 有 175,000,000,000 个参数)\n")

for epoch in range(epochs):
    y_pred = nn.forward(X)
    loss = nn.compute_loss(y_pred, y)
    nn.backward(X, y, y_pred, learning_rate)

    if epoch % 500 == 0:
        accuracy = np.mean((y_pred > 0.5).astype(int) == y) * 100
        print(f"  Epoch {epoch:4d}: loss={loss:.4f}, accuracy={accuracy:.0f}%")

print()


# ============================================================
# 第五部分：测试结果
# ============================================================

print("=" * 60)
print("第五部分：测试结果")
print("=" * 60)

y_pred = nn.forward(X)
print("输入      预测概率    预测值    真实值    正确？")
print("-" * 55)
for i in range(len(X)):
    pred_label = 1 if y_pred[i][0] > 0.5 else 0
    correct = "✓" if pred_label == y[i][0] else "✗"
    print(f"  {X[i]}    {y_pred[i][0]:.4f}      {pred_label}        {y[i][0]}       {correct}")

print()
print("神经网络成功解决了 XOR 问题！")
print("一个简单的线性模型做不到这一点。")
print()


# ============================================================
# 第六部分：看看网络学到了什么
# ============================================================

print("=" * 60)
print("第六部分：学到的参数")
print("=" * 60)

print(f"\nW1 (输入→隐藏, shape {nn.W1.shape}):")
print(np.round(nn.W1, 3))
print(f"\nb1 (隐藏层偏置, shape {nn.b1.shape}):")
print(np.round(nn.b1, 3))
print(f"\nW2 (隐藏→输出, shape {nn.W2.shape}):")
print(np.round(nn.W2, 3))
print(f"\nb2 (输出层偏置, shape {nn.b2.shape}):")
print(np.round(nn.b2, 3))
print()


# ============================================================
# 总结：与大模型的对应关系
# ============================================================

print("=" * 60)
print("总结：与大模型的对应关系")
print("=" * 60)

print("""
我们从零构建的这个网络，包含了深度学习的所有核心概念：

┌─────────────────────────┬─────────────────────────────┐
│ 我们实现的               │ 大模型中对应的               │
├─────────────────────────┼─────────────────────────────┤
│ W @ x + b (线性层)       │ nn.Linear (PyTorch)         │
│ ReLU 激活函数            │ GELU (GPT), SiLU (LLaMA)   │
│ 前向传播 forward()       │ model.forward()             │
│ 反向传播 backward()      │ loss.backward() (自动微分)   │
│ w -= lr * grad           │ optimizer.step() (Adam)      │
│ Binary Cross-Entropy     │ Cross-Entropy (预测下一token) │
│ 我们的网络: 2层           │ GPT-3: 96层 Transformer      │
│ 总参数: 33个              │ GPT-3: 1750亿个              │
└─────────────────────────┴─────────────────────────────┘

下一步你可以：
1. 试着修改 hidden_dim、learning_rate，观察效果
2. 把网络改成3层，看看深度的影响
3. 开始学习第二阶段 —— Transformer 架构！
""")

print("=" * 60)
print("恭喜完成第一阶段的所有课程！")
print("你已经理解了大模型训练的核心原理。")
print("=" * 60)
