"""
======================================================
第 3 课（自写版）：从零搭建神经网络
======================================================

用法：
1) 先运行原版 03_neural_network.py，理解整体流程
2) 运行本文件：python3 03_neural_network_self_write.py
3) 按 TODO-1 到 TODO-7 逐个补全
4) 每完成一个 TODO 就运行一次，查看校验报错

目标：
- 手写 ReLU / Sigmoid 激活函数及其导数
- 手写 Xavier 权重初始化
- 手写前向传播（forward）
- 手写二元交叉熵损失（BCE loss）
- 手写反向传播（backward）—— 核心难点
- 手写训练循环
"""

import numpy as np

np.random.seed(42)


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# ------------------------------------------------------------------
# 校验工具
# ------------------------------------------------------------------

class ValidationError(Exception):
    """统一的练习校验错误。"""


def require_not_none(name: str, value, hint: str) -> None:
    if value is None:
        raise ValidationError(f"{name} 错误：结果是 None。{hint}")


def require_close(name: str, actual, expected, hint: str = "", atol: float = 1e-6) -> None:
    try:
        if not np.allclose(actual, expected, atol=atol):
            raise ValidationError(
                f"{name} 错误：数值不正确。\nactual={actual}\nexpected={expected}\n{hint}"
            )
    except TypeError as err:
        raise ValidationError(f"{name} 错误：类型不正确。{hint}\n底层错误: {err}") from err


def require_shape(name: str, arr, expected_shape: tuple, hint: str = "") -> None:
    if not hasattr(arr, "shape"):
        raise ValidationError(f"{name} 错误：结果不是 ndarray。{hint}")
    if arr.shape != expected_shape:
        raise ValidationError(
            f"{name} 错误：shape={arr.shape}，期望 {expected_shape}。{hint}"
        )


def require_true(name: str, condition: bool, hint: str) -> None:
    if not condition:
        raise ValidationError(f"{name} 错误：{hint}")


def shape_or_none(x):
    return None if x is None else getattr(x, "shape", None)


# ------------------------------------------------------------------
# 参考实现（仅用于自动校验，不要偷看 :)）
# ------------------------------------------------------------------

def _ref_relu(x):
    return np.maximum(0, x)


def _ref_relu_deriv(x):
    return (x > 0).astype(float)


def _ref_sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def _ref_sigmoid_deriv(x):
    s = _ref_sigmoid(x)
    return s * (1 - s)


def _ref_bce(y_pred, y_true):
    eps = 1e-8
    return -np.mean(
        y_true * np.log(y_pred + eps) +
        (1 - y_true) * np.log(1 - y_pred + eps)
    )


# ============================================================
# 第一部分：激活函数 —— 引入非线性（TODO-1, TODO-2）
# ============================================================

section("第一部分：激活函数")

print("""
为什么需要激活函数？
  没有激活函数：多层线性变换 = 一层线性变换（矩阵乘法结合律）
  有了激活函数：每一层都引入非线性，网络能表达复杂模式
""")


def relu(x):
    """
    TODO-1a:
    实现 ReLU 激活函数：
      relu(x) = max(0, x)   （逐元素）

    提示：可以使用 np.maximum(0, x)
    """
    return np.maximum(0, x)
    raise NotImplementedError("TODO-1a 未完成：请实现 relu")


def relu_derivative(x):
    """
    TODO-1b:
    实现 ReLU 的导数：
      x > 0  → 1
      x <= 0 → 0

    提示：(x > 0) 得到布尔数组，用 .astype(float) 转成 0/1
    """
    return (x > 0).astype(float)
    raise NotImplementedError("TODO-1b 未完成：请实现 relu_derivative")


def sigmoid(x):
    """
    TODO-2a:
    实现 Sigmoid 激活函数：
      sigmoid(x) = 1 / (1 + exp(-x))

    数值稳定提示：用 np.clip(x, -500, 500) 再取 exp，防止溢出。
    原因：当 x 是很大的负数（如 -1000）时，-x = 1000，e^1000 超出 float64 范围变成 inf。
    clip 到 ±500 不影响结果（|x|>20 时 sigmoid 已经非常接近 0 或 1）。
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    raise NotImplementedError("TODO-2a 未完成：请实现 sigmoid")


def sigmoid_derivative(x):
    """
    TODO-2b:
    实现 Sigmoid 的导数：
      sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

    提示：先调用你写好的 sigmoid(x)，再计算 s * (1 - s)
    """
    s = sigmoid(x)
    return s * (1 - s)
    raise NotImplementedError("TODO-2b 未完成：请实现 sigmoid_derivative")


test_x = np.array([-2, -1, 0, 1, 2, 3], dtype=float)

relu_err = None
relu_result = None
try:
    relu_result = relu(test_x)
except Exception as err:  # noqa: BLE001
    relu_err = err

relu_d_err = None
relu_d_result = None
try:
    relu_d_result = relu_derivative(test_x)
except Exception as err:  # noqa: BLE001
    relu_d_err = err

sigmoid_err = None
sigmoid_result = None
try:
    sigmoid_result = sigmoid(test_x)
except Exception as err:  # noqa: BLE001
    sigmoid_err = err

sigmoid_d_err = None
sigmoid_d_result = None
try:
    sigmoid_d_result = sigmoid_derivative(test_x)
except Exception as err:  # noqa: BLE001
    sigmoid_d_err = err

print(f"输入 x:               {test_x}")
print(f"ReLU(x):              {relu_result if relu_err is None else f'执行失败 -> {relu_err}'}")
print(f"ReLU_derivative(x):   {relu_d_result if relu_d_err is None else f'执行失败 -> {relu_d_err}'}")
print(f"Sigmoid(x):           {np.round(sigmoid_result, 4) if sigmoid_err is None else f'执行失败 -> {sigmoid_err}'}")
print(f"Sigmoid_derivative(x):{np.round(sigmoid_d_result, 4) if sigmoid_d_err is None else f'执行失败 -> {sigmoid_d_err}'}")
print()


# ============================================================
# 第二部分：生成训练数据（XOR 问题）
# ============================================================

section("第二部分：生成训练数据（异或问题）")

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
y = np.array([[0], [1], [1], [0]])

print("XOR 问题 (异或):")
print("  输入      → 输出")
for i in range(len(X)):
    print(f"  {X[i]}  →  {y[i][0]}")
print()
print("这个问题无法用一条直线分开 → 线性模型解决不了，但神经网络可以！")
print()


# ============================================================
# 第三部分：从零实现 2 层神经网络（TODO-3 ~ TODO-6）
# ============================================================

section("第三部分：2层神经网络 —— 完整实现")

print("""
网络结构：
  输入层 (2个神经元) → 隐藏层 (8个神经元, ReLU) → 输出层 (1个神经元, Sigmoid)

  x ──→ [W1, b1] ──→ ReLU ──→ [W2, b2] ──→ Sigmoid ──→ 预测值
  (4,2)   (2,8)      (4,8)     (8,1)       (4,1)

前向传播（Forward）：从输入算到输出
  z1 = X @ W1 + b1        # 线性变换1       shape: (n, hidden)
  a1 = ReLU(z1)            # 激活            shape: (n, hidden)
  z2 = a1 @ W2 + b2        # 线性变换2       shape: (n, output)
  a2 = Sigmoid(z2)          # 输出概率        shape: (n, output)

反向传播（Backward）：从输出往回算梯度
  δ2   = a2 - y_true                         # 输出层误差        (n, 1)
  dW2  = (1/n) * a1.T @ δ2                   # W2 的梯度         (hidden, 1)
  db2  = (1/n) * sum(δ2, axis=0)             # b2 的梯度         (1, 1)
  da1  = δ2 @ W2.T                           # 误差传回隐藏层     (n, hidden)
  δ1   = da1 * relu_derivative(z1)           # 隐藏层误差        (n, hidden)
  dW1  = (1/n) * X.T @ δ1                    # W1 的梯度         (2, hidden)
  db1  = (1/n) * sum(δ1, axis=0)             # b1 的梯度         (1, hidden)
""")


class NeuralNetwork:
    """一个 2 层全连接神经网络，纯 NumPy 实现"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        TODO-3:
        初始化权重和偏置。

        使用 He 初始化（也叫 Kaiming 初始化），专为 ReLU 设计：
          W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
          b1 = np.zeros((1, hidden_dim))
          W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
          b2 = np.zeros((1, output_dim))

        为什么这样初始化？
          目标：让每一层输出的方差 ≈ 输入的方差，防止信号逐层爆炸或消失。
          推导：z = W @ x → Var(z) = fan_in * Var(W) * Var(x)
                要 Var(z) = Var(x) → Var(W) = 1/fan_in → W ~ randn * sqrt(1/fan_in)
                ReLU 砍掉一半负值（方差减半），补偿 ×2 → W ~ randn * sqrt(2/fan_in)

        常见变体对比：
          Xavier:  sqrt(1/fan_in) — 适合 Sigmoid/Tanh
          He:      sqrt(2/fan_in) — 适合 ReLU

        注意：严格来说，W1 后接 ReLU 应该用 He，W2 后接 Sigmoid 应该用 Xavier。
        这里为教学简洁统一用了 He，小网络下两者差异可忽略。

        把它们存成 self.W1, self.b1, self.W2, self.b2
        """
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        return
        raise NotImplementedError("TODO-3 未完成：请实现权重初始化")


    def forward(self, X):
        """
        TODO-4:
        前向传播：输入 → 预测

        步骤（结果都存到 self 上，反向传播时要用）：
          self.z1 = X @ self.W1 + self.b1
          self.a1 = relu(self.z1)
          self.z2 = self.a1 @ self.W2 + self.b2
          self.a2 = sigmoid(self.z2)

        返回 self.a2
        """
        raise NotImplementedError("TODO-4 未完成：请实现前向传播")


    def compute_loss(self, y_pred, y_true):
        """
        TODO-5:
        二元交叉熵损失 (Binary Cross-Entropy)：
          BCE = -mean( y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred) )

        数值稳定提示：在 y_pred 上加一个极小值 eps = 1e-8 防止 log(0)
          np.log(y_pred + eps) 和 np.log(1 - y_pred + eps)
        """
        raise NotImplementedError("TODO-5 未完成：请实现交叉熵损失")


    def backward(self, X, y_true, y_pred, learning_rate):
        """
        TODO-6:
        反向传播：计算梯度并更新参数

        这是本课最核心的部分！按以下步骤实现：

        1) 样本数：
           n = X.shape[0]

        2) 输出层误差（BCE + Sigmoid 的导数简化结果）：
           dz2 = y_pred - y_true                           # shape: (n, 1)

        3) W2, b2 的梯度：
           dW2 = (1/n) * self.a1.T @ dz2                   # shape: (hidden, 1)
           db2 = (1/n) * np.sum(dz2, axis=0, keepdims=True) # shape: (1, 1)

        4) 误差传回隐藏层：
           da1 = dz2 @ self.W2.T                           # shape: (n, hidden)
           dz1 = da1 * relu_derivative(self.z1)            # shape: (n, hidden)

        5) W1, b1 的梯度：
           dW1 = (1/n) * X.T @ dz1                         # shape: (2, hidden)
           db1 = (1/n) * np.sum(dz1, axis=0, keepdims=True) # shape: (1, hidden)

        6) 更新参数（梯度下降）：
           self.W1 -= learning_rate * dW1
           self.b1 -= learning_rate * db1
           self.W2 -= learning_rate * dW2
           self.b2 -= learning_rate * db2
        """
        raise NotImplementedError("TODO-6 未完成：请实现反向传播")


# 尝试创建网络
init_error = None
nn = None
try:
    np.random.seed(42)
    nn = NeuralNetwork(input_dim=2, hidden_dim=8, output_dim=1)
except Exception as err:  # noqa: BLE001
    init_error = err

if init_error is not None:
    print(f"网络初始化失败 -> {init_error}")
else:
    print(f"网络创建成功！")
    print(f"  W1.shape={nn.W1.shape}, b1.shape={nn.b1.shape}")
    print(f"  W2.shape={nn.W2.shape}, b2.shape={nn.b2.shape}")
    print(f"  总参数量: {nn.W1.size + nn.b1.size + nn.W2.size + nn.b2.size}")
print()


# 尝试前向传播
forward_error = None
y_pred_init = None
if nn is not None:
    try:
        y_pred_init = nn.forward(X)
    except Exception as err:  # noqa: BLE001
        forward_error = err

print(f"初始前向传播: {shape_or_none(y_pred_init) if forward_error is None else f'执行失败 -> {forward_error}'}")
if y_pred_init is not None:
    print(f"  初始预测值: {y_pred_init.flatten()}")
print()

# 尝试计算损失
loss_error = None
loss_init = None
if nn is not None and y_pred_init is not None:
    try:
        loss_init = nn.compute_loss(y_pred_init, y)
    except Exception as err:  # noqa: BLE001
        loss_error = err

print(f"初始损失: {loss_init if loss_error is None else f'执行失败 -> {loss_error}'}")
print()


# ============================================================
# 第四部分：训练循环（TODO-7）
# ============================================================

section("第四部分：训练神经网络（TODO-7）")

print("""
TODO-7:
编写训练循环，让网络学会解决 XOR 问题。

步骤：
1) 重新创建网络（重置参数）
2) 设置超参数：epochs=5000, learning_rate=1.0
3) 每轮：前向传播 → 计算损失 → 反向传播
4) 每 500 轮打印一次损失和准确率
""")

# 重新创建网络（保证每次从相同初始状态开始）
train_error = None
nn_trained = None
loss_hist = []
try:
    np.random.seed(42)
    nn_trained = NeuralNetwork(input_dim=2, hidden_dim=8, output_dim=1)
    epochs = 5000
    learning_rate = 1.0

    # TODO-7:
    # 在这里编写训练循环
    # for epoch in range(epochs):
    #     1) y_pred = nn_trained.forward(X)
    #     2) loss = nn_trained.compute_loss(y_pred, y)
    #     3) nn_trained.backward(X, y, y_pred, learning_rate)
    #     4) loss_hist.append(loss)
    #     5) 每 500 轮打印:
    #        accuracy = np.mean((y_pred > 0.5).astype(int) == y) * 100
    #        print(f"  Epoch {epoch:4d}: loss={loss:.4f}, accuracy={accuracy:.0f}%")
    raise NotImplementedError("TODO-7 未完成：请实现训练循环")

except Exception as err:  # noqa: BLE001
    train_error = err

if train_error is not None:
    print(f"训练失败 -> {train_error}")
print()


# ============================================================
# 第五部分：测试结果
# ============================================================

section("第五部分：测试结果")

if nn_trained is not None and train_error is None:
    y_pred_final = nn_trained.forward(X)
    print("输入      预测概率    预测值    真实值    正确？")
    print("-" * 55)
    for i in range(len(X)):
        pred_label = 1 if y_pred_final[i][0] > 0.5 else 0
        correct = "Y" if pred_label == y[i][0] else "N"
        print(f"  {X[i]}    {y_pred_final[i][0]:.4f}      {pred_label}        {y[i][0]}       {correct}")
    accuracy = np.mean((y_pred_final > 0.5).astype(int) == y) * 100
    print(f"\n最终准确率: {accuracy:.0f}%")
else:
    print("训练未完成，跳过测试。")
print()


# ============================================================
# 第六部分：学到的参数
# ============================================================

section("第六部分：学到的参数")

if nn_trained is not None and train_error is None:
    print(f"W1 (输入→隐藏, shape {nn_trained.W1.shape}):")
    print(np.round(nn_trained.W1, 3))
    print(f"\nb1 (隐藏层偏置, shape {nn_trained.b1.shape}):")
    print(np.round(nn_trained.b1, 3))
    print(f"\nW2 (隐藏→输出, shape {nn_trained.W2.shape}):")
    print(np.round(nn_trained.W2, 3))
    print(f"\nb2 (输出层偏置, shape {nn_trained.b2.shape}):")
    print(np.round(nn_trained.b2, 3))
else:
    print("训练未完成，跳过参数展示。")
print()


# ============================================================
# 总结
# ============================================================

section("总结：与大模型的对应关系")

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
""")


# ============================================================
# 自动校验
# ============================================================

def validate_all() -> None:
    """统一校验：检查 TODO-1 到 TODO-7。"""

    # --- TODO-1: ReLU ---
    if relu_err is not None:
        raise ValidationError(f"TODO-1a 错误：relu 执行失败。底层错误: {relu_err}")
    require_not_none("TODO-1a", relu_result, "relu 返回了 None。")
    require_close("TODO-1a", relu_result, _ref_relu(test_x), "ReLU 输出不正确。")

    if relu_d_err is not None:
        raise ValidationError(f"TODO-1b 错误：relu_derivative 执行失败。底层错误: {relu_d_err}")
    require_not_none("TODO-1b", relu_d_result, "relu_derivative 返回了 None。")
    require_close("TODO-1b", relu_d_result, _ref_relu_deriv(test_x), "ReLU 导数不正确。")

    # --- TODO-2: Sigmoid ---
    if sigmoid_err is not None:
        raise ValidationError(f"TODO-2a 错误：sigmoid 执行失败。底层错误: {sigmoid_err}")
    require_not_none("TODO-2a", sigmoid_result, "sigmoid 返回了 None。")
    require_close("TODO-2a", sigmoid_result, _ref_sigmoid(test_x), "Sigmoid 输出不正确。")

    if sigmoid_d_err is not None:
        raise ValidationError(f"TODO-2b 错误：sigmoid_derivative 执行失败。底层错误: {sigmoid_d_err}")
    require_not_none("TODO-2b", sigmoid_d_result, "sigmoid_derivative 返回了 None。")
    require_close("TODO-2b", sigmoid_d_result, _ref_sigmoid_deriv(test_x), "Sigmoid 导数不正确。")

    # --- TODO-3: __init__ ---
    if init_error is not None:
        raise ValidationError(f"TODO-3 错误：网络初始化失败。底层错误: {init_error}")
    require_true("TODO-3", nn is not None, "网络对象为 None。")
    require_shape("TODO-3", nn.W1, (2, 8), "W1 的 shape 应为 (input_dim, hidden_dim)=(2,8)。")
    require_shape("TODO-3", nn.b1, (1, 8), "b1 的 shape 应为 (1, hidden_dim)=(1,8)。")
    require_shape("TODO-3", nn.W2, (8, 1), "W2 的 shape 应为 (hidden_dim, output_dim)=(8,1)。")
    require_shape("TODO-3", nn.b2, (1, 1), "b2 的 shape 应为 (1, output_dim)=(1,1)。")
    require_close("TODO-3", nn.b1, np.zeros((1, 8)), "b1 应全部初始化为 0。")
    require_close("TODO-3", nn.b2, np.zeros((1, 1)), "b2 应全部初始化为 0。")

    # --- TODO-4: forward ---
    if forward_error is not None:
        raise ValidationError(f"TODO-4 错误：前向传播失败。底层错误: {forward_error}")
    require_not_none("TODO-4", y_pred_init, "forward 返回了 None。")
    require_shape("TODO-4", y_pred_init, (4, 1), "前向传播输出 shape 应为 (4,1)。")
    require_true(
        "TODO-4",
        np.all(y_pred_init >= 0) and np.all(y_pred_init <= 1),
        "Sigmoid 输出应在 [0, 1] 范围内。",
    )

    # --- TODO-5: compute_loss ---
    if loss_error is not None:
        raise ValidationError(f"TODO-5 错误：损失计算失败。底层错误: {loss_error}")
    require_not_none("TODO-5", loss_init, "compute_loss 返回了 None。")
    require_true("TODO-5", np.isscalar(loss_init) or (hasattr(loss_init, 'ndim') and loss_init.ndim == 0),
                 "compute_loss 应返回标量。")
    ref_loss_init = _ref_bce(y_pred_init, y)
    require_close("TODO-5", loss_init, ref_loss_init, "初始 BCE 损失不正确。", atol=1e-5)

    # --- TODO-6: backward ---
    # 单独测试反向传播：从已知状态做一步更新，检查参数是否正确变化
    np.random.seed(42)
    nn_check = NeuralNetwork(input_dim=2, hidden_dim=8, output_dim=1)
    w1_before = nn_check.W1.copy()
    y_pred_check = nn_check.forward(X)
    try:
        nn_check.backward(X, y, y_pred_check, learning_rate=1.0)
    except Exception as err:
        raise ValidationError(f"TODO-6 错误：反向传播失败。底层错误: {err}") from err
    require_true(
        "TODO-6",
        not np.allclose(nn_check.W1, w1_before),
        "反向传播后 W1 应该发生变化（参数被更新了）。",
    )
    require_shape("TODO-6", nn_check.W1, (2, 8), "反向传播后 W1 的 shape 不应改变。")
    require_shape("TODO-6", nn_check.W2, (8, 1), "反向传播后 W2 的 shape 不应改变。")

    # 用参考实现做完整的一步对比
    np.random.seed(42)
    nn_ref = _build_ref_nn(2, 8, 1)
    y_pred_ref = _ref_forward(nn_ref, X)
    _ref_backward(nn_ref, X, y, y_pred_ref, 1.0)
    require_close("TODO-6", nn_check.W1, nn_ref["W1"], "一步反向传播后 W1 与参考值不一致。", atol=1e-6)
    require_close("TODO-6", nn_check.W2, nn_ref["W2"], "一步反向传播后 W2 与参考值不一致。", atol=1e-6)
    require_close("TODO-6", nn_check.b1, nn_ref["b1"], "一步反向传播后 b1 与参考值不一致。", atol=1e-6)
    require_close("TODO-6", nn_check.b2, nn_ref["b2"], "一步反向传播后 b2 与参考值不一致。", atol=1e-6)

    # --- TODO-7: 训练循环 ---
    if train_error is not None:
        raise ValidationError(f"TODO-7 错误：训练循环失败。底层错误: {train_error}")
    require_true("TODO-7", nn_trained is not None, "训练后的网络对象为 None。")
    require_true("TODO-7", len(loss_hist) == 5000, f"loss_hist 长度应为 5000，实际为 {len(loss_hist)}。")
    require_true(
        "TODO-7",
        loss_hist[0] > loss_hist[-1],
        "训练后损失应该下降（first_loss > last_loss）。",
    )
    y_final = nn_trained.forward(X)
    final_acc = np.mean((y_final > 0.5).astype(int) == y) * 100
    require_true(
        "TODO-7",
        final_acc == 100,
        f"XOR 四个样本应全部分类正确（当前准确率 {final_acc:.0f}%）。"
        "如果未达到 100%，尝试增加 epochs 或调整 learning_rate。",
    )


def _build_ref_nn(input_dim, hidden_dim, output_dim):
    nn_d = {}
    nn_d["W1"] = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    nn_d["b1"] = np.zeros((1, hidden_dim))
    nn_d["W2"] = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    nn_d["b2"] = np.zeros((1, output_dim))
    return nn_d


def _ref_forward(nn_d, X):
    nn_d["z1"] = X @ nn_d["W1"] + nn_d["b1"]
    nn_d["a1"] = _ref_relu(nn_d["z1"])
    nn_d["z2"] = nn_d["a1"] @ nn_d["W2"] + nn_d["b2"]
    nn_d["a2"] = _ref_sigmoid(nn_d["z2"])
    return nn_d["a2"]


def _ref_backward(nn_d, X, y_true, y_pred, lr):
    n = X.shape[0]
    dz2 = y_pred - y_true
    dW2 = (1 / n) * nn_d["a1"].T @ dz2
    db2 = (1 / n) * np.sum(dz2, axis=0, keepdims=True)
    da1 = dz2 @ nn_d["W2"].T
    dz1 = da1 * _ref_relu_deriv(nn_d["z1"])
    dW1 = (1 / n) * X.T @ dz1
    db1 = (1 / n) * np.sum(dz1, axis=0, keepdims=True)
    nn_d["W1"] -= lr * dW1
    nn_d["b1"] -= lr * db1
    nn_d["W2"] -= lr * dW2
    nn_d["b2"] -= lr * db2


section("自写版骨架就绪")
print("你可以按 TODO-1 到 TODO-7 逐个补全。")
print("开始自动校验...")
validate_all()
print("校验通过：你当前实现正确。恭喜完成第一阶段所有课程！")
