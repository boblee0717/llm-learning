"""
重置 phase1 第3课自写练习脚本。

用途：
  python3 reset_exercises_03.py

效果：
  - 将 03_neural_network_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path


TARGET_FILE = Path(__file__).with_name("03_neural_network_self_write.py")


def replace_between(text: str, start_marker: str, end_marker: str, replacement: str, label: str) -> str:
    start = text.find(start_marker)
    if start == -1:
        raise RuntimeError(f"重置失败: {label} 找不到起始标记 {start_marker!r}")

    end = text.find(end_marker, start)
    if end == -1:
        raise RuntimeError(f"重置失败: {label} 找不到结束标记 {end_marker!r}")

    return text[:start] + replacement + text[end:]


def main() -> int:
    if not TARGET_FILE.exists():
        print(f"未找到目标文件: {TARGET_FILE}")
        return 1

    text = TARGET_FILE.read_text(encoding="utf-8")

    # TODO-1a: relu
    text = replace_between(
        text,
        "def relu(x):",
        "\ndef relu_derivative(",
        (
            "def relu(x):\n"
            "    \"\"\"\n"
            "    TODO-1a:\n"
            "    实现 ReLU 激活函数：\n"
            "      relu(x) = max(0, x)   （逐元素）\n\n"
            "    提示：可以使用 np.maximum(0, x)\n"
            "    \"\"\"\n"
            "    raise NotImplementedError(\"TODO-1a 未完成：请实现 relu\")\n\n"
        ),
        "TODO-1a",
    )

    # TODO-1b: relu_derivative
    text = replace_between(
        text,
        "def relu_derivative(x):",
        "\ndef sigmoid(",
        (
            "def relu_derivative(x):\n"
            "    \"\"\"\n"
            "    TODO-1b:\n"
            "    实现 ReLU 的导数：\n"
            "      x > 0  → 1\n"
            "      x <= 0 → 0\n\n"
            "    提示：(x > 0) 得到布尔数组，用 .astype(float) 转成 0/1\n"
            "    \"\"\"\n"
            "    raise NotImplementedError(\"TODO-1b 未完成：请实现 relu_derivative\")\n\n"
        ),
        "TODO-1b",
    )

    # TODO-2a: sigmoid
    text = replace_between(
        text,
        "def sigmoid(x):",
        "\ndef sigmoid_derivative(",
        (
            "def sigmoid(x):\n"
            "    \"\"\"\n"
            "    TODO-2a:\n"
            "    实现 Sigmoid 激活函数：\n"
            "      sigmoid(x) = 1 / (1 + exp(-x))\n\n"
            "    数值稳定提示：用 np.clip(x, -500, 500) 再取 exp，防止溢出。\n"
            "    原因：当 x 是很大的负数（如 -1000）时，-x = 1000，e^1000 超出 float64 范围变成 inf。\n"
            "    clip 到 ±500 不影响结果（|x|>20 时 sigmoid 已经非常接近 0 或 1）。\n"
            "    \"\"\"\n"
            "    raise NotImplementedError(\"TODO-2a 未完成：请实现 sigmoid\")\n\n"
        ),
        "TODO-2a",
    )

    # TODO-2b: sigmoid_derivative
    text = replace_between(
        text,
        "def sigmoid_derivative(x):",
        "\ntest_x = ",
        (
            "def sigmoid_derivative(x):\n"
            "    \"\"\"\n"
            "    TODO-2b:\n"
            "    实现 Sigmoid 的导数：\n"
            "      sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))\n\n"
            "    提示：先调用你写好的 sigmoid(x)，再计算 s * (1 - s)\n"
            "    \"\"\"\n"
            "    raise NotImplementedError(\"TODO-2b 未完成：请实现 sigmoid_derivative\")\n\n"
        ),
        "TODO-2b",
    )

    # TODO-3: __init__
    text = replace_between(
        text,
        "    def __init__(self, input_dim, hidden_dim, output_dim):",
        "\n    def forward(self, X):",
        (
            "    def __init__(self, input_dim, hidden_dim, output_dim):\n"
            "        \"\"\"\n"
            "        TODO-3:\n"
            "        初始化权重和偏置。\n\n"
            "        使用 He 初始化（也叫 Kaiming 初始化），专为 ReLU 设计：\n"
            "          W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)\n"
            "          b1 = np.zeros((1, hidden_dim))\n"
            "          W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)\n"
            "          b2 = np.zeros((1, output_dim))\n\n"
            "        为什么这样初始化？\n"
            "          目标：让每一层输出的方差 ≈ 输入的方差，防止信号逐层爆炸或消失。\n"
            "          推导：z = W @ x → Var(z) = fan_in * Var(W) * Var(x)\n"
            "                要 Var(z) = Var(x) → Var(W) = 1/fan_in → W ~ randn * sqrt(1/fan_in)\n"
            "                ReLU 砍掉一半负值（方差减半），补偿 ×2 → W ~ randn * sqrt(2/fan_in)\n\n"
            "        常见变体对比：\n"
            "          Xavier:  sqrt(1/fan_in) — 适合 Sigmoid/Tanh\n"
            "          He:      sqrt(2/fan_in) — 适合 ReLU\n\n"
            "        注意：严格来说，W1 后接 ReLU 应该用 He，W2 后接 Sigmoid 应该用 Xavier。\n"
            "        这里为教学简洁统一用了 He，小网络下两者差异可忽略。\n\n"
            "        把它们存成 self.W1, self.b1, self.W2, self.b2\n"
            "        \"\"\"\n"
            "        raise NotImplementedError(\"TODO-3 未完成：请实现权重初始化\")\n\n"
        ),
        "TODO-3",
    )

    # TODO-4: forward
    text = replace_between(
        text,
        "    def forward(self, X):",
        "\n    def compute_loss(self, y_pred, y_true):",
        (
            "    def forward(self, X):\n"
            "        \"\"\"\n"
            "        TODO-4:\n"
            "        前向传播：输入 → 预测\n\n"
            "        步骤（结果都存到 self 上，反向传播时要用）：\n"
            "          self.z1 = X @ self.W1 + self.b1\n"
            "          self.a1 = relu(self.z1)\n"
            "          self.z2 = self.a1 @ self.W2 + self.b2\n"
            "          self.a2 = sigmoid(self.z2)\n\n"
            "        返回 self.a2\n"
            "        \"\"\"\n"
            "        raise NotImplementedError(\"TODO-4 未完成：请实现前向传播\")\n\n"
        ),
        "TODO-4",
    )

    # TODO-5: compute_loss
    text = replace_between(
        text,
        "    def compute_loss(self, y_pred, y_true):",
        "\n    def backward(self, X, y_true, y_pred, learning_rate):",
        (
            "    def compute_loss(self, y_pred, y_true):\n"
            "        \"\"\"\n"
            "        TODO-5:\n"
            "        二元交叉熵损失 (Binary Cross-Entropy)：\n"
            "          BCE = -mean( y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred) )\n\n"
            "        数值稳定提示：在 y_pred 上加一个极小值 eps = 1e-8 防止 log(0)\n"
            "          np.log(y_pred + eps) 和 np.log(1 - y_pred + eps)\n"
            "        \"\"\"\n"
            "        raise NotImplementedError(\"TODO-5 未完成：请实现交叉熵损失\")\n\n"
        ),
        "TODO-5",
    )

    # TODO-6: backward
    text = replace_between(
        text,
        "    def backward(self, X, y_true, y_pred, learning_rate):",
        "\n# 尝试创建网络",
        (
            "    def backward(self, X, y_true, y_pred, learning_rate):\n"
            "        \"\"\"\n"
            "        TODO-6:\n"
            "        反向传播：计算梯度并更新参数\n\n"
            "        这是本课最核心的部分！按以下步骤实现：\n\n"
            "        1) 样本数：\n"
            "           n = X.shape[0]\n\n"
            "        2) 输出层误差（BCE + Sigmoid 的导数简化结果）：\n"
            "           dz2 = y_pred - y_true                           # shape: (n, 1)\n\n"
            "        3) W2, b2 的梯度：\n"
            "           dW2 = (1/n) * self.a1.T @ dz2                   # shape: (hidden, 1)\n"
            "           db2 = (1/n) * np.sum(dz2, axis=0, keepdims=True) # shape: (1, 1)\n\n"
            "        4) 误差传回隐藏层：\n"
            "           da1 = dz2 @ self.W2.T                           # shape: (n, hidden)\n"
            "           dz1 = da1 * relu_derivative(self.z1)            # shape: (n, hidden)\n\n"
            "        5) W1, b1 的梯度：\n"
            "           dW1 = (1/n) * X.T @ dz1                         # shape: (2, hidden)\n"
            "           db1 = (1/n) * np.sum(dz1, axis=0, keepdims=True) # shape: (1, hidden)\n\n"
            "        6) 更新参数（梯度下降）：\n"
            "           self.W1 -= learning_rate * dW1\n"
            "           self.b1 -= learning_rate * db1\n"
            "           self.W2 -= learning_rate * dW2\n"
            "           self.b2 -= learning_rate * db2\n"
            "        \"\"\"\n"
            "        raise NotImplementedError(\"TODO-6 未完成：请实现反向传播\")\n\n"
        ),
        "TODO-6",
    )

    # TODO-7: 训练循环
    text = replace_between(
        text,
        "    # TODO-7:",
        "\nexcept Exception as err:  # noqa: BLE001\n    train_error = err",
        (
            "    # TODO-7:\n"
            "    # 在这里编写训练循环\n"
            "    # for epoch in range(epochs):\n"
            "    #     1) y_pred = nn_trained.forward(X)\n"
            "    #     2) loss = nn_trained.compute_loss(y_pred, y)\n"
            "    #     3) nn_trained.backward(X, y, y_pred, learning_rate)\n"
            "    #     4) loss_hist.append(loss)\n"
            "    #     5) 每 500 轮打印:\n"
            "    #        accuracy = np.mean((y_pred > 0.5).astype(int) == y) * 100\n"
            '    #        print(f"  Epoch {epoch:4d}: loss={loss:.4f}, accuracy={accuracy:.0f}%")\n'
            '    raise NotImplementedError("TODO-7 未完成：请实现训练循环")\n'
        ),
        "TODO-7",
    )

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
