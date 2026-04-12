"""
重置 phase1 第2课自写练习脚本。

用途：
  python3 reset_exercises_02.py

效果：
  - 将 02_gradient_descent_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path
import re


TARGET_FILE = Path(__file__).with_name("02_gradient_descent_self_write.py")


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"重置失败: {label} 匹配数量为 {count}（期望 1）")
    return new_text


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

    # TODO-1: mse_loss
    text = replace_between(
        text,
        "def mse_loss(",
        "\nw0, b0 = ",
        (
            "def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:\n"
            "    \"\"\"\n"
            "    TODO-1:\n"
            "    实现 MSE（均方误差）\n"
            "      MSE = mean((y_pred - y_true)^2)\n"
            "    \"\"\"\n"
            "    # return ...\n"
            "    raise NotImplementedError(\"TODO-1 未完成：请实现 mse_loss\")\n\n"
        ),
        "TODO-1",
    )

    # TODO-2: compute_gradients
    text = replace_between(
        text,
        "def compute_gradients(",
        "\ngrads_error = ",
        (
            "def compute_gradients(X: np.ndarray, y_true: np.ndarray, w: float, b: float):\n"
            "    \"\"\"\n"
            "    TODO-2:\n"
            "    已知：\n"
            "      y_pred = w * X + b\n"
            "      error  = y_pred - y_true\n\n"
            "    推导结果：\n"
            "      dw = (2/n) * sum(error * X)\n"
            "      db = (2/n) * sum(error)\n"
            "    \"\"\"\n"
            "    # y_pred = ...\n"
            "    # error = ...\n"
            "    # dw = ...\n"
            "    # db = ...\n"
            "    raise NotImplementedError(\"TODO-2 未完成：请实现梯度计算\")\n\n"
        ),
        "TODO-2",
    )

    # TODO-3: gradient_step
    text = replace_between(
        text,
        "def gradient_step(",
        "\nstep_error = ",
        (
            "def gradient_step(w: float, b: float, dw: float, db: float, learning_rate: float):\n"
            "    \"\"\"\n"
            "    TODO-3:\n"
            "    按梯度下降规则更新参数：\n"
            "      w_new = w - learning_rate * dw\n"
            "      b_new = b - learning_rate * db\n"
            "    \"\"\"\n"
            "    # w_new = ...\n"
            "    # b_new = ...\n"
            "    # return w_new, b_new\n"
            "    raise NotImplementedError(\"TODO-3 未完成：请实现单步更新\")\n\n"
        ),
        "TODO-3",
    )

    # TODO-4: train_linear_model
    text = replace_between(
        text,
        "def train_linear_model(",
        "\ntrain_error = ",
        (
            "def train_linear_model(\n"
            "    X: np.ndarray,\n"
            "    y_true: np.ndarray,\n"
            "    learning_rate: float = 0.02,\n"
            "    epochs: int = 200,\n"
            "    print_every: int = 40,\n"
            "):\n"
            "    \"\"\"\n"
            "    TODO-4:\n"
            "    手写训练循环，返回 (w, b, loss_history)\n\n"
            "    建议步骤：\n"
            "    1) 初始化 w,b = 0\n"
            "    2) 每轮计算 y_pred 和 loss\n"
            "    3) 计算 dw, db\n"
            "    4) 参数更新\n"
            "    5) 记录 loss_history\n"
            "    \"\"\"\n"
            "    # w, b = ...\n"
            "    # loss_history = ...\n"
            "    # for epoch in range(epochs):\n"
            "    #     ...\n"
            "    # return ...\n"
            "    raise NotImplementedError(\"TODO-4 未完成：请实现训练循环\")\n\n"
        ),
        "TODO-4",
    )

    # TODO-5: numerical_gradient_w（选做）
    text = replace_between(
        text,
        "def numerical_gradient_w(",
        "\ngrad_check_error = ",
        (
            "def numerical_gradient_w(X: np.ndarray, y_true: np.ndarray, w: float, b: float, eps: float = 1e-5):\n"
            "    \"\"\"\n"
            "    TODO-5（选做）:\n"
            "    用有限差分近似 dw（只检验 w 的梯度）：\n"
            "      dw_num ≈ (L(w + eps, b) - L(w - eps, b)) / (2*eps)\n\n"
            "    说明：\n"
            "    - 这是进阶检查项，用来验证你手写的解析梯度是否可靠\n"
            "    - 不影响主线学习（可先跳过，后续再补）\n"
            "    \"\"\"\n"
            "    # return ...\n"
            "    raise NotImplementedError(\"TODO-5 未完成：请实现数值梯度\")\n\n"
        ),
        "TODO-5",
    )

    # TODO-6: 学习率实验区
    text = replace_between(
        text,
        "# TODO-6:",
        '\nprint(f"lr_results={lr_results}")',
        (
            "# TODO-6:\n"
            "# 1) 用 train_linear_model 分别测试 0.001 / 0.02 / 0.1\n"
            "# 2) 把结果存到 lr_results\n"
            "# 示例：\n"
            "# w_lr, b_lr, hist_lr = train_linear_model(...)\n"
            "# lr_results[0.02] = (w_lr, b_lr, hist_lr[-1])\n"
        ),
        "TODO-6",
    )

    # 防御性重置：确保 lr_results 为初始空字典
    text = replace_once(text, r"^lr_results\s*=.*$", "lr_results = {}", "lr_results")

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
