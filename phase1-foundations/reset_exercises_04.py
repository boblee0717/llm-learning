"""
重置 phase1 第4课自写练习脚本。

用途：
  python3 reset_exercises_04.py

效果：
  - 将 04_optimizers_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path


TARGET_FILE = Path(__file__).with_name("04_optimizers_self_write.py")


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

    # TODO-1: sgd_step
    text = replace_between(
        text,
        "def sgd_step(params, grads, lr):",
        "\n\n\n# ============================================================\n# 第二部分",
        (
            "def sgd_step(params, grads, lr):\n"
            "    \"\"\"\n"
            "    TODO-1:\n"
            "    实现 SGD 的单步更新（就是第 2 课那条公式）：\n"
            "      params_new = params - lr * grads\n\n"
            "    参数都是 np.ndarray，返回更新后的 params。\n"
            "    \"\"\"\n"
            "    raise NotImplementedError(\"TODO-1 未完成：请实现 sgd_step\")"
        ),
        "TODO-1",
    )

    # TODO-2: momentum_step
    text = replace_between(
        text,
        "def momentum_step(params, grads, velocity, lr, beta=0.9):",
        "\n\n\n# ============================================================\n# 第三部分",
        (
            "def momentum_step(params, grads, velocity, lr, beta=0.9):\n"
            "    \"\"\"\n"
            "    TODO-2:\n"
            "    实现 Momentum 单步更新（PyTorch SGD 的 momentum 约定）：\n"
            "      velocity = beta * velocity + grads\n"
            "      params   = params - lr * velocity\n\n"
            "    返回元组 (params_new, velocity_new)。\n"
            "    注意：velocity 是\"状态\"，要返回新的 velocity 供下一步使用。\n"
            "    \"\"\"\n"
            "    raise NotImplementedError(\"TODO-2 未完成：请实现 momentum_step\")"
        ),
        "TODO-2",
    )

    # TODO-3: rmsprop_step
    text = replace_between(
        text,
        "def rmsprop_step(params, grads, sq_avg, lr, beta=0.9, eps=1e-8):",
        "\n\n\n# ============================================================\n# 第四部分",
        (
            "def rmsprop_step(params, grads, sq_avg, lr, beta=0.9, eps=1e-8):\n"
            "    \"\"\"\n"
            "    TODO-3:\n"
            "    实现 RMSprop 单步更新：\n"
            "      sq_avg = beta * sq_avg + (1 - beta) * grads**2\n"
            "      params = params - lr * grads / (sqrt(sq_avg) + eps)\n\n"
            "    返回元组 (params_new, sq_avg_new)。\n"
            "    提示：用 np.sqrt(sq_avg)。\n"
            "    \"\"\"\n"
            "    raise NotImplementedError(\"TODO-3 未完成：请实现 rmsprop_step\")"
        ),
        "TODO-3",
    )

    # TODO-4: adam_step
    text = replace_between(
        text,
        "def adam_step(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):",
        "\n\n\n# ============================================================\n# 第五部分",
        (
            "def adam_step(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):\n"
            "    \"\"\"\n"
            "    TODO-4:\n"
            "    实现 Adam 单步更新（动量 + 自适应 + 偏差修正）：\n"
            "      m = beta1 * m + (1 - beta1) * grads\n"
            "      v = beta2 * v + (1 - beta2) * grads**2\n"
            "      m_hat = m / (1 - beta1**t)      # t 从 1 开始\n"
            "      v_hat = v / (1 - beta2**t)\n"
            "      params = params - lr * m_hat / (sqrt(v_hat) + eps)\n\n"
            "    返回元组 (params_new, m_new, v_new)。\n"
            "    注意：m、v 都是状态，t 是当前步数（用于偏差修正），由调用方传入。\n"
            "    \"\"\"\n"
            "    raise NotImplementedError(\"TODO-4 未完成：请实现 adam_step\")"
        ),
        "TODO-4",
    )

    # TODO-5: adamw_step
    text = replace_between(
        text,
        "def adamw_step(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):",
        "\n\n\n# ============================================================\n# 第六部分",
        (
            "def adamw_step(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):\n"
            "    \"\"\"\n"
            "    TODO-5:\n"
            "    实现 AdamW 单步更新。和 Adam 唯一的区别：权重衰减\"解耦\"出来，\n"
            "    直接作用在参数上，不进梯度、也不被 sqrt(v_hat) 缩放：\n"
            "      m, v, m_hat, v_hat 的算法同 Adam\n"
            "      params = params - lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * params )\n\n"
            "    返回元组 (params_new, m_new, v_new)。\n"
            "    \"\"\"\n"
            "    raise NotImplementedError(\"TODO-5 未完成：请实现 adamw_step\")"
        ),
        "TODO-5",
    )

    # TODO-6: adam_training_bytes_per_param
    text = replace_between(
        text,
        "def adam_training_bytes_per_param():",
        "\n\n\n# ============================================================\n# 把单步函数跑成完整训练",
        (
            "def adam_training_bytes_per_param():\n"
            "    \"\"\"\n"
            "    TODO-6:\n"
            "    返回\"混合精度 + Adam\"训练时，每个参数占用的字节数（经典的 16 字节拆解）：\n"
            "      fp16 权重 (2) + fp16 梯度 (2) + fp32 主权重 (4) + fp32 一阶矩 m (4) + fp32 二阶矩 v (4)\n\n"
            "    返回一个整数（应当等于 16）。\n"
            "    想一想：这 16 字节里，优化器状态(m + v)占了几个字节？（答案：8，是权重本身的好几倍）\n"
            "    \"\"\"\n"
            "    raise NotImplementedError(\"TODO-6 未完成：请返回每参数字节数\")"
        ),
        "TODO-6",
    )

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
