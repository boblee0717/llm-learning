"""
重置 phase1 第1课自写练习脚本。

用途：
  python3 reset_exercises.py

效果：
  - 将 01_numpy_basics_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path
import re


TARGET_FILE = Path(__file__).with_name("01_numpy_basics_self_write.py")


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"重置失败: {label} 匹配数量为 {count}（期望 1）")
    return new_text


def reset_softmax_block(text: str) -> str:
    """
    重置 softmax 函数主体。

    兼容以下情况：
    - 用户保持了 TODO 注释
    - 用户改成了可运行实现（含 axis 参数）
    - 用户只改了部分实现
    """
    replacement = (
        "def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:\n"
        "    \"\"\"\n"
        "    TODO-4:\n"
        "    1) 做数值稳定处理：x - np.max(x)\n"
        "    2) 计算指数\n"
        "    3) 归一化（除以指数和）\n"
        "    \"\"\"\n"
        "    # x_stable = ...\n"
        "    # exp_x = ...\n"
        "    # return ...\n"
        "    raise NotImplementedError(\"TODO-4 未完成：请实现 softmax\")\n"
    )
    start = text.find("def softmax(")
    if start == -1:
        raise RuntimeError("重置失败: 找不到 softmax 函数定义（def softmax(...)）")

    # softmax 代码块之后，脚本里固定会出现 logits = ...
    end = text.find("\nlogits = ", start)
    if end == -1:
        raise RuntimeError("重置失败: 找不到 softmax 后续边界（logits = ...）")

    return text[:start] + replacement + text[end + 1 :]


def main() -> int:
    if not TARGET_FILE.exists():
        print(f"未找到目标文件: {TARGET_FILE}")
        return 1

    text = TARGET_FILE.read_text(encoding="utf-8")

    # 逐题重置为待实现状态
    replacements = [
        (r"^dot\s*=.*$", "dot = None", "TODO-1"),
        (r"^output\s*=.*$", "output = None", "TODO-2"),
        (r"^result\s*=.*$", "result = None", "TODO-3"),
        (r"^cos_sim_12\s*=.*$", "cos_sim_12 = None", "TODO-5-1"),
        (r"^cos_sim_13\s*=.*$", "cos_sim_13 = None", "TODO-5-2"),
        (r"^out\s*=.*$", "out = None", "TODO-6"),
        (r"^scores\s*=.*$", "scores = None", "TODO-7-1"),
        (r"^attn_weights\s*=.*$", "attn_weights = None", "TODO-7-2"),
        (r"^score_matrix\s*=.*$", "score_matrix = None", "TODO-8-1"),
        (r"^attn_matrix\s*=.*$", "attn_matrix = None", "TODO-8-2"),
    ]

    for pattern, replacement, label in replacements:
        text = replace_once(text, pattern, replacement, label)

    text = reset_softmax_block(text)

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
