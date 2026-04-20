"""
重置 phase0 第 4 课自写练习脚本（04_matrix_calculus_self_write.py）。

用法：
    python3 reset_exercises_04.py
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET_FILE = Path(__file__).with_name("04_matrix_calculus_self_write.py")


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"重置失败: {label} 匹配数量为 {count}（期望 1）")
    return new_text


def reset_function(text: str, def_line: str, full_replacement: str, end_marker: str, label: str) -> str:
    start = text.find(def_line)
    if start == -1:
        raise RuntimeError(f"重置失败: 找不到 {label} 的 def 行：{def_line}")
    end = text.find(end_marker, start)
    if end == -1:
        raise RuntimeError(f"重置失败: 找不到 {label} 的结束标记：{end_marker}")
    return text[:start] + full_replacement + text[end:]


def main() -> int:
    if not TARGET_FILE.exists():
        print(f"未找到目标文件: {TARGET_FILE}")
        return 1

    text = TARGET_FILE.read_text(encoding="utf-8")

    line_resets = [
        (r"^dX\s*=.*# TODO-1-X.*$",   "dX = None  # TODO-1-X",   "TODO-1-X"),
        (r"^dW\s*=.*# TODO-1-W.*$",   "dW = None  # TODO-1-W",   "TODO-1-W"),
        (r"^db\s*=.*# TODO-1-b.*$",   "db = None  # TODO-1-b",   "TODO-1-b"),
        (r"^dW2\s*=.*# TODO-2-W2.*$", "dW2 = None  # TODO-2-W2", "TODO-2-W2"),
        (r"^db2\s*=.*# TODO-2-b2.*$", "db2 = None  # TODO-2-b2", "TODO-2-b2"),
        (r"^dH\s*=.*# TODO-2-H.*$",   "dH  = None  # TODO-2-H",  "TODO-2-H"),
        (r"^dW1\s*=.*# TODO-2-W1.*$", "dW1 = None  # TODO-2-W1", "TODO-2-W1"),
        (r"^db1\s*=.*# TODO-2-b1.*$", "db1 = None  # TODO-2-b1", "TODO-2-b1"),
        (r"^dX2\s*=.*# TODO-2-X.*$",  "dX2 = None  # TODO-2-X",  "TODO-2-X"),
        (r"^one_hot\s*=.*# TODO-5-1.*$",  "one_hot = None  # TODO-5-1   形状 (B, V)",                   "TODO-5-1"),
        (r"^dz_batch\s*=.*# TODO-5-2.*$", "dz_batch = None  # TODO-5-2   形状 (B, V)，公式 (probs - one_hot) / B", "TODO-5-2"),
    ]
    for pattern, replacement, label in line_resets:
        text = replace_once(text, pattern, replacement, label)

    jacobian_replacement = (
        'def softmax_jacobian(z):\n'
        '    """TODO-3：返回 (n, n) 的雅可比矩阵，J[i,j] = ∂s_i/∂z_j。"""\n'
        '    # s = softmax_1d(z)\n'
        '    # return np.diag(s) - np.outer(s, s)\n'
        '    raise NotImplementedError("TODO-3 未完成：请实现 softmax_jacobian")\n\n\n'
    )
    text = reset_function(
        text,
        def_line="def softmax_jacobian(",
        full_replacement=jacobian_replacement,
        end_marker="z = np.array([1.0, 2.0, 3.0])",
        label="TODO-3 softmax_jacobian",
    )

    grad_ce_replacement = (
        'def grad_softmax_ce(z, y_onehot):\n'
        '    """TODO-4：返回 dL/dz，shape 与 z 相同。"""\n'
        '    # s = softmax_1d(z)\n'
        '    # return s - y_onehot\n'
        '    raise NotImplementedError("TODO-4 未完成：请实现 grad_softmax_ce")\n\n\n'
    )
    text = reset_function(
        text,
        def_line="def grad_softmax_ce(",
        full_replacement=grad_ce_replacement,
        end_marker="z = np.array([1.0, 2.0, 3.0])\ny",
        label="TODO-4 grad_softmax_ce",
    )

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
