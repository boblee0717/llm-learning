"""
重置 phase0 第 1 课自写练习脚本（01_vectors_and_axes_self_write.py）。

用法：
    python3 reset_exercises_01.py

效果：
    - 把所有 TODO 占位行恢复为 None / NotImplementedError
    - 保留讲解、打印与校验逻辑
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET_FILE = Path(__file__).with_name("01_vectors_and_axes_self_write.py")


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
        (r"^row\s*=.*# TODO-1-row.*$",          "row = None  # TODO-1-row",          "TODO-1-row"),
        (r"^col\s*=.*# TODO-1-col.*$",          "col = None  # TODO-1-col",          "TODO-1-col"),
        (r"^dot\s*=.*# TODO-2.*$",              "dot = None  # TODO-2",              "TODO-2"),
        (r"^cos_ab\s*=.*# TODO-3.*$",           "cos_ab = None  # TODO-3",           "TODO-3"),
        (r"^sum_by_row\s*=.*# TODO-4-row.*$",   "sum_by_row = None  # TODO-4-row    形状 (2,)，期望 [6, 15]", "TODO-4-row"),
        (r"^sum_by_col\s*=.*# TODO-4-col.*$",   "sum_by_col = None  # TODO-4-col    形状 (3,)，期望 [5, 7, 9]", "TODO-4-col"),
        (r"^Y\s*=.*# TODO-6.*$",                "Y = None  # TODO-6",                "TODO-6"),
    ]
    for pattern, replacement, label in line_resets:
        text = replace_once(text, pattern, replacement, label)

    softmax_replacement = (
        'def softmax_rowwise(scores: np.ndarray) -> np.ndarray:\n'
        '    """TODO-5：对最后一维做 softmax。"""\n'
        '    # row_max = ...\n'
        '    # exp_s   = ...\n'
        '    # return  exp_s / ...\n'
        '    raise NotImplementedError("TODO-5 未完成：请实现 softmax_rowwise")\n\n\n'
    )
    text = reset_function(
        text,
        def_line="def softmax_rowwise(",
        full_replacement=softmax_replacement,
        end_marker="scores = np.array",
        label="TODO-5 softmax_rowwise",
    )

    can_bcast_replacement = (
        'def can_broadcast(shape_a, shape_b) -> bool:\n'
        '    """TODO-7：返回 True/False。"""\n'
        '    # 提示：先把两个 shape 反转，然后逐位检查；缺位用 1 兜底。\n'
        '    raise NotImplementedError("TODO-7 未完成：请实现 can_broadcast")\n\n\n'
    )
    text = reset_function(
        text,
        def_line="def can_broadcast(",
        full_replacement=can_bcast_replacement,
        end_marker="cases = [",
        label="TODO-7 can_broadcast",
    )

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
