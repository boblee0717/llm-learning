"""
重置 phase0 第 2 课自写练习脚本（02_matmul_and_shapes_self_write.py）。

用法：
    python3 reset_exercises_02.py
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET_FILE = Path(__file__).with_name("02_matmul_and_shapes_self_write.py")


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


def reset_todo2_loop(text: str) -> str:
    """
    重置 TODO-2 的 for 循环回到注释占位。

    边界：
      - 上：c_row0 = np.zeros(4)\n  这一行之后
      - 下：\n\nrequire_close 之前
    """
    anchor = "c_row0 = np.zeros(4)\n"
    start = text.find(anchor)
    if start == -1:
        raise RuntimeError("重置失败: 找不到 TODO-2 起点（c_row0 = np.zeros(4)）")
    start += len(anchor)
    end = text.find("\nrequire_close", start)
    if end == -1:
        raise RuntimeError("重置失败: 找不到 TODO-2 结束标记（require_close）")

    placeholder = (
        "# TODO-2: 用 for 循环计算 c_row0[j] = A[0] @ B[:, j]\n"
        "# for j in range(4):\n"
        "#     c_row0[j] = ...\n"
    )
    return text[:start] + placeholder + text[end:]


def main() -> int:
    if not TARGET_FILE.exists():
        print(f"未找到目标文件: {TARGET_FILE}")
        return 1

    text = TARGET_FILE.read_text(encoding="utf-8")

    line_resets = [
        (r"^C\s*=.*# TODO-1.*$",                  "C = None  # TODO-1",                  "TODO-1"),
        (r"^Y\s*=.*# TODO-3.*$",                  "Y = None  # TODO-3",                  "TODO-3"),
        (r"^scores\s*=.*# TODO-4.*$",             "scores = None  # TODO-4",             "TODO-4"),
        (r"^scores_einsum\s*=.*# TODO-5.*$",      "scores_einsum = None  # TODO-5",      "TODO-5"),
        (r"^out\s*=.*# TODO-6.*$",                "out = None  # TODO-6",                "TODO-6"),
    ]
    for pattern, replacement, label in line_resets:
        text = replace_once(text, pattern, replacement, label)

    text = reset_todo2_loop(text)

    matmul_shape_replacement = (
        'def matmul_shape(shape_a, shape_b):\n'
        '    """TODO-7：返回结果 shape（tuple），不合法时返回 None。"""\n'
        '    # 实现思路：\n'
        '    # 1) 取出 a 的最后两维 (m, ka) 和 b 的最后两维 (kb, n)\n'
        '    # 2) 检查 ka == kb，否则返回 None\n'
        '    # 3) 对 a/b 的前缀维按"右对齐 + 1 可广播"规则合并\n'
        '    raise NotImplementedError("TODO-7 未完成：请实现 matmul_shape")\n\n\n'
    )
    text = reset_function(
        text,
        def_line="def matmul_shape(",
        full_replacement=matmul_shape_replacement,
        end_marker="cases = [",
        label="TODO-7 matmul_shape",
    )

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
