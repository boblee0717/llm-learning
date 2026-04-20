"""
重置 phase0 第 3 课自写练习脚本（03_reshape_transpose_split_self_write.py）。

用法：
    python3 reset_exercises_03.py
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET_FILE = Path(__file__).with_name("03_reshape_transpose_split_self_write.py")


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
        (r"^m\s*=.*# TODO-1.*$",            "m = None  # TODO-1",            "TODO-1"),
        (r"^y_t\s*=.*# TODO-2.*$",          "y_t = None  # TODO-2",          "TODO-2"),
        (r"^QKV\s*=.*# TODO-5-1.*$",        "QKV = None  # TODO-5-1",        "TODO-5-1"),
        (r"^Q,\s*K,\s*V\s*=.*# TODO-5-2.*$","Q, K, V = None, None, None  # TODO-5-2", "TODO-5-2"),
        (r"^outer\s*=.*# TODO-6.*$",        "outer = None  # TODO-6",        "TODO-6"),
        (r"^scores_masked\s*=.*# TODO-7.*$","scores_masked = None  # TODO-7","TODO-7"),
    ]
    for pattern, replacement, label in line_resets:
        text = replace_once(text, pattern, replacement, label)

    split_replacement = (
        'def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:\n'
        '    """TODO-3：把最后一维拆成 (H, d_h)，再把 H 换到 T 前面。"""\n'
        '    # B_, T_, d_ = x.shape\n'
        '    # d_h_ = d_ // num_heads\n'
        '    # step1 = x.reshape(...)\n'
        '    # return step1.transpose(...)\n'
        '    raise NotImplementedError("TODO-3 未完成：请实现 split_heads")\n\n\n'
    )
    text = reset_function(
        text,
        def_line="def split_heads(",
        full_replacement=split_replacement,
        end_marker="X_split = split_heads",
        label="TODO-3 split_heads",
    )

    merge_replacement = (
        'def merge_heads(x: np.ndarray) -> np.ndarray:\n'
        '    """TODO-4：把 (B, H, T, d_h) 合并回 (B, T, H*d_h)。"""\n'
        '    # step1 = x.transpose(...)\n'
        '    # B_, T_, H_, d_h_ = step1.shape\n'
        '    # return step1.reshape(...)\n'
        '    raise NotImplementedError("TODO-4 未完成：请实现 merge_heads")\n\n\n'
    )
    text = reset_function(
        text,
        def_line="def merge_heads(",
        full_replacement=merge_replacement,
        end_marker="X_back = merge_heads",
        label="TODO-4 merge_heads",
    )

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
