"""
重置 phase2 第 2 课自写练习脚本。

用途：
  python3 reset_exercises_02.py

效果：
  - 将 02_self_attention_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET_FILE = Path(__file__).with_name("02_self_attention_self_write.py")


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"重置失败: {label} 匹配数量为 {count}（期望 1）")
    return new_text


def replace_block(text: str, start_marker: str, end_marker: str, replacement: str, label: str) -> str:
    """把 [start_marker 起点, end_marker 之前) 的整块替换成 replacement。"""
    start = text.find(start_marker)
    if start == -1:
        raise RuntimeError(f"重置失败: {label} 找不到起点 {start_marker!r}")
    end = text.find(end_marker, start)
    if end == -1:
        raise RuntimeError(f"重置失败: {label} 找不到终点 {end_marker!r}")
    return text[:start] + replacement + text[end:]


SOFTMAX_BLANK = (
    "def softmax(x, axis=-1):\n"
    "    # TODO-1: 实现 softmax\n"
    "    # 提示：np.max(..., axis=axis, keepdims=True) / np.exp / np.sum\n"
    "    return None\n"
    "\n\n"
)

SELF_ATTENTION_BLANK = (
    "def self_attention(X, W_Q, W_K, W_V):\n"
    "    # TODO-6: 用 softmax / 矩阵乘法实现完整的 scaled dot-product attention\n"
    "    # Q = ...\n"
    "    # K = ...\n"
    "    # V = ...\n"
    "    # d_k_local = K.shape[-1]\n"
    "    # scores_local = ...\n"
    "    # weights_local = ...\n"
    "    # output_local = ...\n"
    "    # return output_local, weights_local\n"
    "    return None, None\n"
    "\n\n"
)

CAUSAL_SELF_ATTENTION_BLANK = (
    "def causal_self_attention(X, W_Q, W_K, W_V):\n"
    "    # TODO-8:\n"
    "    # Q = ...\n"
    "    # K = ...\n"
    "    # V = ...\n"
    "    # d_k_local = K.shape[-1]\n"
    "    # scores_local = ...\n"
    "    # mask_local = np.triu(np.ones((X.shape[0], X.shape[0])), k=1)\n"
    "    # scores_local = scores_local - mask_local * 1e9\n"
    "    # weights_local = ...\n"
    "    # output_local = ...\n"
    "    # return output_local, weights_local\n"
    "    return None, None\n"
    "\n\n"
)


def main() -> int:
    if not TARGET_FILE.exists():
        print(f"未找到目标文件: {TARGET_FILE}")
        return 1

    text = TARGET_FILE.read_text(encoding="utf-8")

    # 单行赋值类 TODO
    line_replacements = [
        (r"^Q = .*# TODO-2\s*$", "Q = None  # TODO-2", "TODO-2 Q"),
        (r"^K = .*# TODO-2\s*$", "K = None  # TODO-2", "TODO-2 K"),
        (r"^V = .*# TODO-2\s*$", "V = None  # TODO-2", "TODO-2 V"),
        (r"^scores = .*# TODO-3\s*$", "scores = None  # TODO-3", "TODO-3"),
        (r"^scaled_scores = .*# TODO-4\s*$", "scaled_scores = None  # TODO-4", "TODO-4"),
        (r"^attention_weights = .*# TODO-5\s*$", "attention_weights = None  # TODO-5", "TODO-5 weights"),
        (r"^output = .*# TODO-5\s*$", "output = None  # TODO-5", "TODO-5 output"),
        (r"^mask = .*# TODO-7\s*$", "mask = None  # TODO-7", "TODO-7"),
    ]
    for pattern, replacement, label in line_replacements:
        text = replace_once(text, pattern, replacement, label)

    # 函数体类 TODO
    text = replace_block(text, "def softmax(x, axis=-1):", "_test_x = np.array",
                         SOFTMAX_BLANK, "TODO-1 softmax")
    text = replace_block(text, "def self_attention(X, W_Q, W_K, W_V):", "_o, _w = self_attention",
                         SELF_ATTENTION_BLANK, "TODO-6 self_attention")
    text = replace_block(text, "def causal_self_attention(X, W_Q, W_K, W_V):", "_oc, _wc = causal_self_attention",
                         CAUSAL_SELF_ATTENTION_BLANK, "TODO-8 causal_self_attention")

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
