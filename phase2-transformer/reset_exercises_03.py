"""
重置 phase2 第 3 课自写练习脚本。

用途：
  python3 reset_exercises_03.py

效果：
  - 将 03_multi_head_attention_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET_FILE = Path(__file__).with_name("03_multi_head_attention_self_write.py")


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


BLANK_BLOCKS = [
    # (起点, 终点, 替换内容, 标签)
    (
        "def softmax(x, axis=-1):",
        "_softmax_test = np.array",
        "def softmax(x, axis=-1):\n"
        "    # TODO-1: 实现 softmax\n"
        "    return None\n"
        "\n\n",
        "TODO-1 softmax",
    ),
    (
        "def single_head_attention(Q, K, V, mask=None):",
        "Q_one = np.random.randn",
        "def single_head_attention(Q, K, V, mask=None):\n"
        "    # TODO-2: 实现单头注意力\n"
        "    #   scores → /sqrt(d_k) → (mask: scores - mask * 1e9) → softmax → @ V\n"
        "    #   返回 (output, weights)\n"
        "    return None, None\n"
        "\n\n",
        "TODO-2 single_head_attention",
    ),
    (
        "def split_heads(x, num_heads):",
        "Q_full = X @ W_Q",
        "def split_heads(x, num_heads):\n"
        "    # TODO-3: 把最后一维 d_model 拆成 num_heads * d_head\n"
        "    #   提示：reshape 后再 transpose 把 head 维提到最前；\n"
        "    #         d_model 不能被 num_heads 整除时 raise ValueError\n"
        "    return None\n"
        "\n\n",
        "TODO-3 split_heads",
    ),
    (
        "def merge_heads(x):",
        "Q_back = merge_heads",
        "def merge_heads(x):\n"
        "    # TODO-4: 把多个 head 合并回最后一维（split_heads 的逆操作）\n"
        "    return None\n"
        "\n\n",
        "TODO-4 merge_heads",
    ),
    (
        "def multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads, mask=None):",
        "mha_out, mha_weights = multi_head_attention",
        "def multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads, mask=None):\n"
        "    # TODO-5: 实现完整多头注意力\n"
        "    #   投影 → split_heads → 每个 head 独立 single_head_attention\n"
        "    #   → stack + merge_heads → @ W_O\n"
        "    #   返回 (output, head_weights 列表)\n"
        "    return None, None\n"
        "\n\n",
        "TODO-5 multi_head_attention",
    ),
    (
        "def residual_connection(x, sublayer_out):",
        "residual_out = residual_connection",
        "def residual_connection(x, sublayer_out):\n"
        "    # TODO-7: 返回残差连接结果\n"
        "    return None\n"
        "\n\n",
        "TODO-7 residual_connection",
    ),
    (
        "def layer_norm(x, gamma=None, beta=None, eps=1e-5):",
        "normed = layer_norm",
        "def layer_norm(x, gamma=None, beta=None, eps=1e-5):\n"
        "    # TODO-8: 实现 LayerNorm\n"
        "    return None\n"
        "\n\n",
        "TODO-8 layer_norm",
    ),
    (
        "def post_norm_block(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):",
        "def pre_norm_block",
        "def post_norm_block(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):\n"
        "    # TODO-9a: Attention -> Residual -> LayerNorm\n"
        "    return None\n"
        "\n\n",
        "TODO-9a post_norm_block",
    ),
    (
        "def pre_norm_block(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):",
        "post_out = post_norm_block",
        "def pre_norm_block(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):\n"
        "    # TODO-9b: LayerNorm -> Attention -> Residual\n"
        "    return None\n"
        "\n\n",
        "TODO-9b pre_norm_block",
    ),
]


def main() -> int:
    if not TARGET_FILE.exists():
        print(f"未找到目标文件: {TARGET_FILE}")
        return 1

    text = TARGET_FILE.read_text(encoding="utf-8")

    text = replace_once(
        text, r"^causal_mask = .*# TODO-6\s*$", "causal_mask = None  # TODO-6", "TODO-6"
    )

    for start_marker, end_marker, replacement, label in BLANK_BLOCKS:
        text = replace_block(text, start_marker, end_marker, replacement, label)

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
