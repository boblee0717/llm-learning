"""
重置 phase2 第 4 课自写练习脚本。

用途：
  python3 reset_exercises_04.py

效果：
  - 将 04_transformer_block_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path

TARGET_FILE = Path(__file__).with_name("04_transformer_block_self_write.py")


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
        "def layer_norm(x, gamma=None, beta=None, eps=1e-5):",
        "ln_out = layer_norm",
        "def layer_norm(x, gamma=None, beta=None, eps=1e-5):\n"
        "    # TODO-2: 实现 LayerNorm\n"
        "    return None\n"
        "\n\n",
        "TODO-2 layer_norm",
    ),
    (
        "def gelu(x):",
        "_gelu_test = np.array",
        "def gelu(x):\n"
        "    # TODO-3: 实现 GELU（tanh 近似）\n"
        "    return None\n"
        "\n\n",
        "TODO-3 gelu",
    ),
    (
        "def feed_forward(x, W1, b1, W2, b2):",
        "ffn_out = feed_forward",
        "def feed_forward(x, W1, b1, W2, b2):\n"
        "    # TODO-4: 实现两层 FFN：GELU(x @ W1 + b1) @ W2 + b2\n"
        "    return None\n"
        "\n\n",
        "TODO-4 feed_forward",
    ),
    (
        "def multi_head_attention(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):",
        "mha_out = multi_head_attention",
        "def multi_head_attention(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):\n"
        "    # TODO-5: 实现多头注意力\n"
        "    #   投影 → reshape + transpose 分头 → scores/√d_k → mask → softmax\n"
        "    #   → weights @ V → 拼回 (seq_len, d_model) → @ W_O\n"
        "    return None\n"
        "\n\n",
        "TODO-5 multi_head_attention",
    ),
    (
        "def pre_norm_block(x, params, num_heads, mask=None):",
        "pre_out = pre_norm_block",
        "def pre_norm_block(x, params, num_heads, mask=None):\n"
        "    # TODO-6: 实现 Pre-Norm Transformer Block\n"
        "    # 提示：params 是一个 dict，含 W_Q/W_K/W_V/W_O/W1/b1/W2/b2\n"
        "    return None\n"
        "\n\n",
        "TODO-6 pre_norm_block",
    ),
    (
        "def post_norm_block(x, params, num_heads, mask=None):",
        "post_out = post_norm_block",
        "def post_norm_block(x, params, num_heads, mask=None):\n"
        "    # TODO-7: 实现 Post-Norm Transformer Block\n"
        "    return None\n"
        "\n\n",
        "TODO-7 post_norm_block",
    ),
    (
        "def stack_blocks(x, layer_params_list, num_heads, mask=None):",
        "stacked_out = stack_blocks",
        "def stack_blocks(x, layer_params_list, num_heads, mask=None):\n"
        "    # TODO-8: 把 pre_norm_block 串联 len(layer_params_list) 次，每层用各自的参数\n"
        "    #   提示：用一个循环，h = pre_norm_block(h, params, num_heads, mask=mask)\n"
        "    return None\n"
        "\n\n",
        "TODO-8 stack_blocks",
    ),
    (
        "def dropout(x, rate=0.1, training=True, rng=None):",
        "# 推理模式下应该完全不变",
        "def dropout(x, rate=0.1, training=True, rng=None):\n"
        "    # TODO-9: 实现 inverted dropout\n"
        "    #   提示：\n"
        "    #     if (not training) or rate == 0: 直接返回 x\n"
        "    #     rng 为 None 时用 np.random；keep = (rng.rand(*x.shape) > rate)\n"
        "    #     注意保持 dtype：用 np.asarray(..., dtype=x.dtype) 包一下\n"
        "    return None\n"
        "\n\n",
        "TODO-9 dropout",
    ),
]


def main() -> int:
    if not TARGET_FILE.exists():
        print(f"未找到目标文件: {TARGET_FILE}")
        return 1

    text = TARGET_FILE.read_text(encoding="utf-8")

    for start_marker, end_marker, replacement, label in BLANK_BLOCKS:
        text = replace_block(text, start_marker, end_marker, replacement, label)

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
