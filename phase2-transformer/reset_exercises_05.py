"""
重置 phase2 第 5 课自写练习脚本。

用途：
  python3 reset_exercises_05.py

效果：
  - 将 05_gpt_from_scratch_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path

TARGET_FILE = Path(__file__).with_name("05_gpt_from_scratch_self_write.py")


def replace_block(
    text: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    label: str,
    search_from: str | None = None,
) -> str:
    """把 [start_marker 起点, end_marker 之前) 的整块替换成 replacement。

    search_from：先定位到该字符串，再从那里开始找 start_marker——
    用来区分多个类里同名的 forward 方法。
    """
    base = 0
    if search_from is not None:
        base = text.find(search_from)
        if base == -1:
            raise RuntimeError(f"重置失败: {label} 找不到范围锚点 {search_from!r}")
    start = text.find(start_marker, base)
    if start == -1:
        raise RuntimeError(f"重置失败: {label} 找不到起点 {start_marker!r}")
    end = text.find(end_marker, start)
    if end == -1:
        raise RuntimeError(f"重置失败: {label} 找不到终点 {end_marker!r}")
    return text[:start] + replacement + text[end:]


BLANK_BLOCKS = [
    # (范围锚点, 起点, 终点, 替换内容, 标签)
    (
        None,
        "def build_causal_mask(context_len):",
        "_mask = build_causal_mask(8)",
        "def build_causal_mask(context_len):\n"
        "    # TODO-1: 返回 (context_len, context_len) 的因果掩码（float 张量，1=屏蔽 0=可见）\n"
        "    return None\n"
        "\n\n",
        "TODO-1 build_causal_mask",
    ),
    (
        None,
        "def split_qkv_heads(qkv, n_head):",
        "_B, _T, _C, _H = 2, 5, 16, 4",
        "def split_qkv_heads(qkv, n_head):\n"
        "    # TODO-2: 返回 (q, k, v) 三元组，各为 (B, n_head, T, d_k)\n"
        "    return None\n"
        "\n\n",
        "TODO-2 split_qkv_heads",
    ),
    (
        None,
        "def causal_attention(q, k, v, mask):",
        "_B, _H, _T, _Dk = 2, 2, 4, 3",
        "def causal_attention(q, k, v, mask):\n"
        "    # TODO-3: 实现带因果掩码的多头注意力计算（不含输出投影 W_O）\n"
        "    return None\n"
        "\n\n",
        "TODO-3 causal_attention",
    ),
    (
        "class FeedForward(nn.Module):",
        "    def forward(self, x):",
        "_cfg = GPTConfig()",
        "    def forward(self, x):\n"
        "        # TODO-4: 实现 FFN 前向\n"
        "        return None\n"
        "\n\n",
        "TODO-4 FeedForward.forward",
    ),
    (
        "class TransformerBlock(nn.Module):",
        "    def forward(self, x):",
        "torch.manual_seed(1)",
        "    def forward(self, x):\n"
        "        # TODO-5: 实现 Pre-Norm Block 前向（两条残差支路）\n"
        "        return None\n"
        "\n\n",
        "TODO-5 TransformerBlock.forward",
    ),
    (
        "class GPT(nn.Module):",
        "    def tie_weights(self):",
        "    def _init_weights(self):",
        "    def tie_weights(self):\n"
        "        # TODO-6: 实现权重共享（weight tying）\n"
        "        #   要点：让两者的 .weight 指向【同一个】Parameter 对象（赋值共享内存），\n"
        "        #         而不是数值拷贝（copy_ 之后还是两份独立参数，训练会各走各的）\n"
        "        #   想清楚把谁赋给谁（提示：nn.Linear 的 weight 形状本来就是 (out, in)，\n"
        "        #   和 nn.Embedding 的 (vocab_size, n_embd) 恰好一致），写一行赋值即可\n"
        "        pass  # ← 实现后删掉这行\n"
        "\n",
        "TODO-6 GPT.tie_weights",
    ),
    (
        "class GPT(nn.Module):",
        "    def forward(self, idx, targets=None):",
        "    @torch.no_grad()",
        "    def forward(self, idx, targets=None):\n"
        "        # TODO-7: 实现 GPT 前向，return logits, loss\n"
        "        return None\n"
        "\n",
        "TODO-7 GPT.forward",
    ),
    (
        "class GPT(nn.Module):",
        "    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):",
        "torch.manual_seed(2)",
        "    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):\n"
        "        # TODO-10: 实现自回归生成（先做完 TODO-8/9 再回来写这里）\n"
        "        # 每生成一个 token 重复以下步骤：\n"
        "        #   1. 截断上下文：idx_crop = idx[:, -self.config.context_len:]\n"
        "        #   2. 前向拿 logits，只取最后一个位置：logits[:, -1, :]\n"
        "        #   3. 除以 temperature（越低分布越尖 → 越确定）\n"
        "        #   4. top_k 不为 None 时，用 apply_top_k 过滤\n"
        "        #   5. softmax → torch.multinomial 采样 1 个 token\n"
        "        #   6. torch.cat 拼回 idx\n"
        "        return None\n"
        "\n\n",
        "TODO-10 GPT.generate",
    ),
    (
        None,
        "def get_batch(data, context_len, batch_size):",
        "_data = torch.arange(200",
        "def get_batch(data, context_len, batch_size):\n"
        "    # TODO-8: 返回 (x, y)，各为 (batch_size, context_len)\n"
        "    return None\n"
        "\n\n",
        "TODO-8 get_batch",
    ),
    (
        None,
        "def apply_top_k(logits, k):",
        "_tk_logits = torch.tensor",
        "def apply_top_k(logits, k):\n"
        "    # TODO-9: 返回过滤后的 logits，形状不变 (B, vocab_size)\n"
        "    return None\n"
        "\n\n",
        "TODO-9 apply_top_k",
    ),
    (
        None,
        "def apply_top_p(logits, p):",
        "_tp_probs = torch.tensor",
        "def apply_top_p(logits, p):\n"
        "    # TODO-11: 返回过滤后的 logits，形状不变 (B, vocab_size)（可跳过）\n"
        "    return None\n"
        "\n\n",
        "TODO-11 apply_top_p",
    ),
    (
        None,
        "def encode_with_gpt2_bpe(text):",
        '_sample = "To be or not to be that is the question"',
        "def encode_with_gpt2_bpe(text):\n"
        "    # TODO-12: 返回 GPT-2 BPE 编码后的 token id 列表（可跳过）\n"
        "    return None\n"
        "\n\n",
        "TODO-12 encode_with_gpt2_bpe",
    ),
]


def main() -> int:
    if not TARGET_FILE.exists():
        print(f"未找到目标文件: {TARGET_FILE}")
        return 1

    text = TARGET_FILE.read_text(encoding="utf-8")

    for search_from, start_marker, end_marker, replacement, label in BLANK_BLOCKS:
        text = replace_block(text, start_marker, end_marker, replacement, label, search_from)

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
