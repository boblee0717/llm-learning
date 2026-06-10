"""
======================================================
phase2 / 第 5 课（自写版）：从零构建 GPT（PyTorch）
======================================================

用法：
1. 运行：python 05_gpt_from_scratch_self_write.py
2. 按 TODO-1 到 TODO-12 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错
   （没填的 TODO 会返回 None，校验提示「未实现」，这是正常的）

目标：
- 手写因果掩码（torch.triu）
- 手写注意力的核心计算：QKV 分头 → scores → mask → softmax → 加权 → 合并
- 手写 FFN 与 Pre-Norm TransformerBlock 的 forward
- 手写权重共享 weight tying（embedding 和 lm_head 共用一份参数）
- 手写 GPT.forward（embedding 相加 → N 层 Block → ln_f → lm_head → loss）
- 手写训练数据采样 get_batch（next token prediction 的本质就藏在这里）
- 手写 Top-K 过滤与自回归生成 generate
- （进阶，可跳过）手写 Top-P / nucleus 过滤
- （进阶，可跳过）用 tiktoken 体验 BPE 分词，对比字符级分词

与第 4 课自写版的关系：
- 第 4 课用 NumPy 手写了所有组件的"裸公式"
- 本课换成 PyTorch：同样的数学，但用 nn.Module / 自动求导 / 优化器表达
- 各个类的 __init__（建层）都已给出，你只需要写 forward（数据流）——
  这正是读懂任何 PyTorch 模型代码的关键能力

为聚焦数据流，本自写版省略了主课中的 Dropout（主课里它只是在
attention weights / 残差输出上各加一层 nn.Dropout，原理见第 4 课 TODO-9）。

全部 TODO 校验通过后，脚本会用你写的模型真的训练 300 步并生成文本。
（想用真实数据练？把 tiny-shakespeare 存为本目录的 tiny_shakespeare.txt，
  训练环节会自动优先使用，下载方式见「终极验证」一节的注释。）
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")  # ValidationError 走 stderr，也要防乱码

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


class ValidationError(Exception):
    pass


def require_not_none(name, value):
    if value is None:
        raise ValidationError(f"{name} 未实现：结果是 None。")


def require_true(name, cond, hint=""):
    if not cond:
        raise ValidationError(f"{name} 条件不满足：{hint}")


def require_shape(name, actual, expected_shape):
    require_not_none(name, actual)
    if tuple(actual.shape) != tuple(expected_shape):
        raise ValidationError(
            f"{name} 形状不对：actual={tuple(actual.shape)}, expected={tuple(expected_shape)}"
        )


def require_close(name, actual, expected, atol=1e-5):
    require_not_none(name, actual)
    if not torch.allclose(actual, expected, atol=atol):
        raise ValidationError(
            f"{name} 数值不对\nactual=\n{actual}\nexpected=\n{expected}"
        )


# ---------- 模型配置（不需要你改） ----------
class GPTConfig:
    def __init__(self, vocab_size=64, context_len=32, n_layer=2, n_head=2, n_embd=16):
        assert n_embd % n_head == 0
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd


# ============================================================
section("TODO-1：构造因果掩码 build_causal_mask")
# ============================================================
# 因果掩码：上三角为 1（屏蔽未来），对角线及以下为 0（可见）。
# 与第 2/3/4 课的 np.triu(..., k=1) 完全同一个东西，换成 torch 写法。
#
# 提示：torch.triu(torch.ones(T, T), diagonal=1)


def build_causal_mask(context_len):
    # TODO-1: 返回 (context_len, context_len) 的因果掩码（float 张量，1=屏蔽 0=可见）
    return None


_mask = build_causal_mask(8)
require_shape("TODO-1 mask", _mask, (8, 8))
require_close("TODO-1 对角线应为 0（自己能看自己）", _mask.diagonal(), torch.zeros(8))
require_close("TODO-1 上三角应为 1", _mask[0, 1:], torch.ones(7))
require_close("TODO-1 下三角应为 0", _mask[-1], torch.zeros(8))
print("因果掩码 OK：")
print(_mask.int())


# ============================================================
section("TODO-2：把合并的 QKV 拆成多头 split_qkv_heads")
# ============================================================
# 主课的 CausalSelfAttention 用一个大矩阵 c_attn 一次算出 QKV（效率更高），
# 得到形状 (B, T, 3C)。本题把它拆开并分头：
#
#   输入 qkv: (B, T, 3C)
#   输出 q, k, v: 各为 (B, n_head, T, d_k)，其中 d_k = C // n_head
#
# 提示：
#   1. q, k, v = qkv.split(C, dim=2)        # 各 (B, T, C)
#   2. 再 view 成 (B, T, n_head, d_k)，transpose(1, 2) 把 head 提前
#      （和第 4 课 numpy 的 reshape + transpose(1,0,2) 是同一件事，多了 batch 维）


def split_qkv_heads(qkv, n_head):
    # TODO-2: 返回 (q, k, v) 三元组，各为 (B, n_head, T, d_k)
    return None


_B, _T, _C, _H = 2, 5, 16, 4
_qkv = torch.randn(_B, _T, 3 * _C)
_res = split_qkv_heads(_qkv, _H)
require_not_none("TODO-2 split_qkv_heads", _res)
require_true("TODO-2 返回三元组", isinstance(_res, tuple) and len(_res) == 3, "应 return q, k, v")
_q, _k, _v = _res
require_shape("TODO-2 q", _q, (_B, _H, _T, _C // _H))
require_shape("TODO-2 k", _k, (_B, _H, _T, _C // _H))
require_shape("TODO-2 v", _v, (_B, _H, _T, _C // _H))
_expected_q = _qkv[..., :_C].view(_B, _T, _H, _C // _H).transpose(1, 2)
_expected_v = _qkv[..., 2 * _C:].view(_B, _T, _H, _C // _H).transpose(1, 2)
require_close("TODO-2 q 内容", _q, _expected_q)
require_close("TODO-2 v 内容", _v, _expected_v)
print(f"split_qkv_heads OK：(B={_B}, T={_T}, 3C={3 * _C}) → 3 × (B, {_H}, {_T}, {_C // _H})")


# ============================================================
section("TODO-3：注意力核心计算 causal_attention")
# ============================================================
# 输入 q, k, v: (B, n_head, T, d_k)，mask: (T, T)
# 输出: (B, T, C)，其中 C = n_head * d_k
#
# 论文公式：Attention(Q,K,V) = softmax(QKᵀ/√d_k) V
#
# 提示（对应主课 forward 的 112~120 行）：
#   1. scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)     # (B, H, T, T)
#   2. scores = scores.masked_fill(mask == 1, float("-inf"))  # 屏蔽未来
#   3. weights = softmax(scores, dim=-1)
#   4. out = weights @ v                                      # (B, H, T, d_k)
#   5. 合并多头：transpose(1, 2) → contiguous() → view(B, T, C)


def causal_attention(q, k, v, mask):
    # TODO-3: 实现带因果掩码的多头注意力计算（不含输出投影 W_O）
    return None


_B, _H, _T, _Dk = 2, 2, 4, 3
_q = torch.randn(_B, _H, _T, _Dk)
_k = torch.randn(_B, _H, _T, _Dk)
_v = torch.randn(_B, _H, _T, _Dk)
_mask4 = build_causal_mask(_T)

_attn_out = causal_attention(_q, _k, _v, _mask4)
require_shape("TODO-3 输出", _attn_out, (_B, _T, _H * _Dk))

_scores = _q @ _k.transpose(-2, -1) / math.sqrt(_Dk)
_scores = _scores.masked_fill(_mask4 == 1, float("-inf"))
_ref_out = (F.softmax(_scores, dim=-1) @ _v).transpose(1, 2).contiguous().view(_B, _T, _H * _Dk)
require_close("TODO-3 数值", _attn_out, _ref_out)

# 因果性的直接体现：第 0 个 token 只能看到自己 → 输出就是它自己的 v
require_close(
    "TODO-3 第 0 个 token 只能看自己",
    _attn_out[:, 0],
    _v[:, :, 0, :].reshape(_B, _H * _Dk),
)
print("causal_attention OK：第 0 个 token 的输出 = 它自己的 v（mask 生效）")


# ---------- CausalSelfAttention 模块（forward 已给出，复用你的 TODO-2/3） ----------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        # QKV 合并成一个大矩阵，一次矩阵乘法算完，比三个小矩阵更高效
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # mask 注册为 buffer：跟着模型走（保存/搬 device），但不参与训练
        self.register_buffer("mask", build_causal_mask(config.context_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = split_qkv_heads(self.c_attn(x), self.n_head)
        out = causal_attention(q, k, v, self.mask[:T, :T])
        return self.c_proj(out)  # W_O：混合各 head 的信息


# ============================================================
section("TODO-4：FeedForward 的 forward")
# ============================================================
# 第 4 课手写过的两层 MLP，这次用 nn.Linear 表达：
#   FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
#          = self.c_proj(F.gelu(self.c_fc(x)))
# 升维 4 倍 → GELU → 降回 d_model。


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        # TODO-4: 实现 FFN 前向
        return None


_cfg = GPTConfig()
torch.manual_seed(0)
_ffn = FeedForward(_cfg)
_x = torch.randn(2, 6, _cfg.n_embd)
_ffn_out = _ffn(_x)
require_shape("TODO-4 ffn_out", _ffn_out, (2, 6, _cfg.n_embd))
with torch.no_grad():
    _ref_ffn = _ffn.c_proj(F.gelu(_ffn.c_fc(_x)))
require_close("TODO-4 数值", _ffn_out, _ref_ffn)
print("FeedForward OK：升维 4 倍 + GELU + 降维")


# ============================================================
section("TODO-5：Pre-Norm TransformerBlock 的 forward")
# ============================================================
# 第 4 课 TODO-6 的 PyTorch 版，伪代码完全一样：
#   x = x + Attention(LayerNorm(x))
#   x = x + FFN(LayerNorm(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # TODO-5: 实现 Pre-Norm Block 前向（两条残差支路）
        return None


torch.manual_seed(1)
_block = TransformerBlock(_cfg)
_x = torch.randn(2, 6, _cfg.n_embd)
_block_out = _block(_x)
require_shape("TODO-5 block_out", _block_out, (2, 6, _cfg.n_embd))
with torch.no_grad():
    _ref_b = _x + _block.attn(_block.ln_1(_x))
    _ref_b = _ref_b + _block.ffn(_block.ln_2(_ref_b))
require_close("TODO-5 数值", _block_out, _ref_b)
print("TransformerBlock OK：输入输出 shape 相同 → 可堆叠任意层")


# ============================================================
section("TODO-6/7：权重共享 tie_weights 与 GPT.forward")
# ============================================================
# TODO-6（在 GPT 类的 tie_weights 方法里）：
#   token_embedding.weight 和 lm_head.weight 形状都是 (vocab_size, n_embd)：
#   一个把 token 查成向量（查表），一个把向量和每个词的"原型向量"做点积打回词表（反查），
#   语义上互为逆操作 → 可以共用同一份参数（GPT-2/GPT-3 都这么做）。
#
# TODO-7（GPT.forward）数据流（对应主课 Part 3）：
#   idx (B, T)
#     → token_embedding + position_embedding   (B, T, C)
#     → N × TransformerBlock                   (B, T, C)
#     → ln_f（final LayerNorm，第 3 课复盘提过的 Pre-Norm 收尾）
#     → lm_head                                (B, T, vocab_size)
#   targets 不为 None 时再算 cross_entropy loss
#
# 提示：
#   1. 位置索引 torch.arange(T, device=idx.device) —— device 要跟着输入走！
#   2. pos_emb 形状 (T, C)，与 tok_emb (B, T, C) 相加时自动广播
#   3. cross_entropy 要求把 (B, T, V) 摊平成 (B*T, V)、targets 摊平成 (B*T,)
#   4. 没有 targets 时 loss 返回 None


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_len, config.n_embd)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tie_weights()
        self._init_weights()

    def tie_weights(self):
        # TODO-6: 实现权重共享（weight tying）
        #   要点：让两者的 .weight 指向【同一个】Parameter 对象（赋值共享内存），
        #         而不是数值拷贝（copy_ 之后还是两份独立参数，训练会各走各的）
        #   想清楚把谁赋给谁（提示：nn.Linear 的 weight 形状本来就是 (out, in)，
        #   和 nn.Embedding 的 (vocab_size, n_embd) 恰好一致），写一行赋值即可
        pass  # ← 实现后删掉这行

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # TODO-7: 实现 GPT 前向，return logits, loss
        return None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # TODO-10: 实现自回归生成（先做完 TODO-8/9 再回来写这里）
        # 每生成一个 token 重复以下步骤：
        #   1. 截断上下文：idx_crop = idx[:, -self.config.context_len:]
        #   2. 前向拿 logits，只取最后一个位置：logits[:, -1, :]
        #   3. 除以 temperature（越低分布越尖 → 越确定）
        #   4. top_k 不为 None 时，用 apply_top_k 过滤
        #   5. softmax → torch.multinomial 采样 1 个 token
        #   6. torch.cat 拼回 idx
        return None


torch.manual_seed(2)
_gpt_cfg = GPTConfig(vocab_size=64, context_len=32, n_layer=2, n_head=2, n_embd=16)
_model = GPT(_gpt_cfg)
_idx = torch.randint(0, _gpt_cfg.vocab_size, (2, 10))
_targets = torch.randint(0, _gpt_cfg.vocab_size, (2, 10))

# ---- TODO-6 校验：weight tying ----
require_true(
    "TODO-6 weight tying：共享同一个 Parameter",
    _model.token_embedding.weight is _model.lm_head.weight,
    "应让 token_embedding.weight 和 lm_head.weight 指向同一个 Parameter 对象"
    "（用赋值共享，而不是 copy_ 数值拷贝；实现后记得删掉 pass）",
)
_tied = _model.token_embedding.weight.numel()
print(f"weight tying OK：embedding 和 lm_head 共用一份参数，省下 {_tied:,} 个参数")

# ---- TODO-7 校验：GPT.forward ----
_fwd = _model(_idx, _targets)
require_not_none("TODO-7 GPT.forward", _fwd)
require_true("TODO-7 返回 (logits, loss)", isinstance(_fwd, tuple) and len(_fwd) == 2,
             "应 return logits, loss")
_logits, _loss = _fwd
require_shape("TODO-7 logits", _logits, (2, 10, _gpt_cfg.vocab_size))
require_not_none("TODO-7 loss", _loss)

# 和参考前向对数值
with torch.no_grad():
    _h = _model.token_embedding(_idx) + _model.position_embedding(
        torch.arange(_idx.shape[1], device=_idx.device)
    )
    for _b in _model.blocks:
        _h = _b(_h)
    _ref_logits = _model.lm_head(_model.ln_f(_h))
    _ref_loss = F.cross_entropy(
        _ref_logits.view(-1, _gpt_cfg.vocab_size), _targets.view(-1)
    )
require_close("TODO-7 logits 数值", _logits, _ref_logits)
require_close("TODO-7 loss 数值", _loss, _ref_loss)

# 没有 targets 时 loss 应为 None
_logits2, _loss2 = _model(_idx)
require_true("TODO-7 无 targets 时 loss=None", _loss2 is None, "推理时不该算 loss")

# 因果性验证：改动最后一个 token，前面所有位置的 logits 不应变化
_idx_mod = _idx.clone()
_idx_mod[:, -1] = (_idx_mod[:, -1] + 1) % _gpt_cfg.vocab_size
_logits_mod, _ = _model(_idx_mod)
require_close("TODO-7 因果性（改未来不影响过去）", _logits_mod[:, :-1], _logits[:, :-1])

print(f"GPT.forward OK：随机初始化 loss = {_loss.item():.4f}"
      f"（≈ ln({_gpt_cfg.vocab_size}) = {math.log(_gpt_cfg.vocab_size):.4f}，即均匀瞎猜）")


# ============================================================
section("TODO-8：训练数据采样 get_batch")
# ============================================================
# Language Modeling 的本质就在这两行里：
#   x = data[i     : i+context_len]      # 输入
#   y = data[i+1   : i+context_len+1]    # 目标 = 输入右移一位
# 每个位置都在学「预测下一个 token」。
#
# 提示：
#   1. ix = torch.randint(len(data) - context_len, (batch_size,)) 随机起点
#   2. torch.stack 把 batch_size 个切片摞成 (batch_size, context_len)


def get_batch(data, context_len, batch_size):
    # TODO-8: 返回 (x, y)，各为 (batch_size, context_len)
    return None


_data = torch.arange(200, dtype=torch.long)
_batch = get_batch(_data, context_len=8, batch_size=4)
require_not_none("TODO-8 get_batch", _batch)
require_true("TODO-8 返回 (x, y)", isinstance(_batch, tuple) and len(_batch) == 2, "应 return x, y")
_bx, _by = _batch
require_shape("TODO-8 x", _bx, (4, 8))
require_shape("TODO-8 y", _by, (4, 8))
# data 是 0..199 的等差数列，所以 y 必须恰好等于 x + 1（右移一位的直接证据）
require_close("TODO-8 y = x 右移一位", _by.float(), (_bx + 1).float())
require_true("TODO-8 索引没越界", _by.max().item() < 200, "y 的最大索引超出了 data 范围")
print("get_batch OK：y 恰好是 x 右移一位 → 每个位置都在学预测下一个 token")


# ============================================================
section("TODO-9：Top-K 过滤 apply_top_k")
# ============================================================
# 把每行 logits 中「不在前 k 大」的位置填成 -inf，
# softmax 后这些位置概率为 0 → 永远不会被采样到。
#
# 提示：
#   1. torch.topk(logits, k) 拿到每行第 k 大的值（返回值的最后一列）
#   2. masked_fill / 布尔索引，把小于该值的位置设为 float("-inf")


def apply_top_k(logits, k):
    # TODO-9: 返回过滤后的 logits，形状不变 (B, vocab_size)
    return None


_tk_logits = torch.tensor([[1.0, 5.0, 3.0, 2.0],
                           [-1.0, 0.0, 2.0, 1.0]])
_tk_out = apply_top_k(_tk_logits.clone(), 2)
require_shape("TODO-9 输出形状", _tk_out, (2, 4))
_finite = torch.isfinite(_tk_out)
require_close("TODO-9 每行保留 2 个", _finite.sum(dim=-1).float(), torch.tensor([2.0, 2.0]))
require_true("TODO-9 保留的是最大的两个",
             bool(_finite[0, 1] and _finite[0, 2] and _finite[1, 2] and _finite[1, 3]),
             "第 0 行应保留 5.0/3.0，第 1 行应保留 2.0/1.0")
require_close("TODO-9 保留位置数值不变", _tk_out[_finite], _tk_logits[_finite])
print("apply_top_k OK：")
print(_tk_out)


# ============================================================
section("TODO-10：自回归生成 GPT.generate")
# ============================================================
# 回到上面 GPT 类里把 generate 补全（步骤注释已写在方法里）。
# 校验思路：top_k=1 时只剩一个候选 → 采样退化为贪心 → 必须和逐位 argmax 一致。

_model.eval()
_prompt = torch.randint(0, _gpt_cfg.vocab_size, (1, 4))
_gen = _model.generate(_prompt.clone(), max_new_tokens=6, temperature=1.0, top_k=1)
require_not_none("TODO-10 generate", _gen)
require_shape("TODO-10 输出长度 = 输入 + 新 token", _gen, (1, 4 + 6))
require_close("TODO-10 前缀保持不变", _gen[:, :4].float(), _prompt.float())

with torch.no_grad():
    _ref_gen = _prompt.clone()
    for _ in range(6):
        _l, _ = _model(_ref_gen[:, -_gpt_cfg.context_len:])
        _next = _l[:, -1, :].argmax(dim=-1, keepdim=True)
        _ref_gen = torch.cat([_ref_gen, _next], dim=1)
require_close("TODO-10 top_k=1 等价贪心解码", _gen.float(), _ref_gen.float())
print(f"generate OK：top_k=1 与贪心解码完全一致 → 采样管线正确")


# ============================================================
section("TODO-11（进阶，可跳过）：Top-P / nucleus 过滤 apply_top_p")
# ============================================================
# ChatGPT 实际用的采样策略。规则：
#   1. 把 logits softmax 成概率，按概率从高到低排序
#   2. 取累计概率刚好达到 p 的最小集合（包含越过 p 的那一个）
#   3. 集合外的位置填 -inf
#
# 提示：torch.sort(descending=True) → softmax → cumsum →
#       找出「前一位累计已 ≥ p」的位置标记删除 → scatter 还原回原顺序


def apply_top_p(logits, p):
    # TODO-11: 返回过滤后的 logits，形状不变 (B, vocab_size)（可跳过）
    return None


_tp_probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
_tp_logits = torch.log(_tp_probs)
_tp_out = apply_top_p(_tp_logits.clone(), 0.75)
if _tp_out is None:
    print("TODO-11 未实现，跳过（这是可选进阶题）")
else:
    _finite_p = torch.isfinite(_tp_out)
    # p=0.75：0.5 不够，加上 0.3 后 0.8 ≥ 0.75 → 保留前两个
    require_true("TODO-11 p=0.75 保留 {0.5, 0.3}",
                 bool(_finite_p[0, 0] and _finite_p[0, 1]
                      and not _finite_p[0, 2] and not _finite_p[0, 3]),
                 f"isfinite={_finite_p.tolist()}，期望 [True, True, False, False]")
    _tp_out2 = apply_top_p(_tp_logits.clone(), 0.9)
    _finite_p2 = torch.isfinite(_tp_out2)
    # p=0.9：0.5+0.3=0.8 < 0.9，要再加 0.15 → 保留前三个
    require_true("TODO-11 p=0.9 保留前三个",
                 bool(_finite_p2[0, :3].all() and not _finite_p2[0, 3]),
                 f"isfinite={_finite_p2.tolist()}，期望 [True, True, True, False]")
    print("apply_top_p OK：nucleus 采样过滤正确")


# ============================================================
section("TODO-12（进阶，可跳过）：用 tiktoken 体验 BPE 分词")
# ============================================================
# 我们的迷你 GPT 用字符级分词（词表只有 30 来个字符），真实 GPT 用 BPE 子词分词
# （GPT-2 词表 50257）。本题用 GPT-2 的 BPE 编码同一句话，直观对比两种粒度。
#
# 提示：
#   import tiktoken
#   enc = tiktoken.get_encoding("gpt2")   # 首次运行需联网下载 BPE 词表
#   return enc.encode(text)
# （requirements.txt 里已包含 tiktoken；没装或没网就先跳过，不影响后面）


def encode_with_gpt2_bpe(text):
    # TODO-12: 返回 GPT-2 BPE 编码后的 token id 列表（可跳过）
    return None


_sample = "To be or not to be that is the question"
try:
    _bpe_ids = encode_with_gpt2_bpe(_sample)
except Exception as _e:  # tiktoken 未安装 / 下载词表失败等
    _bpe_ids = None
    print(f"TODO-12 跳过（tiktoken 不可用：{_e}）")

if _bpe_ids is None:
    print("TODO-12 未实现或不可用，跳过（这是可选进阶题）")
else:
    import tiktoken

    _enc = tiktoken.get_encoding("gpt2")
    require_true("TODO-12 解码还原", _enc.decode(_bpe_ids) == _sample,
                 "enc.encode 后 enc.decode 应能还原原文")
    require_true("TODO-12 BPE 序列更短", len(_bpe_ids) < len(_sample),
                 f"BPE 子词数（{len(_bpe_ids)}）应明显少于字符数（{len(_sample)}）")
    print(f"同一句话：字符级 {len(_sample)} 个 token  vs  GPT-2 BPE {len(_bpe_ids)} 个 token")
    print("BPE 切分结果：", [_enc.decode([t]) for t in _bpe_ids])
    print("→ 子词分词让序列短 ~4 倍：同样 context_len 能装下更多内容，这就是真实 GPT 不用字符级的原因")


# ============================================================
section("终极验证：用你写的 GPT 真的训练 + 生成")
# ============================================================
# 走到这里说明 TODO-1~10 全部通过。下面的代码（不需要你改）会用
# 你自己写的模型训练 300 步，再生成文本。
#
# 想换数据玩（对应主课练习 3 / 6，都不影响校验）：
#   * 练习 6：下载 tiny-shakespeare 存到本课同目录，会自动优先使用——
#     Invoke-WebRequest https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -OutFile phase2-transformer/tiny_shakespeare.txt
#   * 练习 3：直接把下面的 training_text 换成中文、代码或歌词，
#     观察字符级 GPT 学不同「语言」的速度差异

_ts_path = Path(__file__).with_name("tiny_shakespeare.txt")
if _ts_path.exists():
    # 真实数据集约 1.1M 字符，CPU 上训练取前 5 万字符即可看到效果
    training_text = _ts_path.read_text(encoding="utf-8")[:50_000]
    print("检测到 tiny_shakespeare.txt → 用真实数据集训练（截取前 50,000 字符）")
else:
    training_text = """To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them To die to sleep
No more and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to Tis a consummation
Devoutly to be wished To die to sleep
To sleep perchance to dream""" * 3

chars = sorted(set(training_text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

train_config = GPTConfig(
    vocab_size=len(chars), context_len=32, n_layer=2, n_head=2, n_embd=32
)
torch.manual_seed(42)
model = GPT(train_config)
data = torch.tensor([char_to_idx[c] for c in training_text], dtype=torch.long)

print(f"训练文本: {len(training_text)} 字符，词汇表: {len(chars)}，"
      f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
model.train()
first_loss = None
for step in range(300):
    xb, yb = get_batch(data, train_config.context_len, batch_size=16)
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if first_loss is None:
        first_loss = loss.item()
    if step % 100 == 0 or step == 299:
        print(f"  Step {step:>4d}/300  Loss: {loss.item():.4f}")

require_true(
    "训练后 loss 显著下降",
    loss.item() < first_loss * 0.75,
    f"初始 {first_loss:.4f} → 最终 {loss.item():.4f}，下降不足，检查 forward/get_batch",
)

model.eval()
prompt = "To be or"
prompt_ids = torch.tensor([[char_to_idx[c] for c in prompt]], dtype=torch.long)
out = model.generate(prompt_ids, max_new_tokens=120, temperature=0.8, top_k=8)
print(f"\n提示词: '{prompt}'，生成结果（temperature=0.8, top_k=8）:")
print("".join(idx_to_char[i] for i in out[0].tolist()))


# ============================================================
section("全部 TODO 校验通过 ✓")
# ============================================================
print("""
你已经手写完成：
  1. 因果掩码（torch.triu）
  2. QKV 合并投影的拆分与分头
  3. 注意力核心计算（scores → mask → softmax → 加权 → 合并多头）
  4. FeedForward / Pre-Norm TransformerBlock 的 forward
  5. weight tying（embedding 和 lm_head 共享同一个 Parameter）
  6. GPT.forward（embedding 相加 → Blocks → ln_f → lm_head → cross_entropy）
  7. get_batch（y = x 右移一位 = next token prediction 的本质）
  8. Top-K 过滤 + 自回归 generate（temperature / top_k）
  9.（可选）Top-P / nucleus 过滤
 10.（可选）tiktoken BPE 分词，对比字符级粒度

复盘三问（对照第 5 课 README 的步骤⑥）：
  * 输入是什么？—— (B, T) 的 token 索引
  * 核心计算是什么？—— embedding 相加 → N × (attention + FFN, 都带 Pre-Norm 残差) → lm_head
  * 输出是什么？—— (B, T, vocab) 的 logits；训练时附带 cross_entropy loss

延伸思考（对应主课练习区，做完可在那边划掉）：
  * 练习 1：把训练步数改成 1000~2000，生成质量有什么变化？loss 还能降多少？
  * 练习 3：把 training_text 换成中文/代码/歌词，观察字符级 GPT 的学习差异
  * 练习 6：下载 tiny_shakespeare.txt（方法见「终极验证」注释），用真实数据训练
  * 把 get_batch 改成 90/10 切分 train/val，观察 val loss 何时开始不降（过拟合）
  * 在 generate 里接上你的 apply_top_p，对比 top_k=8 和 top_p=0.9 的生成差异
""")
