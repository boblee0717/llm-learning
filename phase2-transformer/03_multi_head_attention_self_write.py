"""
======================================================
phase2 / 第 3 课（自写版）：Multi-Head Attention
======================================================

用法：
1. 运行：python3 03_multi_head_attention_self_write.py
2. 按 TODO-1 到 TODO-9 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错

目标：
- 复习数值稳定版 softmax
- 手写单头 scaled dot-product attention
- 手写多头切分 split_heads 与合并 merge_heads
- 手写完整 multi_head_attention
- 验证 causal mask 对所有 head 生效
- 手写残差连接、LayerNorm、Pre-Norm / Post-Norm

核心公式：
    head_i = Attention(X W_i^Q, X W_i^K, X W_i^V)
    MultiHead(X) = Concat(head_1, ..., head_h) W^O
"""

import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


class ValidationError(Exception):
    pass


def require_not_none(name, value):
    if value is None:
        raise ValidationError(f"{name} 未实现：结果是 None。")


def require_shape(name, actual, expected_shape):
    require_not_none(name, actual)
    if actual.shape != expected_shape:
        raise ValidationError(
            f"{name} 形状不对：actual={actual.shape}, expected={expected_shape}"
        )


def require_close(name, actual, expected, atol=1e-6):
    require_not_none(name, actual)
    if not np.allclose(actual, expected, atol=atol):
        raise ValidationError(
            f"{name} 数值不对\nactual=\n{actual}\nexpected=\n{expected}"
        )


def require_true(name, cond, hint=""):
    if not cond:
        raise ValidationError(f"{name} 条件不满足：{hint}")


# ---------- 参考实现（仅供校验使用） ----------
def _ref_softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _ref_single_head_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores - mask * 1e9
    weights = _ref_softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


def _ref_split_heads(x, num_heads):
    seq_len, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model 必须能被 num_heads 整除")
    d_head = d_model // num_heads
    return x.reshape(seq_len, num_heads, d_head).transpose(1, 0, 2)


def _ref_merge_heads(x):
    num_heads, seq_len, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(seq_len, num_heads * d_head)


def _ref_multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads, mask=None):
    Q = _ref_split_heads(X @ W_Q, num_heads)
    K = _ref_split_heads(X @ W_K, num_heads)
    V = _ref_split_heads(X @ W_V, num_heads)

    head_outputs = []
    head_weights = []
    for h in range(num_heads):
        head_output, head_weight = _ref_single_head_attention(Q[h], K[h], V[h], mask)
        head_outputs.append(head_output)
        head_weights.append(head_weight)

    concat = _ref_merge_heads(np.stack(head_outputs, axis=0))
    return concat @ W_O, head_weights


def _ref_layer_norm(x, gamma=None, beta=None, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma
    if beta is not None:
        out = out + beta
    return out


# ---------- 准备数据（不需要你改） ----------
np.random.seed(42)

seq_len = 6
d_model = 16
num_heads = 4
d_head = d_model // num_heads

X = np.random.randn(seq_len, d_model) * 0.5
W_Q = np.random.randn(d_model, d_model) * 0.1
W_K = np.random.randn(d_model, d_model) * 0.1
W_V = np.random.randn(d_model, d_model) * 0.1
W_O = np.random.randn(d_model, d_model) * 0.1

print(f"输入 X.shape = {X.shape}")
print(f"d_model={d_model}, num_heads={num_heads}, d_head={d_head}")


# ============================================================
section("TODO-1：实现数值稳定版 softmax")
# ============================================================
# 提示：
#   1. 先减去每行最大值，避免 exp(大数) 溢出
#   2. 对最后一维归一化时，要保留维度 keepdims=True


def softmax(x, axis=-1):
    # TODO-1: 实现 softmax
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


_softmax_test = np.array([[1.0, 2.0, 3.0],
                          [1000.0, 1001.0, 1002.0]])
_softmax_out = softmax(_softmax_test, axis=-1)
require_shape("TODO-1 softmax", _softmax_out, (2, 3))
require_close("TODO-1 每行和=1", _softmax_out.sum(axis=-1), np.ones(2))
require_close("TODO-1 数值稳定", _softmax_out, _ref_softmax(_softmax_test, axis=-1))
print("softmax 大数行 =", _softmax_out[1])


# ============================================================
section("TODO-2：实现单头 scaled dot-product attention")
# ============================================================
# 公式：
#   scores = Q @ K.T / sqrt(d_k)
#   weights = softmax(scores)
#   output = weights @ V
#
# mask 约定与第 2 课一致：
#   mask.shape = (seq_len, seq_len)
#   1 表示屏蔽，0 表示可见


def single_head_attention(Q, K, V, mask=None):
    # TODO-2: 实现单头注意力
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores - mask * 1e9
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


Q_one = np.random.randn(seq_len, d_head)
K_one = np.random.randn(seq_len, d_head)
V_one = np.random.randn(seq_len, d_head)
one_out, one_weights = single_head_attention(Q_one, K_one, V_one)
ref_one_out, ref_one_weights = _ref_single_head_attention(Q_one, K_one, V_one)

require_shape("TODO-2 output", one_out, (seq_len, d_head))
require_shape("TODO-2 weights", one_weights, (seq_len, seq_len))
require_close("TODO-2 weights 每行和=1", one_weights.sum(axis=-1), np.ones(seq_len))
require_close("TODO-2 output 数值", one_out, ref_one_out)
require_close("TODO-2 weights 数值", one_weights, ref_one_weights)
print("单头 attention OK：output.shape =", one_out.shape)


# ============================================================
section("TODO-3：实现 split_heads")
# ============================================================
# 目标：
#   (seq_len, d_model) -> (num_heads, seq_len, d_head)
#
# 步骤：
#   1. reshape:   (seq_len, d_model) -> (seq_len, num_heads, d_head)
#   2. transpose: (seq_len, num_heads, d_head) -> (num_heads, seq_len, d_head)


def split_heads(x, num_heads):
    # TODO-3: 把最后一维 d_model 拆成 num_heads * d_head
    seq_len_, d_model_ = x.shape
    if d_model_ % num_heads != 0:
        raise ValueError("d_model 必须能被 num_heads 整除")
    d_head_ = d_model_ // num_heads
    return x.reshape(seq_len_, num_heads, d_head_).transpose(1, 0, 2)


Q_full = X @ W_Q
K_full = X @ W_K
V_full = X @ W_V

Q_heads = split_heads(Q_full, num_heads)
K_heads = split_heads(K_full, num_heads)
V_heads = split_heads(V_full, num_heads)

require_shape("TODO-3 Q_heads", Q_heads, (num_heads, seq_len, d_head))
require_shape("TODO-3 K_heads", K_heads, (num_heads, seq_len, d_head))
require_shape("TODO-3 V_heads", V_heads, (num_heads, seq_len, d_head))
require_close("TODO-3 head0 token0", Q_heads[0, 0], Q_full[0, :d_head])
require_close("TODO-3 head1 token0", Q_heads[1, 0], Q_full[0, d_head:2 * d_head])

_split_error = None
try:
    split_heads(X, 3)
except ValueError as err:
    _split_error = err
require_true("TODO-3 非整除要报错", _split_error is not None, "16 不能被 3 整除")
print("Q_heads.shape =", Q_heads.shape)


# ============================================================
section("TODO-4：实现 merge_heads")
# ============================================================
# 目标：
#   (num_heads, seq_len, d_head) -> (seq_len, d_model)
#
# 这是 split_heads 的逆操作：
#   1. transpose: (num_heads, seq_len, d_head) -> (seq_len, num_heads, d_head)
#   2. reshape:   (seq_len, num_heads, d_head) -> (seq_len, d_model)


def merge_heads(x):
    # TODO-4: 把多个 head 合并回最后一维
    num_heads_, seq_len_, d_head_ = x.shape
    return x.transpose(1, 0, 2).reshape(seq_len_, num_heads_ * d_head_)


Q_back = merge_heads(Q_heads)
require_shape("TODO-4 Q_back", Q_back, (seq_len, d_model))
require_close("TODO-4 split + merge 还原", Q_back, Q_full)
print("split_heads + merge_heads 可以还原原矩阵")


# ============================================================
section("TODO-5：实现完整 multi_head_attention")
# ============================================================
# 流程：
#   1. X @ W_Q / W_K / W_V 得到完整 Q/K/V
#   2. split_heads 切成多个头
#   3. 每个 head 独立做 single_head_attention
#   4. stack + merge_heads 拼回 (seq_len, d_model)
#   5. 乘 W_O，让不同 head 的信息再次混合


def multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads, mask=None):
    # TODO-5: 实现完整多头注意力
    Q = split_heads(X @ W_Q, num_heads)
    K = split_heads(X @ W_K, num_heads)
    V = split_heads(X @ W_V, num_heads)

    head_outputs = []
    head_weights = []
    for h in range(num_heads):
        head_output, head_weight = single_head_attention(Q[h], K[h], V[h], mask)
        head_outputs.append(head_output)
        head_weights.append(head_weight)

    concat = merge_heads(np.stack(head_outputs, axis=0))
    output = concat @ W_O
    return output, head_weights


mha_out, mha_weights = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads)
ref_mha_out, ref_mha_weights = _ref_multi_head_attention(
    X, W_Q, W_K, W_V, W_O, num_heads
)

require_shape("TODO-5 output", mha_out, (seq_len, d_model))
require_true("TODO-5 head 数量", len(mha_weights) == num_heads, "应返回每个 head 的权重")
for h in range(num_heads):
    require_shape(f"TODO-5 head {h} weights", mha_weights[h], (seq_len, seq_len))
    require_close(f"TODO-5 head {h} weights 数值", mha_weights[h], ref_mha_weights[h])
require_close("TODO-5 output 数值", mha_out, ref_mha_out)

print("各 head 中第 0 个 token 的注意力分布：")
for h, weights in enumerate(mha_weights):
    row = ", ".join(f"{w:.3f}" for w in weights[0])
    print(f"  head {h}: [{row}]")


# ============================================================
section("TODO-6：给多头注意力加 causal mask")
# ============================================================
# 目标：
#   第 i 个位置只能看 0..i，不能看未来 j>i。
#
# 提示：
#   np.triu(np.ones((seq_len, seq_len)), k=1)
#   主对角线以上为 1，表示未来位置要屏蔽。

causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # TODO-6
causal_out, causal_weights = multi_head_attention(
    X, W_Q, W_K, W_V, W_O, num_heads, mask=causal_mask
)
ref_causal_out, ref_causal_weights = _ref_multi_head_attention(
    X, W_Q, W_K, W_V, W_O, num_heads, mask=causal_mask
)

require_shape("TODO-6 causal_mask", causal_mask, (seq_len, seq_len))
require_close("TODO-6 causal_mask 数值", causal_mask, np.triu(np.ones((seq_len, seq_len)), k=1))
require_shape("TODO-6 causal_out", causal_out, (seq_len, d_model))
require_close("TODO-6 causal_out 数值", causal_out, ref_causal_out)

for h in range(num_heads):
    require_close(f"TODO-6 head {h} causal weights", causal_weights[h], ref_causal_weights[h])
    require_true(
        f"TODO-6 head {h} 上三角为 0",
        np.triu(causal_weights[h], k=1).sum() < 1e-6,
        "所有 head 都应该看不到未来 token",
    )

print("causal mask 已对所有 head 生效")
print("第 0 个 head 的第 0 行权重 =", causal_weights[0][0])


# ============================================================
section("TODO-7：实现残差连接")
# ============================================================
# Transformer 子层的残差形式：
#   output = x + sublayer(x)
#
# 注意：能做残差相加的前提是二者 shape 一样。


def residual_connection(x, sublayer_out):
    # TODO-7: 返回残差连接结果
    return x + sublayer_out


residual_out = residual_connection(X, mha_out)
require_shape("TODO-7 residual_out", residual_out, (seq_len, d_model))
require_close("TODO-7 residual 数值", residual_out, X + mha_out)
print("残差连接后 shape 不变：", residual_out.shape)


# ============================================================
section("TODO-8：实现 LayerNorm")
# ============================================================
# LayerNorm 对每个 token 的最后一维特征做归一化：
#   mean = mean(x, axis=-1)
#   var  = var(x, axis=-1)
#   x_norm = (x - mean) / sqrt(var + eps)
#
# gamma / beta 是可学习仿射参数，shape = (d_model,)


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    # TODO-8: 实现 LayerNorm
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma
    if beta is not None:
        out = out + beta
    return out


normed = layer_norm(residual_out)
require_shape("TODO-8 normed", normed, (seq_len, d_model))
require_close("TODO-8 与参考一致", normed, _ref_layer_norm(residual_out))
require_close("TODO-8 每个 token 均值约 0", normed.mean(axis=-1), np.zeros(seq_len), atol=1e-6)
require_close("TODO-8 每个 token 标准差约 1", normed.std(axis=-1), np.ones(seq_len), atol=1e-4)

gamma = np.linspace(0.8, 1.2, d_model)
beta = np.linspace(-0.1, 0.1, d_model)
normed_affine = layer_norm(residual_out, gamma=gamma, beta=beta)
require_close(
    "TODO-8 gamma/beta 仿射",
    normed_affine,
    _ref_layer_norm(residual_out, gamma=gamma, beta=beta),
)
print("LayerNorm 后每个 token 的 mean =", normed.mean(axis=-1))
print("LayerNorm 后每个 token 的 std  =", normed.std(axis=-1))


# ============================================================
section("TODO-9：实现 Post-Norm 与 Pre-Norm attention block")
# ============================================================
# Post-Norm（原始 Transformer）：
#   LayerNorm(x + Attention(x))
#
# Pre-Norm（GPT-2/3、LLaMA 等现代大模型常用）：
#   x + Attention(LayerNorm(x))


def post_norm_block(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):
    # TODO-9a: Attention -> Residual -> LayerNorm
    attn_out, _ = multi_head_attention(x, W_Q, W_K, W_V, W_O, num_heads, mask=mask)
    return layer_norm(x + attn_out)


def pre_norm_block(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):
    # TODO-9b: LayerNorm -> Attention -> Residual
    x_norm = layer_norm(x)
    attn_out, _ = multi_head_attention(x_norm, W_Q, W_K, W_V, W_O, num_heads, mask=mask)
    return x + attn_out


post_out = post_norm_block(X, W_Q, W_K, W_V, W_O, num_heads, mask=causal_mask)
pre_out = pre_norm_block(X, W_Q, W_K, W_V, W_O, num_heads, mask=causal_mask)

ref_post_out = _ref_layer_norm(
    X + _ref_multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads, mask=causal_mask)[0]
)
ref_pre_attn, _ = _ref_multi_head_attention(
    _ref_layer_norm(X), W_Q, W_K, W_V, W_O, num_heads, mask=causal_mask
)
ref_pre_out = X + ref_pre_attn

require_shape("TODO-9 post_out", post_out, (seq_len, d_model))
require_shape("TODO-9 pre_out", pre_out, (seq_len, d_model))
require_close("TODO-9 Post-Norm 数值", post_out, ref_post_out)
require_close("TODO-9 Pre-Norm 数值", pre_out, ref_pre_out)
require_true(
    "TODO-9 Pre/Post 输出不同",
    not np.allclose(post_out, pre_out),
    "二者 LayerNorm 放置位置不同，通常不应完全相同",
)

pre_with_final_ln = layer_norm(pre_out)
require_close("TODO-9 final LayerNorm 后均值约 0", pre_with_final_ln.mean(axis=-1), np.zeros(seq_len), atol=1e-6)
require_close("TODO-9 final LayerNorm 后标准差约 1", pre_with_final_ln.std(axis=-1), np.ones(seq_len), atol=1e-4)

print("Post-Norm 输出第 0 个 token 前 4 维 =", post_out[0, :4])
print("Pre-Norm 输出第 0 个 token 前 4 维  =", pre_out[0, :4])
print("Pre-Norm 后补 final LayerNorm，可把每个 token 重新拉回稳定尺度")


# ============================================================
section("全部 TODO 校验通过")
# ============================================================
print("""
你已经手写完成：
  1. 数值稳定版 softmax
  2. 单头 scaled dot-product attention
  3. split_heads / merge_heads
  4. 完整 multi_head_attention
  5. 所有 head 共享 causal mask
  6. 残差连接
  7. LayerNorm
  8. Post-Norm / Pre-Norm block

下一课：把 Attention 子层和 FFN 子层拼起来，就是完整 Transformer Block。
""")
