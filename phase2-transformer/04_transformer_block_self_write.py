"""
======================================================
phase2 / 第 4 课（自写版）：完整的 Transformer Block
======================================================

用法：
1. 运行：python3 04_transformer_block_self_write.py
2. 按 TODO-1 到 TODO-9 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错

目标：
- 手写 GELU 激活函数
- 手写 Feed-Forward Network（两层 MLP + GELU）
- 复习 MultiHeadAttention（直接复用上一课的实现）
- 手写 Pre-Norm 风格的 TransformerBlock
- 手写 Post-Norm 风格的 TransformerBlock 并对比差异
- 堆叠多层，验证残差 + LayerNorm 让深度可扩展
- 手写 inverted dropout（训练 / 推理两种模式）

关键观察：
- FFN 中间维度 d_ff = 4 × d_model（约定俗成）
- FFN 占了 Block 中约 2/3 的参数（"模型记忆知识的地方"）
- 输入输出 shape 完全相同 → 可以堆叠任意多层
- Pre-Norm 在深层模型中训练更稳定（GPT-2/3、LLaMA 都用 Pre-Norm）
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


def _ref_layer_norm(x, gamma=None, beta=None, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma
    if beta is not None:
        out = out + beta
    return out


def _ref_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _ref_ffn(x, W1, b1, W2, b2):
    hidden = _ref_gelu(x @ W1 + b1)
    return hidden @ W2 + b2


def _ref_mha(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):
    seq_len_, d_model_ = x.shape
    d_head_ = d_model_ // num_heads
    Q = (x @ W_Q).reshape(seq_len_, num_heads, d_head_).transpose(1, 0, 2)
    K = (x @ W_K).reshape(seq_len_, num_heads, d_head_).transpose(1, 0, 2)
    V = (x @ W_V).reshape(seq_len_, num_heads, d_head_).transpose(1, 0, 2)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head_)
    if mask is not None:
        scores = scores - mask * 1e9
    weights = _ref_softmax(scores, axis=-1)
    head_outputs = weights @ V
    concat = head_outputs.transpose(1, 0, 2).reshape(seq_len_, d_model_)
    return concat @ W_O


def _ref_pre_norm_block(x, params, num_heads, mask=None):
    x_norm = _ref_layer_norm(x)
    attn_out = _ref_mha(
        x_norm, params["W_Q"], params["W_K"], params["W_V"], params["W_O"], num_heads, mask
    )
    x = x + attn_out
    x_norm = _ref_layer_norm(x)
    ffn_out = _ref_ffn(x_norm, params["W1"], params["b1"], params["W2"], params["b2"])
    return x + ffn_out


def _ref_post_norm_block(x, params, num_heads, mask=None):
    attn_out = _ref_mha(
        x, params["W_Q"], params["W_K"], params["W_V"], params["W_O"], num_heads, mask
    )
    x = _ref_layer_norm(x + attn_out)
    ffn_out = _ref_ffn(x, params["W1"], params["b1"], params["W2"], params["b2"])
    return _ref_layer_norm(x + ffn_out)


# ---------- 准备数据（不需要你改） ----------
np.random.seed(42)

seq_len = 6
d_model = 16
num_heads = 4
d_head = d_model // num_heads
d_ff = d_model * 4

X = np.random.randn(seq_len, d_model) * 0.5

# 注意力子层的参数
W_Q = np.random.randn(d_model, d_model) * 0.1
W_K = np.random.randn(d_model, d_model) * 0.1
W_V = np.random.randn(d_model, d_model) * 0.1
W_O = np.random.randn(d_model, d_model) * 0.1

# FFN 子层的参数
W1 = np.random.randn(d_model, d_ff) * 0.1
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.1
b2 = np.zeros(d_model)

BLOCK_PARAMS = {
    "W_Q": W_Q, "W_K": W_K, "W_V": W_V, "W_O": W_O,
    "W1": W1, "b1": b1, "W2": W2, "b2": b2,
}

causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)

print(f"输入 X.shape = {X.shape}")
print(f"d_model={d_model}, num_heads={num_heads}, d_head={d_head}, d_ff={d_ff}")


# ============================================================
section("TODO-1：实现数值稳定版 softmax（复习）")
# ============================================================
# 提示：
#   1. 先减去每行最大值，避免 exp(大数) 溢出
#   2. 在指定的 axis 上归一化时，要 keepdims=True


def softmax(x, axis=-1):
    # TODO-1: 实现 softmax
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


_softmax_test = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
_softmax_out = softmax(_softmax_test, axis=-1)
require_shape("TODO-1 softmax", _softmax_out, (2, 3))
require_close("TODO-1 每行和=1", _softmax_out.sum(axis=-1), np.ones(2))
require_close("TODO-1 数值稳定", _softmax_out, _ref_softmax(_softmax_test, axis=-1))
print("softmax OK，大数行 =", _softmax_out[1])


# ============================================================
section("TODO-2：实现 LayerNorm（复习）")
# ============================================================
# LayerNorm 对每个 token 的最后一维特征做归一化：
#   mean = mean(x, axis=-1)
#   var  = var(x, axis=-1)
#   x_norm = (x - mean) / sqrt(var + eps)
#   y = gamma * x_norm + beta（gamma/beta 可学习，shape=(d_model,)）


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    # TODO-2: 实现 LayerNorm
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma
    if beta is not None:
        out = out + beta
    return out


ln_out = layer_norm(X)
require_shape("TODO-2 ln_out", ln_out, (seq_len, d_model))
require_close("TODO-2 与参考一致", ln_out, _ref_layer_norm(X))
require_close("TODO-2 每个 token 均值约 0", ln_out.mean(axis=-1), np.zeros(seq_len), atol=1e-6)
require_close("TODO-2 每个 token 标准差约 1", ln_out.std(axis=-1), np.ones(seq_len), atol=1e-4)
print("LayerNorm OK，每个 token 的 mean =", np.round(ln_out.mean(axis=-1), 6))


# ============================================================
section("TODO-3：实现 GELU 激活函数")
# ============================================================
# GELU 是 GPT 系列使用的激活函数，公式（tanh 近似）：
#   GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
#
# 与 ReLU 的区别：
#   - ReLU(x) = max(0, x)，在 x<0 时严格为 0、不可导
#   - GELU 在 x<0 时不是完全为 0，保留一点点信号 → 更平滑、训练更稳定


def gelu(x):
    # TODO-3: 实现 GELU（tanh 近似）
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


_gelu_test = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
_gelu_out = gelu(_gelu_test)
require_shape("TODO-3 gelu", _gelu_out, _gelu_test.shape)
require_close("TODO-3 数值", _gelu_out, _ref_gelu(_gelu_test))
require_close("TODO-3 gelu(0)=0", _gelu_out[3], 0.0, atol=1e-6)
require_true("TODO-3 负数区不全为 0", abs(_gelu_out[0]) > 1e-4, "ReLU 在 -2 处为 0，GELU 不应该")
require_true("TODO-3 大正数 ≈ x", abs(_gelu_out[-1] - 2.0) < 0.06, "在大正数处 GELU(x) ≈ x，gelu(2.0) ≈ 1.9546")
print("GELU vs ReLU：")
for v, g in zip(_gelu_test, _gelu_out):
    print(f"  x={v:+.1f}  ReLU={max(0, v):.4f}  GELU={g:.4f}")


# ============================================================
section("TODO-4：实现 Feed-Forward Network（两层 MLP）")
# ============================================================
# 公式：
#   FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
#
# 形状：
#   x:  (seq_len, d_model)
#   W1: (d_model, d_ff)    b1: (d_ff,)
#   W2: (d_ff, d_model)    b2: (d_model,)
#   返回: (seq_len, d_model)
#
# 关键观察：FFN 是逐 token 独立计算的（不混合 token 之间的信息）。


def feed_forward(x, W1, b1, W2, b2):
    # TODO-4: 实现两层 MLP + GELU
    hidden = gelu(x @ W1 + b1)
    return hidden @ W2 + b2


ffn_out = feed_forward(X, W1, b1, W2, b2)
require_shape("TODO-4 ffn_out", ffn_out, (seq_len, d_model))
require_close("TODO-4 数值", ffn_out, _ref_ffn(X, W1, b1, W2, b2))

# 验证 FFN 是逐 token 独立的：对某个 token 单独跑，结果应与整批跑出的一致
single_token_out = feed_forward(X[2:3], W1, b1, W2, b2)
require_close(
    "TODO-4 FFN 逐 token 独立",
    single_token_out,
    ffn_out[2:3],
    atol=1e-8,
)

# 参数量统计
ffn_param_count = d_model * d_ff + d_ff + d_ff * d_model + d_model
print(f"FFN 参数量: {ffn_param_count:,}")
print(f"  W1: {d_model}×{d_ff}={d_model * d_ff}, b1: {d_ff}")
print(f"  W2: {d_ff}×{d_model}={d_ff * d_model}, b2: {d_model}")


# ============================================================
section("TODO-5：实现多头注意力 multi_head_attention（复用第 3 课）")
# ============================================================
# 流程（简化为一次性 reshape，不再用 for-head 循环）：
#   1. Q = X @ W_Q，K = X @ W_K，V = X @ W_V
#   2. 把最后一维拆成 (num_heads, d_head)，并把 head 放到最前面
#      shape: (seq_len, d_model) -> (num_heads, seq_len, d_head)
#   3. scores = Q @ K^T / sqrt(d_head)，加 mask，softmax
#   4. weights @ V → 拼回 (seq_len, d_model)
#   5. 乘 W_O 混合各 head 的信息


def multi_head_attention(x, W_Q, W_K, W_V, W_O, num_heads, mask=None):
    # TODO-5: 实现多头注意力
    seq_len_, d_model_ = x.shape
    d_head_ = d_model_ // num_heads

    Q = (x @ W_Q).reshape(seq_len_, num_heads, d_head_).transpose(1, 0, 2)
    K = (x @ W_K).reshape(seq_len_, num_heads, d_head_).transpose(1, 0, 2)
    V = (x @ W_V).reshape(seq_len_, num_heads, d_head_).transpose(1, 0, 2)

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head_)
    if mask is not None:
        scores = scores - mask * 1e9
    weights = softmax(scores, axis=-1)
    head_outputs = weights @ V  # (num_heads, seq_len, d_head)

    concat = head_outputs.transpose(1, 0, 2).reshape(seq_len_, d_model_)
    return concat @ W_O


mha_out = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads)
require_shape("TODO-5 mha_out", mha_out, (seq_len, d_model))
require_close("TODO-5 无 mask 数值", mha_out, _ref_mha(X, W_Q, W_K, W_V, W_O, num_heads))

mha_causal = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads, mask=causal_mask)
require_close(
    "TODO-5 causal mask 数值",
    mha_causal,
    _ref_mha(X, W_Q, W_K, W_V, W_O, num_heads, mask=causal_mask),
)
print("multi_head_attention OK，输出 shape =", mha_out.shape)


# ============================================================
section("TODO-6：实现 Pre-Norm 风格的 Transformer Block")
# ============================================================
# Pre-Norm（GPT-2/3、LLaMA 等现代模型用的就是这种）：
#
#   x ─┬─► LayerNorm ─► MHA ─┐
#      │                     ▼
#      └────► + ──────────► (1)
#                 │
#                 ├─► LayerNorm ─► FFN ─┐
#                 │                     ▼
#                 └────► + ──────────► output
#
# 伪代码：
#   x = x + MHA(LayerNorm(x))
#   x = x + FFN(LayerNorm(x))


def pre_norm_block(x, params, num_heads, mask=None):
    # TODO-6: 实现 Pre-Norm Transformer Block
    x_norm = layer_norm(x)
    attn_out = multi_head_attention(
        x_norm, params["W_Q"], params["W_K"], params["W_V"], params["W_O"], num_heads, mask
    )
    x = x + attn_out

    x_norm = layer_norm(x)
    ffn_out = feed_forward(x_norm, params["W1"], params["b1"], params["W2"], params["b2"])
    x = x + ffn_out
    return x


pre_out = pre_norm_block(X, BLOCK_PARAMS, num_heads, mask=causal_mask)
require_shape("TODO-6 pre_out", pre_out, (seq_len, d_model))
require_close(
    "TODO-6 Pre-Norm 数值",
    pre_out,
    _ref_pre_norm_block(X, BLOCK_PARAMS, num_heads, mask=causal_mask),
)
print("Pre-Norm Block OK，输出 shape =", pre_out.shape)
print("→ 输入和输出 shape 完全一样 → 可以堆叠任意层")


# ============================================================
section("TODO-7：实现 Post-Norm 风格的 Transformer Block")
# ============================================================
# Post-Norm（原始 Transformer 论文用的版本）：
#   x = LayerNorm(x + MHA(x))
#   x = LayerNorm(x + FFN(x))
#
# 与 Pre-Norm 的差别：LayerNorm 放在残差相加 *之后* 而不是子层 *之前*。
# 经验上 Post-Norm 在深层模型里梯度不稳定，所以现代模型几乎都用 Pre-Norm。


def post_norm_block(x, params, num_heads, mask=None):
    # TODO-7: 实现 Post-Norm Transformer Block
    attn_out = multi_head_attention(
        x, params["W_Q"], params["W_K"], params["W_V"], params["W_O"], num_heads, mask
    )
    x = layer_norm(x + attn_out)

    ffn_out = feed_forward(x, params["W1"], params["b1"], params["W2"], params["b2"])
    x = layer_norm(x + ffn_out)
    return x


post_out = post_norm_block(X, BLOCK_PARAMS, num_heads, mask=causal_mask)
require_shape("TODO-7 post_out", post_out, (seq_len, d_model))
require_close(
    "TODO-7 Post-Norm 数值",
    post_out,
    _ref_post_norm_block(X, BLOCK_PARAMS, num_heads, mask=causal_mask),
)
require_true(
    "TODO-7 Pre/Post 输出不同",
    not np.allclose(pre_out, post_out),
    "LayerNorm 位置不同，输出应该不同",
)
require_close(
    "TODO-7 Post-Norm 每个 token 均值约 0",
    post_out.mean(axis=-1),
    np.zeros(seq_len),
    atol=1e-6,
)
require_close(
    "TODO-7 Post-Norm 每个 token 标准差约 1",
    post_out.std(axis=-1),
    np.ones(seq_len),
    atol=1e-4,
)
print("Post-Norm Block OK，输出每个 token 都被 LayerNorm 规整过")


# ============================================================
section("TODO-8：堆叠多个 Pre-Norm Block（每层独立参数），观察数值稳定性")
# ============================================================
# 真实的 GPT/LLaMA 是把 Pre-Norm Block 堆叠几十层，并且 **每层有自己独立的参数**。
# 所以本题：
#   1. 为 n_layers 层各自生成一份独立的随机参数
#   2. 串联跑前向，逐层记录 mean / std / |max|
#   3. 校验数值始终 finite 且不爆炸
#
# 你应该看到：即使堆叠 6 层，std 和 |max| 都不会随层数指数增长
# （注意：Pre-Norm 的最终输出 *不会* 被 LayerNorm 规整，所以 std 不是 1，
#   但只要逐层增量是有界的，深度就是可扩展的 —— 这正是残差+LN 想要的）。

n_layers = 6


def make_block_params(d_model, num_heads, d_ff, rng):
    # 给一层 block 生成一份独立的随机参数（不需要你改）
    return {
        "W_Q": rng.randn(d_model, d_model) * 0.1,
        "W_K": rng.randn(d_model, d_model) * 0.1,
        "W_V": rng.randn(d_model, d_model) * 0.1,
        "W_O": rng.randn(d_model, d_model) * 0.1,
        "W1":  rng.randn(d_model, d_ff) * 0.1,
        "b1":  np.zeros(d_ff),
        "W2":  rng.randn(d_ff, d_model) * 0.1,
        "b2":  np.zeros(d_model),
    }


_layer_rng = np.random.RandomState(123)
layer_params = [make_block_params(d_model, num_heads, d_ff, _layer_rng) for _ in range(n_layers)]


def stack_blocks(x, layer_params_list, num_heads, mask=None):
    # TODO-8: 把 pre_norm_block 串联 len(layer_params_list) 次，每层用各自的参数
    h = x
    for params in layer_params_list:
        h = pre_norm_block(h, params, num_heads, mask=mask)
    return h


stacked_out = stack_blocks(X, layer_params, num_heads, mask=causal_mask)
require_shape("TODO-8 stacked_out", stacked_out, (seq_len, d_model))
require_true(
    "TODO-8 没有 NaN/Inf",
    np.all(np.isfinite(stacked_out)),
    "堆叠多层后出现 NaN/Inf 说明残差或 LayerNorm 有问题",
)
require_true(
    "TODO-8 数值没有爆炸",
    np.abs(stacked_out).max() < 50.0,
    f"|max|={np.abs(stacked_out).max():.2f} 太大，怀疑残差没接住",
)

# 与参考实现逐层对齐，确保 stack 顺序正确（而不是反着用 params）
ref_stacked = X.copy()
for params in layer_params:
    ref_stacked = _ref_pre_norm_block(ref_stacked, params, num_heads, mask=causal_mask)
require_close("TODO-8 stacked 数值", stacked_out, ref_stacked)

# 逐层记录数值演化
print(f"堆叠 {n_layers} 层 Pre-Norm Block（每层独立参数）：")
print(f"  {'层':>4s}  {'mean':>10s}  {'std':>10s}  {'|max|':>10s}")
h = X
print(f"  {'in':>4s}  {h.mean():>+10.4f}  {h.std():>10.4f}  {np.abs(h).max():>10.4f}")
for i, params in enumerate(layer_params):
    h = pre_norm_block(h, params, num_heads, mask=causal_mask)
    print(f"  {i:>4d}  {h.mean():>+10.4f}  {h.std():>10.4f}  {np.abs(h).max():>10.4f}")
print("→ 逐层 std、|max| 都没有指数增长 → 残差 + LayerNorm 起作用了")
print("  (要把最终输出拉回 std≈1，需要在堆栈末尾再补一次 LayerNorm，这就是 GPT 的 final LN)")


# ============================================================
section("TODO-9：实现 inverted dropout")
# ============================================================
# Dropout（inverted 版本，PyTorch / TF 默认实现方式）：
#
#   训练时：
#     mask = (rand(*x.shape) > rate)          # True 表示保留
#     y = x * mask / (1 - rate)               # 缩放，保持期望不变
#
#   推理时（training=False）：
#     y = x                                    # 不做任何事
#
# 为什么训练时要除以 (1 - rate)？
#   - E[mask] = 1 - rate
#   - 除以 (1 - rate) 后 E[y] = E[x]，训练 / 推理期望一致
#   - 推理时就不需要再缩放


def dropout(x, rate=0.1, training=True, rng=None):
    # TODO-9: 实现 inverted dropout
    if (not training) or rate == 0:
        return x
    if rng is None:
        rng = np.random
    keep = np.asarray(rng.rand(*x.shape) > rate, dtype=x.dtype)
    scale = np.asarray(1.0 / (1.0 - rate), dtype=x.dtype)
    return x * keep * scale


# 推理模式下应该完全不变
infer_in = np.random.randn(4, 8)
infer_out = dropout(infer_in, rate=0.5, training=False)
require_close("TODO-9 推理模式不动", infer_out, infer_in)

# 训练模式：用大样本验证期望保持不变
big = np.ones((2000, 200))
big_rng = np.random.RandomState(0)
big_out = dropout(big, rate=0.3, training=True, rng=big_rng)
require_true(
    "TODO-9 训练模式期望保持",
    abs(big_out.mean() - 1.0) < 0.02,
    f"E[y] 应该 ≈ E[x]=1，实际 {big_out.mean():.4f}",
)
require_true(
    "TODO-9 训练模式确实丢了值",
    (big_out == 0).mean() > 0.25,
    "应该有约 rate 比例的位置变 0",
)

# 非零位置应该被放大 1/(1-rate) 倍
non_zero_vals = big_out[big_out != 0]
require_true(
    "TODO-9 非零位置缩放正确",
    abs(non_zero_vals.mean() - 1.0 / (1.0 - 0.3)) < 0.02,
    f"非零值应该 ≈ 1/(1-0.3) ≈ 1.4286，实际 {non_zero_vals.mean():.4f}",
)

# rate=0 应该什么都不做
require_close("TODO-9 rate=0 不动", dropout(infer_in, rate=0.0, training=True), infer_in)

# dtype 应该保持不变（实战中模型常用 float32，不能被悄悄升回 float64）
f32_in = np.ones((4, 8), dtype=np.float32)
f32_out = dropout(f32_in, rate=0.3, training=True, rng=np.random.RandomState(1))
require_true(
    "TODO-9 dtype 保持",
    f32_out.dtype == np.float32,
    f"输入 float32，输出却变成 {f32_out.dtype}；小心 Python 浮点会把数组升回 float64",
)

print("dropout(rate=0.3) 训练模式：")
print(f"  零值比例    = {(big_out == 0).mean():.4f}（约等于 rate=0.3）")
print(f"  非零值均值  = {non_zero_vals.mean():.4f}（约等于 1/(1-0.3)=1.4286）")
print(f"  整体期望    = {big_out.mean():.4f}（约等于 1.0，与原输入一致）")


# ============================================================
section("Part X：参数量统计与真实模型对比（无需补全，直接观察）")
# ============================================================

attn_params = 4 * d_model * d_model  # W_Q, W_K, W_V, W_O
ffn_params = 2 * d_model * d_ff + d_ff + d_model  # W1, W2, b1, b2
block_params = attn_params + ffn_params

print(f"单个 Block 参数量：{block_params:,}")
print(f"  注意力层：{attn_params:,}  ({attn_params / block_params * 100:.0f}%)")
print(f"  FFN  层：{ffn_params:,}  ({ffn_params / block_params * 100:.0f}%)")
print("→ FFN 占了约 2/3 的参数，是模型'存储知识'的主要地方\n")

real_models = {
    "我们的玩具模型": (n_layers, d_model, num_heads, d_ff),
    "GPT-2 Small":   (12, 768, 12, 3072),
    "GPT-2 Large":   (36, 1280, 20, 5120),
    "GPT-3 175B":    (96, 12288, 96, 49152),
    "LLaMA-2 70B":   (80, 8192, 64, 28672),
}

print(f"{'模型':<18s} {'层数':>6s} {'d_model':>8s} {'heads':>6s} {'d_ff':>8s} {'~参数量':>10s}")
print("-" * 64)
for name, (n, dm, h, ff) in real_models.items():
    p = n * (4 * dm * dm + 2 * dm * ff + ff + dm)
    if p > 1e9:
        s = f"{p / 1e9:.1f}B"
    elif p > 1e6:
        s = f"{p / 1e6:.1f}M"
    else:
        s = f"{p / 1e3:.1f}K"
    print(f"{name:<18s} {n:>6d} {dm:>8d} {h:>6d} {ff:>8d} {s:>10s}")


# ============================================================
section("全部 TODO 校验通过 ✓")
# ============================================================
print("""
你已经手写完成：
  1. softmax / LayerNorm（复习）
  2. GELU 激活函数
  3. Feed-Forward Network（两层 MLP + GELU）
  4. multi_head_attention（reshape 版本，一次算所有 head）
  5. Pre-Norm Transformer Block
  6. Post-Norm Transformer Block，并对比两者差异
  7. 堆叠 N 层 Block，验证数值稳定性
  8. inverted dropout（训练 / 推理两种模式 + 期望保持）

下一课（05）：把 Token Embedding + Position Encoding + N 层 Block + LM Head
              拼起来，得到一个能预测下一个 token 的迷你 GPT。

延伸思考：
  * 把 d_ff 改成 2×d_model 或 8×d_model，看输出有什么差异；
    现代 SwiGLU FFN（LLaMA 用）的等效隐藏维度约为 2/3 × 4 × d_model。
  * 注释掉 Pre-Norm 里的两次 layer_norm，再堆 20 层，看数值是否爆炸。
  * 把 dropout 加到 attention weights 和 FFN 输出上，看训练时数值的抖动。
""")
