"""
======================================================
phase2 / 第 2 课（自写版）：Self-Attention
======================================================

用法：
1. 运行：python 02_self_attention_self_write.py
2. 按 TODO-1 到 TODO-8 顺序补全 `xxx = None` 处的实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错

目标：
- 手写 softmax（数值稳定版）
- 手写 Q / K / V 投影
- 手写 Scaled Dot-Product Attention 的每一步
- 理解为什么要除以 sqrt(d_k)
- 手写因果掩码（GPT 的关键）
- 把整个流程封装成函数

参考公式：
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
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


# ---------- 参考实现（仅供 validate 内部用，请不要偷看 :) ） ----------
def _ref_softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _ref_self_attention(X, W_Q, W_K, W_V, causal=False):
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if causal:
        seq_len = X.shape[0]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores = scores - mask * 1e9
    weights = _ref_softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


# ---------- 准备数据（不需要你改） ----------
np.random.seed(42)

sentence = ["小猫", "坐在", "垫子", "上", "它", "打呼噜"]
seq_len = len(sentence)
d_model = 8
d_k = 4   # Q / K 维度（必须一致）
d_v = 3   # V 维度（可独立）

embeddings = np.random.randn(seq_len, d_model) * 0.5
embeddings[4] = embeddings[0] + np.random.randn(d_model) * 0.1  # 让 "它" 像 "小猫"

W_Q = np.random.randn(d_model, d_k) * 0.3
W_K = np.random.randn(d_model, d_k) * 0.3
W_V = np.random.randn(d_model, d_v) * 0.3


# ============================================================
section("TODO-1：实现数值稳定版 softmax")
# ============================================================
# 公式：softmax(x)_i = exp(x_i) / sum_j exp(x_j)
# 数值稳定技巧：先减去 max，再 exp，避免 exp(大数) 溢出
# 提示：np.max(..., axis=axis, keepdims=True) / np.exp / np.sum

def softmax(x, axis=-1):
    # TODO-1: 实现 softmax
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


_test_x = np.array([[1.0, 2.0, 3.0],
                    [1000.0, 1001.0, 1002.0]])  # 第二行测试数值稳定性
_test_out = softmax(_test_x, axis=-1)
require_shape("TODO-1 softmax shape", _test_out, (2, 3))
require_close("TODO-1 softmax row sum", _test_out.sum(axis=-1), np.ones(2))
require_close("TODO-1 softmax 数值稳定", _test_out, _ref_softmax(_test_x, axis=-1))
print("softmax 第一行 =", _test_out[0])
print("softmax 第二行（大数）=", _test_out[1])


# ============================================================
section("TODO-2：把 embeddings 投影成 Q, K, V")
# ============================================================
# 提示：矩阵乘法 X @ W
# Q.shape 应是 (seq_len, d_k)
# K.shape 应是 (seq_len, d_k)
# V.shape 应是 (seq_len, d_v)

Q = embeddings @ W_Q  # TODO-2
K = embeddings @ W_K  # TODO-2
V = embeddings @ W_V  # TODO-2

require_shape("TODO-2 Q", Q, (seq_len, d_k))
require_shape("TODO-2 K", K, (seq_len, d_k))
require_shape("TODO-2 V", V, (seq_len, d_v))
require_close("TODO-2 Q 数值", Q, embeddings @ W_Q)
print("Q.shape =", Q.shape, "K.shape =", K.shape, "V.shape =", V.shape)


# ============================================================
section("TODO-3：计算注意力分数 scores = Q @ K^T")
# ============================================================
# 提示：K.T 或 K.transpose()
# scores.shape 应是 (seq_len, seq_len)
# scores[i, j] = 第 i 个词对第 j 个词的"原始相关性"

scores = Q @ K.T  # TODO-3

require_shape("TODO-3 scores", scores, (seq_len, seq_len))
require_close("TODO-3 scores 数值", scores, Q @ K.T)
print("scores[0] =", scores[0])


# ============================================================
section("TODO-4：缩放分数 scaled_scores = scores / sqrt(d_k)")
# ============================================================
# 为什么要除以 sqrt(d_k)？
#   d_k 越大，点积的方差越大，softmax 会变得过分尖锐 → 梯度消失
#   除以 sqrt(d_k) 让分数的方差大致回到 1
# 提示：np.sqrt(d_k)

scaled_scores = scores / np.sqrt(d_k) # TODO-4

require_shape("TODO-4 scaled_scores", scaled_scores, (seq_len, seq_len))
require_close("TODO-4 scaled_scores 数值", scaled_scores, scores / np.sqrt(d_k))
print(f"缩放前范围: [{scores.min():+.3f}, {scores.max():+.3f}]")
print(f"缩放后范围: [{scaled_scores.min():+.3f}, {scaled_scores.max():+.3f}]")


# ============================================================
section("TODO-5：softmax 得到注意力权重，再加权求和得到输出")
# ============================================================
# 提示：
#   attention_weights = softmax(scaled_scores, axis=-1)  # 每行和=1
#   output = attention_weights @ V                       # (seq_len, d_v)

attention_weights = softmax(scaled_scores, axis=-1)  # TODO-5
output = attention_weights @ V             # TODO-5

require_shape("TODO-5 weights", attention_weights, (seq_len, seq_len))
require_shape("TODO-5 output", output, (seq_len, d_v))
require_close("TODO-5 每行和=1", attention_weights.sum(axis=-1), np.ones(seq_len))
_ref_out, _ref_w = _ref_self_attention(embeddings, W_Q, W_K, W_V, causal=False)
require_close("TODO-5 weights 数值", attention_weights, _ref_w)
require_close("TODO-5 output 数值", output, _ref_out)

print("\n注意力权重矩阵：")
for i, word in enumerate(sentence):
    row = ", ".join(f"{w:.3f}" for w in attention_weights[i])
    print(f"  {word:4s}: [{row}]")


# ============================================================
section("TODO-6：把上面的步骤封装成 self_attention 函数")
# ============================================================
# 要求：完整复用 TODO-1 ~ TODO-5 的逻辑
# 输入：X (seq_len, d_model), W_Q/W_K (d_model, d_k), W_V (d_model, d_v)
# 输出：(output, weights)
#   output.shape  = (seq_len, d_v)
#   weights.shape = (seq_len, seq_len)

def self_attention(X, W_Q, W_K, W_V):
    # TODO-6: 用 softmax / 矩阵乘法实现完整的 scaled dot-product attention
    # Q = ...
    # K = ...
    # V = ...
    # d_k_local = K.shape[-1]
    # scores_local = ...
    # weights_local = ...
    # output_local = ...
    # return output_local, weights_local
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    d_k_local = K.shape[-1]
    scaled_scores_local = Q @ K.T / np.sqrt(d_k_local)
    weights_local = softmax(scaled_scores_local, axis=-1)
    output_local = weights_local @ V
    return output_local, weights_local


_o, _w = self_attention(embeddings, W_Q, W_K, W_V)
require_shape("TODO-6 output", _o, (seq_len, d_v))
require_shape("TODO-6 weights", _w, (seq_len, seq_len))
require_close("TODO-6 output 与参考一致", _o, _ref_out)
require_close("TODO-6 weights 与参考一致", _w, _ref_w)
print("self_attention OK：output.shape =", _o.shape, "weights.shape =", _w.shape)


# ============================================================
section("TODO-7：构造因果掩码 mask")
# ============================================================
# 目标：让位置 i 看不到位置 j>i（GPT 的"自回归"约束）
# 提示：np.triu(np.ones((seq_len, seq_len)), k=1)
#   - 主对角线以上全为 1（= 未来位置 = 要屏蔽）
#   - 主对角线及以下全为 0（= 当前及过去 = 允许看）

mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # TODO-7

require_shape("TODO-7 mask", mask, (seq_len, seq_len))
require_close("TODO-7 mask 数值", mask, np.triu(np.ones((seq_len, seq_len)), k=1))
require_true("TODO-7 主对角线=0", np.all(np.diag(mask) == 0), "主对角线（当前位置）应允许看，值为 0")
require_true("TODO-7 上三角=1", mask[0, 1] == 1.0 and mask[0, -1] == 1.0, "未来位置应屏蔽，值为 1")
print("mask =\n", mask)


# ============================================================
section("TODO-8：实现带因果掩码的 causal_self_attention")
# ============================================================
# 提示：
#   1. 算 scores 后，把 mask=1 的位置加上一个非常大的负数（例如 -1e9）
#      这样 softmax 后这些位置 ≈ 0
#   2. 不要直接用 -np.inf（数值上更稳定的做法是 -1e9）
#   3. 其余步骤和 TODO-6 一样

def causal_self_attention(X, W_Q, W_K, W_V):
    # TODO-8:
    # Q = ...
    # K = ...
    # V = ...
    # d_k_local = K.shape[-1]
    # scores_local = ...
    # mask_local = np.triu(np.ones((X.shape[0], X.shape[0])), k=1)
    # scores_local = scores_local - mask_local * 1e9
    # weights_local = ...
    # output_local = ...
    # return output_local, weights_local
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    d_k_local = K.shape[-1]
    scaled_scores_local = Q @ K.T / np.sqrt(d_k_local)
    mask_local = np.triu(np.ones((X.shape[0], X.shape[0])), k=1)
    scaled_scores_local = scaled_scores_local - mask_local * 1e9
    weights_local = softmax(scaled_scores_local, axis=-1)
    output_local = weights_local @ V
    return output_local, weights_local


_oc, _wc = causal_self_attention(embeddings, W_Q, W_K, W_V)
_ref_oc, _ref_wc = _ref_self_attention(embeddings, W_Q, W_K, W_V, causal=True)
require_shape("TODO-8 output", _oc, (seq_len, d_v))
require_shape("TODO-8 weights", _wc, (seq_len, seq_len))
require_close("TODO-8 weights 与参考一致", _wc, _ref_wc)
require_close("TODO-8 output 与参考一致", _oc, _ref_oc)

# 关键性质：因果 mask 后，权重矩阵应该是下三角（上三角 ≈ 0）
upper_triangle_sum = np.triu(_wc, k=1).sum()
require_true(
    "TODO-8 上三角应为 0",
    upper_triangle_sum < 1e-6,
    f"上三角之和应接近 0，实际 = {upper_triangle_sum}",
)

print("\n因果注意力权重（GPT 风格，应为下三角）：")
for i, word in enumerate(sentence):
    row = ", ".join(f"{w:.3f}" for w in _wc[i])
    print(f"  {word:4s}: [{row}]")


# ============================================================
section("全部 TODO 校验通过 ✅")
# ============================================================
print("""
你已经手写完成：
  1. 数值稳定的 softmax
  2. Q / K / V 三个线性投影
  3. Scaled Dot-Product Attention 的 5 个步骤
  4. self_attention 函数封装
  5. 因果掩码（np.triu）
  6. causal_self_attention（GPT 用的版本）

下一课：把这个 attention 复制成多个"头"，就是 Multi-Head Attention！
""")
