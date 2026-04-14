"""
======================================================
第 1 课（自写版）：词嵌入与位置编码
======================================================

用法：
1) 运行：python3 01_word_embeddings_self_write.py
2) 按 TODO-1 到 TODO-6 逐个补全
3) 每完成一个 TODO 就运行一次，查看校验报错

目标：
- 理解 one-hot 的局限
- 理解 embedding lookup 的本质
- 手写正弦位置编码
- 串起 Token Embedding + Position Encoding 的完整输入流程
"""

import numpy as np


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


class ValidationError(Exception):
    """统一的练习校验错误。"""


def require_not_none(name: str, value, hint: str) -> None:
    if value is None:
        raise ValidationError(f"{name} 错误：结果是 None。{hint}")


def require_close(name: str, actual, expected, hint: str = "", atol: float = 1e-6) -> None:
    try:
        if not np.allclose(actual, expected, atol=atol):
            raise ValidationError(
                f"{name} 错误：数值不正确。\nactual={actual}\nexpected={expected}\n{hint}"
            )
    except TypeError as err:
        raise ValidationError(f"{name} 错误：类型不正确。{hint}\n底层错误: {err}") from err


def require_shape(name: str, actual, expected_shape, hint: str = "") -> None:
    real_shape = getattr(actual, "shape", None) if actual is not None else None
    if real_shape != expected_shape:
        raise ValidationError(
            f"{name} 错误：shape 不正确。actual_shape={real_shape}, expected_shape={expected_shape}\n{hint}"
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


def reference_sinusoidal_position_encoding(max_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


np.random.seed(42)

section("第一部分：One-Hot 与词表")
vocab = ["我", "喜欢", "猫", "狗", "爱", "你"]
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}

for word in vocab:
    one_hot = np.zeros(vocab_size)
    one_hot[word_to_idx[word]] = 1
    print(f"{word}: {one_hot}")
print()


section("第二部分：Embedding 矩阵与查表（TODO-1, TODO-2）")
embedding_dim = 4

# TODO-1:
# 构造 embedding_matrix，形状应为 (vocab_size, embedding_dim)
# 建议：np.random.randn(vocab_size, embedding_dim) * 0.1
# embedding_matrix = ...
embedding_matrix = None

target_word = "猫"
target_idx = word_to_idx[target_word]

# TODO-2:
# 取出 target_word 的 embedding 向量
# cat_embedding = ...
cat_embedding = None

print(f"embedding_matrix.shape = {getattr(embedding_matrix, 'shape', None)}")
print(f"{target_word} 的向量 = {cat_embedding}")
print()


section("第三部分：语义相似度（TODO-3）")
semantic_embeddings = {
    "猫": np.array([0.9, 0.1, -0.3, 0.7]),
    "狗": np.array([0.9, 0.2, 0.3, 0.8]),
    "爱": np.array([0.0, 0.9, 0.0, 0.9]),
}

# TODO-3:
# 计算两组余弦相似度
# sim_cat_dog = ...
# sim_cat_love = ...
sim_cat_dog = None
sim_cat_love = None

print(f"sim(猫, 狗) = {sim_cat_dog}")
print(f"sim(猫, 爱) = {sim_cat_love}")
print()


section("第四部分：正弦位置编码（TODO-4）")


def sinusoidal_position_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    TODO-4:
    按公式实现正弦位置编码
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    # pe = ...
    # position = ...
    # div_term = ...
    # pe[:, 0::2] = ...
    # pe[:, 1::2] = ...
    # return pe
    raise NotImplementedError("TODO-4 未完成：请实现 sinusoidal_position_encoding")


max_len = 10
d_model = 8
pe_error = None
pe = None
try:
    pe = sinusoidal_position_encoding(max_len, d_model)
except Exception as err:  # noqa: BLE001
    pe_error = err

print(f"pe.shape = {getattr(pe, 'shape', None) if pe is not None else f'执行失败 -> {pe_error}'}")
print()


section("第五部分：完整输入嵌入（TODO-5）")
sentence = ["我", "喜欢", "猫"]
token_indices = [word_to_idx[w] for w in sentence]

token_embedding = np.random.randn(vocab_size, d_model) * 0.1
pos_encoding = reference_sinusoidal_position_encoding(20, d_model)

token_vecs = token_embedding[token_indices]
pos_vecs = pos_encoding[: len(sentence)]

# TODO-5:
# 计算最终输入：final_embedding = token_vecs + pos_vecs
# final_embedding = ...
final_embedding = None

print(f"token_vecs.shape = {token_vecs.shape}")
print(f"pos_vecs.shape = {pos_vecs.shape}")
print(f"final_embedding.shape = {getattr(final_embedding, 'shape', None)}")
print()


section("第六部分：位置相似度（TODO-6）")
# TODO-6:
# 用 pe（第四部分生成的位置编码）比较位置相似度：
# pos01_sim = cosine_similarity(pe[0], pe[1])
# pos049_sim = cosine_similarity(pe[0], pe[9])
pos01_sim = None
pos049_sim = None

print(f"sim(pos0, pos1) = {pos01_sim}")
print(f"sim(pos0, pos9) = {pos049_sim}")
print()


def validate_all() -> None:
    require_not_none("TODO-1", embedding_matrix, "请创建 embedding_matrix。")
    require_shape(
        "TODO-1",
        embedding_matrix,
        (vocab_size, embedding_dim),
        "embedding_matrix 的形状应为 (vocab_size, embedding_dim)。",
    )

    require_not_none("TODO-2", cat_embedding, "请从 embedding_matrix 中按 index 取向量。")
    require_shape("TODO-2", cat_embedding, (embedding_dim,), "单词 embedding 应是一维向量。")
    require_close("TODO-2", cat_embedding, embedding_matrix[target_idx], "查表结果应与矩阵行一致。")

    require_not_none("TODO-3", sim_cat_dog, "请计算 sim(猫, 狗)。")
    require_not_none("TODO-3", sim_cat_love, "请计算 sim(猫, 爱)。")
    require_true = lambda name, cond, hint: (_ for _ in ()).throw(ValidationError(f"{name} 错误：{hint}")) if not cond else None
    require_true("TODO-3", sim_cat_dog > sim_cat_love, "语义上猫和狗应比猫和爱更相似。")

    if pe_error is not None:
        raise ValidationError(f"TODO-4 错误：位置编码函数执行失败。底层错误: {pe_error}")
    require_not_none("TODO-4", pe, "请返回位置编码矩阵。")
    require_shape("TODO-4", pe, (max_len, d_model), "位置编码形状应为 (max_len, d_model)。")
    require_close(
        "TODO-4",
        pe,
        reference_sinusoidal_position_encoding(max_len, d_model),
        "位置编码数值与参考实现不一致。",
    )

    require_not_none("TODO-5", final_embedding, "请计算 final_embedding。")
    require_shape("TODO-5", final_embedding, (len(sentence), d_model), "最终输入形状应为 (seq_len, d_model)。")
    require_close("TODO-5", final_embedding, token_vecs + pos_vecs, "最终输入应是 token_vecs + pos_vecs。")

    require_not_none("TODO-6", pos01_sim, "请计算 sim(pos0, pos1)。")
    require_not_none("TODO-6", pos049_sim, "请计算 sim(pos0, pos9)。")
    require_true("TODO-6", pos01_sim > pos049_sim, "通常相邻位置编码应比远位置更相似。")


section("自写版骨架就绪")
print("你可以按 TODO-1 到 TODO-6 逐个补全。")
print("开始自动校验...")
validate_all()
print("校验通过：你当前实现正确。")
