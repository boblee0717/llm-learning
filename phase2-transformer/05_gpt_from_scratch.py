"""
第 5 课：从零构建 GPT
======================
用 PyTorch 实现一个完整的 GPT 模型，并训练它生成文本

核心问题：
- GPT 的完整架构长什么样？
- 模型怎么学会生成下一个词？
- Temperature 和 Top-K 采样是什么？
- 训练一个能生成文本的模型需要什么？

与大模型的关系：
- 这就是 GPT 的完整架构，只是小了 1000 倍
- ChatGPT 和你写的模型本质上是同一种东西
- 理解了这个，你就理解了所有 GPT 系列模型

前置知识：
- 前 4 课的所有内容
- 基础 PyTorch（会在代码中解释）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Part 1: GPT 的配置
# ============================================================

print("=" * 60)
print("Part 1: 模型配置")
print("=" * 60)


class GPTConfig:
    """GPT 模型配置 —— 所有超参数集中管理"""
    def __init__(
        self,
        vocab_size=256,       # 词汇表大小（我们用字符级别，ASCII 256 个字符）
        block_size=64,        # 最大序列长度（上下文窗口）
        n_layer=4,            # Transformer Block 的层数
        n_head=4,             # 注意力头数
        n_embd=64,            # 嵌入维度
        dropout=0.1,          # Dropout 率
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


config = GPTConfig()
print(f"词汇表大小: {config.vocab_size}")
print(f"上下文窗口: {config.block_size}")
print(f"层数: {config.n_layer}")
print(f"注意力头数: {config.n_head}")
print(f"嵌入维度: {config.n_embd}")


# ============================================================
# Part 2: 各组件的 PyTorch 实现
# ============================================================

print("\n" + "=" * 60)
print("Part 2: GPT 组件实现")
print("=" * 60)


class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.d_k = config.n_embd // config.n_head

        # Q, K, V 合并成一个矩阵，效率更高
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码：上三角矩阵，注册为 buffer（不参与训练）
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1)
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, embedding_dim

        # 一次性计算 Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # 分成多个头: (B, T, C) → (B, n_head, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # 注意力分数
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 1, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # 加权求和
        out = weights @ v  # (B, n_head, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


class FeedForward(nn.Module):
    """Feed-Forward Network with GELU"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.c_fc(x))
        x = self.dropout(self.c_proj(x))
        return x


class TransformerBlock(nn.Module):
    """一个 Transformer Block (Pre-Norm)"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


# ============================================================
# Part 3: 完整的 GPT 模型
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 完整的 GPT 模型")
print("=" * 60)


class GPT(nn.Module):
    """
    完整的 GPT 模型

    结构:
    Token Embedding + Position Embedding
    → N × TransformerBlock
    → LayerNorm
    → Linear (输出 logits)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：token embedding 和 output projection 共享权重
        self.token_embedding.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token 索引
        targets: (B, T) 目标 token 索引（训练时提供）
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"序列长度 {T} 超过最大长度 {self.config.block_size}"

        # Token + Position Embedding
        tok_emb = self.token_embedding(idx)                    # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T))     # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # N 个 Transformer Block
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成文本

        temperature: 控制随机性（越低越确定，越高越随机）
        top_k: 只从概率最高的 k 个 token 中采样
        """
        for _ in range(max_new_tokens):
            # 截断到最大上下文长度
            idx_crop = idx[:, -self.config.block_size:]

            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature  # 只取最后一个位置

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# 创建模型
model = GPT(config)
n_params = model.count_parameters()

print(f"\n模型总参数量: {n_params:,}")
print(f"\n模型结构:")
for name, module in model.named_children():
    if hasattr(module, 'weight'):
        print(f"  {name}: {module.weight.shape}")
    else:
        print(f"  {name}: {module}")


# ============================================================
# Part 4: 训练循环
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 训练 —— 用莎士比亚文本")
print("=" * 60)

# 用一段简单的重复文本来训练（真正训练需要更多数据）
training_text = """To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them To die to sleep
No more and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to Tis a consummation
Devoutly to be wished To die to sleep
To sleep perchance to dream""" * 5

# 字符级别的分词（最简单的 tokenizer）
chars = sorted(set(training_text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
actual_vocab_size = len(chars)

print(f"训练文本长度: {len(training_text)} 字符")
print(f"实际词汇表大小: {actual_vocab_size}")
print(f"词汇表: {''.join(chars)}")

# 重新创建适合实际词汇表大小的模型
config = GPTConfig(vocab_size=actual_vocab_size, block_size=64, n_layer=4, n_head=4, n_embd=64)
model = GPT(config)

# 编码文本
data = torch.tensor([char_to_idx[c] for c in training_text], dtype=torch.long)


def get_batch(data, block_size, batch_size):
    """随机采样训练批次"""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
model.train()

print("\n--- 开始训练 ---")
n_steps = 500
batch_size = 8

for step in range(n_steps):
    x_batch, y_batch = get_batch(data, config.block_size, batch_size)
    logits, loss = model(x_batch, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0 or step == n_steps - 1:
        print(f"  Step {step:>4d}/{n_steps}  Loss: {loss.item():.4f}")


# ============================================================
# Part 5: 文本生成
# ============================================================

print("\n" + "=" * 60)
print("Part 5: 生成文本！")
print("=" * 60)

model.eval()

# 生成函数
def generate_text(prompt, max_tokens=100, temperature=1.0, top_k=None):
    tokens = [char_to_idx.get(c, 0) for c in prompt]
    idx = torch.tensor([tokens], dtype=torch.long)
    output = model.generate(idx, max_new_tokens=max_tokens,
                            temperature=temperature, top_k=top_k)
    return ''.join([idx_to_char.get(i, '?') for i in output[0].tolist()])


# 用不同的参数生成
prompt = "To be or"

print(f"\n提示词: '{prompt}'")
print(f"\n--- Temperature = 0.5 (更确定) ---")
print(generate_text(prompt, max_tokens=100, temperature=0.5, top_k=10))

print(f"\n--- Temperature = 1.0 (标准) ---")
print(generate_text(prompt, max_tokens=100, temperature=1.0, top_k=10))

print(f"\n--- Temperature = 1.5 (更随机) ---")
print(generate_text(prompt, max_tokens=100, temperature=1.5, top_k=10))

print("""
Temperature 的作用:
  - 低温 (< 1.0): 输出更确定、更保守、可能重复
  - 高温 (> 1.0): 输出更随机、更有创意、可能不连贯
  - 温度 = 0:    贪心解码，永远选概率最高的

Top-K 采样:
  - 只从概率最高的 K 个 token 中选择
  - 避免采样到非常不可能的 token
  - ChatGPT 用的是 Top-P (nucleus sampling)
""")


# ============================================================
# Part 6: 理解训练的本质
# ============================================================

print("=" * 60)
print("Part 6: 训练的本质 —— 预测下一个 Token")
print("=" * 60)

print("""
GPT 训练的核心任务:
  给定前面的所有 token，预测下一个 token

  输入: "To be or not to"
  目标: "o be or not to "
         ^  每个位置预测下一个字符

  这就是所谓的 "Language Modeling"
  - 模型并不"理解"语言
  - 它只是在做非常好的"下一个词预测"
  - 但当预测做得足够好时，它就表现出了"理解"

从我们的小模型到 ChatGPT:
  1. 更大的模型 (175B vs 我们的几万参数)
  2. 更多的数据 (整个互联网 vs 几行莎士比亚)
  3. 更好的分词器 (BPE vs 字符级别)
  4. RLHF 对齐 (让模型遵循指令)
  
  但核心架构完全一样！
""")

# 查看模型的预测
model.eval()
test_text = "To be or not"
test_tokens = torch.tensor([[char_to_idx[c] for c in test_text]], dtype=torch.long)

with torch.no_grad():
    logits, _ = model(test_tokens)

print(f"输入: '{test_text}'")
print(f"\n每个位置的 top-3 预测:")
for i, char in enumerate(test_text):
    probs = F.softmax(logits[0, i], dim=-1)
    top3 = torch.topk(probs, 3)
    predictions = [(idx_to_char[idx.item()], prob.item())
                    for idx, prob in zip(top3.indices, top3.values)]
    actual_next = test_text[i + 1] if i + 1 < len(test_text) else "?"
    pred_str = ", ".join([f"'{c}' ({p:.2f})" for c, p in predictions])
    print(f"  位置 {i} '{char}' → 预测: [{pred_str}]  实际下一个: '{actual_next}'")


# ============================================================
# 练习
# ============================================================

print("\n" + "=" * 60)
print("练习")
print("=" * 60)
print("""
1. 调整训练步数 (500 → 2000)，观察 loss 和生成质量的变化

2. 修改模型大小: 试试 n_layer=2, n_head=2, n_embd=32
   - 更小的模型训练更快，但效果更差

3. 用不同的训练文本（比如中文、代码、歌词）

4. 实现 Top-P (Nucleus) 采样:
   - 按概率从高到低排序
   - 选择累计概率超过 P 的最小集合
   - 从这个集合中采样

5. (进阶) 用 tiktoken 替换字符级分词器:
   import tiktoken
   enc = tiktoken.get_encoding("gpt2")
   tokens = enc.encode("Hello world")

6. (进阶) 下载 tiny-shakespeare 数据集，用更多数据训练:
   wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
""")
