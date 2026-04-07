"""
======================================================
第 5 课：推理优化 —— 让模型响应速度快 10 倍
======================================================

核心问题：GPT 生成 100 个 token 为什么这么慢？
答案：每生成一个 token 都要完整跑一遍模型，而且大量计算是重复的。

自回归生成的瓶颈：
  生成第 1 个 token: 处理 prompt (N 个 token)    → 计算量 O(N²)
  生成第 2 个 token: 处理 N+1 个 token           → 计算量 O((N+1)²)
  生成第 3 个 token: 处理 N+2 个 token           → 计算量 O((N+2)²)
  ...
  大量重复计算！前面的 token 每次都要重新算 attention

学习目标：
1. 理解自回归生成的计算瓶颈
2. 从零实现 KV Cache
3. 掌握各种采样策略
4. 理解投机解码的原理

运行方式：python3 05_inference_optimization.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# Part 1: 自回归生成的瓶颈
# ============================================================

print("=" * 60)
print("Part 1: 自回归生成的瓶颈")
print("=" * 60)
print("""
自回归 = 一次只生成一个 token，然后把它拼回输入继续生成

  输入:  "今天天气"
  步骤1: model("今天天气") → "真"
  步骤2: model("今天天气真") → "好"
  步骤3: model("今天天气真好") → "啊"
  ...

问题：步骤2 中 "今天天气" 的 attention 已经在步骤1 算过了！
     步骤3 又全部重新算一遍！
     → 生成 T 个 token 的总计算量 ∝ T × N²

解决：把之前算过的 Key 和 Value 缓存起来 → KV Cache
""")


class SimpleAttention(nn.Module):
    """最简单的单头注意力，用于演示"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """无缓存版本：每次都重新计算所有 K, V"""
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scale = math.sqrt(self.d_model)
        scores = Q @ K.transpose(-2, -1) / scale

        T = x.size(1)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        return attn @ V

    def forward_with_cache(self, x, kv_cache=None):
        """有缓存版本：只计算新 token 的 Q, K, V，复用之前的 K, V"""
        Q = self.W_q(x)
        K_new = self.W_k(x)
        V_new = self.W_v(x)

        if kv_cache is not None:
            K_cached, V_cached = kv_cache
            K = torch.cat([K_cached, K_new], dim=1)
            V = torch.cat([V_cached, V_new], dim=1)
        else:
            K = K_new
            V = V_new

        scale = math.sqrt(self.d_model)
        scores = Q @ K.transpose(-2, -1) / scale
        attn = F.softmax(scores, dim=-1)
        output = attn @ V

        new_cache = (K, V)
        return output, new_cache


d_model = 64
attn = SimpleAttention(d_model)

x = torch.randn(1, 10, d_model)

out_no_cache = attn(x)
print(f"无缓存: 输入 {x.shape} → 输出 {out_no_cache.shape}")
print(f"  每次都计算全部 10 个 token 的 K, V")

cache = None
outputs = []
for t in range(10):
    x_t = x[:, t:t+1, :]
    out_t, cache = attn.forward_with_cache(x_t, cache)
    outputs.append(out_t)

out_with_cache = torch.cat(outputs, dim=1)
diff = (out_no_cache - out_with_cache).abs().max()
print(f"\n有缓存: 逐 token 生成，复用之前的 K, V")
print(f"  缓存中 K, V 的 shape: {cache[0].shape}")
print(f"  与无缓存版本的最大差异: {diff:.8f} (应该接近 0)")
print()


# ============================================================
# Part 2: 完整的 KV Cache 实现
# ============================================================

print("=" * 60)
print("Part 2: 完整的 KV Cache GPT")
print("=" * 60)


class CachedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        h = self.ln1(x)

        Q = self.W_q(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            K_cached, V_cached = kv_cache
            K = torch.cat([K_cached, K], dim=2)
            V = torch.cat([V_cached, V], dim=2)

        new_cache = (K, V)

        scale = math.sqrt(self.head_dim)
        scores = Q @ K.transpose(-2, -1) / scale

        seq_len = K.size(2)
        q_len = Q.size(2)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask[seq_len - q_len:, :]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)

        x = x + out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


class CachedGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_seq_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [CachedTransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, kv_caches=None, start_pos=0):
        B, T = x.shape
        pos = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches else None
            h, new_cache = block(h, cache)
            new_caches.append(new_cache)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, new_caches

    @torch.no_grad()
    def generate_no_cache(self, prompt, max_new_tokens=20):
        """无缓存生成：每步重新计算全部"""
        tokens = prompt.clone()
        for _ in range(max_new_tokens):
            logits, _ = self.forward(tokens)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

    @torch.no_grad()
    def generate_with_cache(self, prompt, max_new_tokens=20):
        """有缓存生成：只计算新 token"""
        logits, caches = self.forward(prompt)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = [prompt, next_token]
        pos = prompt.size(1)

        for _ in range(max_new_tokens - 1):
            logits, caches = self.forward(next_token, caches, start_pos=pos)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token)
            pos += 1

        return torch.cat(tokens, dim=1)


vocab_size = 200
model = CachedGPT(vocab_size=vocab_size, n_layers=2)
prompt = torch.randint(0, vocab_size, (1, 20))

start = time.time()
out_no_cache = model.generate_no_cache(prompt, max_new_tokens=50)
time_no_cache = time.time() - start

start = time.time()
out_with_cache = model.generate_with_cache(prompt, max_new_tokens=50)
time_with_cache = time.time() - start

match = (out_no_cache == out_with_cache).all().item()
print(f"无缓存生成: {time_no_cache:.4f}s")
print(f"有缓存生成: {time_with_cache:.4f}s")
print(f"加速比:     {time_no_cache/time_with_cache:.2f}x")
print(f"结果一致:   {match}")
print(f"\n→ KV Cache 节省了大量重复计算，生成越长加速越明显")
print()


# ============================================================
# Part 3: 采样策略
# ============================================================

print("=" * 60)
print("Part 3: 采样策略")
print("=" * 60)
print("""
模型输出的是概率分布，如何从中选 token？

  1. 贪心 (Greedy):     直接选概率最大的
     → 确定性，但容易重复
  2. Temperature:       调节分布的"平坦度"
     → T<1 更集中，T>1 更随机
  3. Top-K:            只从前 K 个最可能的 token 中采样
     → 避免选到极不可能的 token
  4. Top-P (Nucleus):  选概率累积到 P 的最小 token 集合
     → 自适应的 Top-K
""")

torch.manual_seed(42)
logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.1, -1.0, -2.0, -3.0, -4.0, -5.0])
tokens = [f"tok_{i}" for i in range(len(logits))]

print("原始 logits:", [f"{l:.1f}" for l in logits.tolist()])
print()


def greedy_sample(logits):
    return logits.argmax()


def temperature_sample(logits, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item(), probs


def top_k_sample(logits, k=3, temperature=1.0):
    top_k_logits, top_k_indices = logits.topk(k)
    probs = F.softmax(top_k_logits / temperature, dim=-1)
    idx = torch.multinomial(probs, 1).item()
    return top_k_indices[idx].item(), probs, top_k_indices


def top_p_sample(logits, p=0.9, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = probs.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)

    mask = cumsum - sorted_probs > p
    sorted_probs[mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()

    idx = torch.multinomial(sorted_probs, 1).item()
    return sorted_indices[idx].item(), sorted_probs, sorted_indices


print("1. 贪心采样:")
idx = greedy_sample(logits)
print(f"   选择: {tokens[idx]} (总是选概率最大的)")

print("\n2. Temperature 采样:")
for temp in [0.5, 1.0, 2.0]:
    probs = F.softmax(logits / temp, dim=-1)
    top3 = probs.topk(3)
    desc = ", ".join(f"{tokens[i]}={p:.3f}" for p, i in zip(top3.values, top3.indices))
    print(f"   T={temp}: {desc}")
print("   → T 越小越确定，T 越大越随机")

print("\n3. Top-K 采样 (K=3):")
probs_full = F.softmax(logits, dim=-1)
top3 = probs_full.topk(3)
desc = ", ".join(f"{tokens[i]}={p:.3f}" for p, i in zip(top3.values, top3.indices))
print(f"   只从 [{desc}] 中采样")

print("\n4. Top-P 采样 (P=0.9):")
probs = F.softmax(logits, dim=-1)
sorted_p, sorted_i = probs.sort(descending=True)
cumsum = sorted_p.cumsum(dim=-1)
n_tokens = (cumsum <= 0.9).sum().item() + 1
selected = [(tokens[sorted_i[j].item()], sorted_p[j].item()) for j in range(n_tokens)]
desc = ", ".join(f"{t}={p:.3f}" for t, p in selected)
print(f"   累积概率达到 90% 的 token: [{desc}]")
print()


# ============================================================
# Part 4: 采样策略对比实验
# ============================================================

print("=" * 60)
print("Part 4: 采样策略对比")
print("=" * 60)


@torch.no_grad()
def generate_with_strategy(model, prompt, max_new_tokens=30, strategy="greedy", **kwargs):
    tokens = prompt.clone()
    caches = None
    logits, caches = model.forward(tokens)

    for step in range(max_new_tokens):
        next_logits = logits[:, -1, :]

        if strategy == "greedy":
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        elif strategy == "temperature":
            temp = kwargs.get("temperature", 1.0)
            probs = F.softmax(next_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
        elif strategy == "top_k":
            k = kwargs.get("k", 10)
            temp = kwargs.get("temperature", 1.0)
            top_k_logits, top_k_indices = next_logits.topk(k)
            probs = F.softmax(top_k_logits / temp, dim=-1)
            idx = torch.multinomial(probs, 1)
            next_token = top_k_indices.gather(1, idx)
        elif strategy == "top_p":
            p = kwargs.get("p", 0.9)
            temp = kwargs.get("temperature", 1.0)
            probs = F.softmax(next_logits / temp, dim=-1)
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            mask = (cumsum - sorted_probs) > p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            idx = torch.multinomial(sorted_probs, 1)
            next_token = sorted_indices.gather(1, idx)

        tokens = torch.cat([tokens, next_token], dim=1)
        logits, caches = model.forward(next_token, caches, start_pos=tokens.size(1)-1)

    return tokens


prompt = torch.randint(0, vocab_size, (1, 5))

print("同一个 prompt，不同策略生成的 token 序列：\n")
strategies = [
    ("greedy", {}),
    ("temperature", {"temperature": 0.5}),
    ("temperature", {"temperature": 1.5}),
    ("top_k", {"k": 5}),
    ("top_p", {"p": 0.9}),
]

for name, kwargs in strategies:
    torch.manual_seed(42)
    output = generate_with_strategy(model, prompt, max_new_tokens=15, strategy=name, **kwargs)
    generated = output[0, 5:].tolist()[:10]
    label = f"{name}({kwargs})" if kwargs else name
    print(f"  {label:35s} → {generated}")

print("""
观察：
  - greedy 总是生成相同的序列
  - temperature=0.5 比较保守
  - temperature=1.5 更加随机多样
  - top_k 和 top_p 在随机性和质量之间取得平衡
""")


# ============================================================
# Part 5: 投机解码 (Speculative Decoding)
# ============================================================

print("=" * 60)
print("Part 5: 投机解码")
print("=" * 60)
print("""
核心思想：用小模型"猜"多个 token，再用大模型一次性验证

  传统方式（大模型逐个生成）：
    大模型("今天") → "天"
    大模型("今天天") → "气"
    大模型("今天天气") → "真"
    → 3 次大模型推理

  投机解码：
    小模型("今天") → "天气真"  (猜 3 个)
    大模型("今天天气真") → 验证全部  (1 次推理)
    如果都猜对 → 3 个 token 只用了 1 次大模型推理！
    如果猜错 → 从第一个错的位置开始重新生成

为什么有效：
  - 大部分 token 是可预测的（"天气"后面很可能是"真好"）
  - 小模型推理成本极低（可能只有大模型的 1/10）
  - 验证是一次前向传播，比逐个生成快得多
""")


class DraftModel(nn.Module):
    """小模型（Draft Model），参数少，速度快"""

    def __init__(self, vocab_size, d_model=16, n_layers=1, max_seq_len=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, start_pos=0):
        B, T = x.shape
        pos = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos)
        h = F.gelu(self.fc(h))
        return self.head(h)


@torch.no_grad()
def speculative_decode(
    target_model, draft_model, prompt,
    max_new_tokens=30, num_speculative=4,
):
    """投机解码的简化实现"""
    tokens = prompt.clone()
    accepted_count = 0
    total_drafted = 0
    target_calls = 0

    while tokens.size(1) - prompt.size(1) < max_new_tokens:
        draft_tokens = tokens.clone()
        for _ in range(num_speculative):
            draft_logits = draft_model(draft_tokens)
            next_token = draft_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            draft_tokens = torch.cat([draft_tokens, next_token], dim=1)

        target_logits, _ = target_model(draft_tokens)
        target_calls += 1

        n_accepted = 0
        start_pos = tokens.size(1)
        for i in range(num_speculative):
            pos = start_pos + i
            target_token = target_logits[:, pos - 1, :].argmax(dim=-1)
            draft_token = draft_tokens[:, pos]

            if target_token.item() == draft_token.item():
                n_accepted += 1
            else:
                tokens = torch.cat([
                    tokens,
                    draft_tokens[:, start_pos:start_pos + n_accepted],
                    target_token.unsqueeze(0).unsqueeze(0),
                ], dim=1)
                break
        else:
            bonus_token = target_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([
                tokens,
                draft_tokens[:, start_pos:],
                bonus_token,
            ], dim=1)
            n_accepted = num_speculative

        if n_accepted == 0 and tokens.size(1) == start_pos:
            fallback = target_logits[:, start_pos - 1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, fallback], dim=1)

        accepted_count += n_accepted
        total_drafted += num_speculative

    tokens = tokens[:, :prompt.size(1) + max_new_tokens]

    accept_rate = accepted_count / max(total_drafted, 1)
    return tokens, accept_rate, target_calls


target = CachedGPT(vocab_size=vocab_size, n_layers=2, d_model=64)
draft = DraftModel(vocab_size=vocab_size, d_model=16)

prompt = torch.randint(0, vocab_size, (1, 10))

start = time.time()
out_normal = target.generate_no_cache(prompt, max_new_tokens=30)
time_normal = time.time() - start

start = time.time()
out_spec, accept_rate, n_calls = speculative_decode(
    target, draft, prompt, max_new_tokens=30, num_speculative=4
)
time_spec = time.time() - start

target_params = sum(p.numel() for p in target.parameters())
draft_params = sum(p.numel() for p in draft.parameters())

print(f"大模型参数: {target_params:,}")
print(f"小模型参数: {draft_params:,} ({draft_params/target_params*100:.1f}%)")
print(f"\n普通生成:   {time_normal:.4f}s, 大模型调用 30 次")
print(f"投机解码:   {time_spec:.4f}s, 大模型调用 {n_calls} 次")
print(f"猜测接受率: {accept_rate:.1%}")
print("""
→ 当小模型和大模型的预测一致度高时，投机解码效果显著
  实际场景中（如 Llama-70B + Llama-7B），加速 2-3x 很常见
""")


# ============================================================
# Part 6: KV Cache 的内存分析
# ============================================================

print("=" * 60)
print("Part 6: KV Cache 的内存分析")
print("=" * 60)
print("""
KV Cache 用空间换时间，需要了解它占多少内存。

每一层的 KV Cache 大小：
  K: (batch_size, n_heads, seq_len, head_dim)
  V: (batch_size, n_heads, seq_len, head_dim)
  单层 = 2 × batch_size × n_heads × seq_len × head_dim × bytes_per_element

以 LLaMA-70B 为例：
  n_layers=80, n_heads=64, head_dim=128, FP16
""")

configs = [
    ("GPT-2 (124M)", 12, 12, 64, 1024),
    ("LLaMA-7B", 32, 32, 128, 4096),
    ("LLaMA-70B", 80, 64, 128, 4096),
]

batch_size = 1
bytes_per_element = 2  # FP16

for name, n_layers, n_heads, head_dim, max_seq_len in configs:
    kv_size = 2 * n_layers * batch_size * n_heads * max_seq_len * head_dim * bytes_per_element
    kv_size_gb = kv_size / (1024**3)
    print(f"  {name}:")
    print(f"    seq_len={max_seq_len}, KV Cache = {kv_size_gb:.2f} GB (batch=1)")
    kv_batch32 = kv_size_gb * 32
    print(f"    batch=32 时: {kv_batch32:.1f} GB")
    print()

print("""
优化 KV Cache 内存的方法：
  1. GQA (Grouped Query Attention): K, V 用更少的 head → 内存直降
  2. 量化 KV Cache: FP16 → INT8 → 内存减半
  3. 滑动窗口: 只缓存最近 N 个 token 的 KV
  4. PagedAttention (vLLM): 按需分配内存，减少碎片
""")


# ============================================================
# 练习
# ============================================================

print("=" * 60)
print("动手练习")
print("=" * 60)
print("""
练习 1：KV Cache 加速测量
  修改 prompt 长度（10, 50, 100, 200），固定生成 50 个 token
  画出有/无 KV Cache 的耗时对比图
  验证：prompt 越长，KV Cache 加速比越大

练习 2：采样策略组合
  实现 Top-K + Top-P + Temperature 的组合采样
  参考 HuggingFace 的 generate() 接口

练习 3：投机解码的接受率
  让 draft_model 的参数更大（更接近 target_model）
  观察接受率的变化
  思考：draft_model 多大时，投机解码不再划算？

练习 4：Beam Search
  实现 beam_size=3 的 Beam Search
  对比贪心搜索和 Beam Search 的生成质量
  提示：维护 beam_size 个候选序列，每步扩展并保留 top-beam_size
""")

print("=" * 60)
print("恭喜完成第 5 课！")
print("至此，你已完成第三阶段的全部课程。")
print("你现在具备了理解和使用大模型的完整知识体系！")
print("=" * 60)
