"""
======================================================
第 4 课：推理优化 —— 减少重复计算与生成延迟
======================================================

核心问题：GPT 生成 100 个 token 为什么这么慢？
答案：自回归生成一次只能前进一步；如果没有缓存，还会反复计算旧 token 的 K、V。

自回归生成的瓶颈：
  生成第 1 个 token: 处理 prompt (N 个 token)    → Attention O(N²)
  生成第 2 个 token: 处理 N+1 个 token           → Attention O((N+1)²)
  生成第 3 个 token: 处理 N+2 个 token           → Attention O((N+2)²)
  ...
  无缓存总成本是 Σ(N+t)²，而不是简单的一次 O(N²)。
  KV Cache 把 prefill 保留为 O(N²)，每个 decode token 的 Attention 降为 O(N+t)。

学习目标：
1. 区分 prefill 与 decode 两个阶段
2. 从零实现 KV Cache，并验证缓存前后输出一致
3. 掌握各种采样策略
4. 理解 greedy 版投机解码的控制流与适用边界
5. 会估算 MHA / GQA 的 KV Cache 显存

运行方式：python3 04_inference_optimization.py
"""

import math
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(42)


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
     → 无缓存生成 T 个 token，要做 Σₜ(N+t)² 规模的 attention

解决：把之前算过的 Key 和 Value 缓存起来 → KV Cache
     prefill 仍一次处理整个 prompt；之后 decode 每步只输入 1 个新 token
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
        """
        有缓存版本：只计算新 token 的 Q, K, V，复用之前的 K, V。

        这里每次只传入一个新 token，因此 K 中只有“过去 + 自己”，天然看不到未来，
        decode 步不需要 causal mask。若一次传入多个新 token，仍需对新 chunk 加 mask。
        """
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
torch.testing.assert_close(out_no_cache, out_with_cache, atol=1e-6, rtol=1e-5)
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
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")

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
            # 教学版用 cat 便于看清语义；生产实现会预分配或分页管理，避免每步复制旧 cache。
            K = torch.cat([K_cached, K], dim=2)
            V = torch.cat([V_cached, V], dim=2)

        new_cache = (K, V)

        scale = math.sqrt(self.head_dim)
        scores = Q @ K.transpose(-2, -1) / scale

        seq_len = K.size(2)
        q_len = Q.size(2)
        if q_len > 1:
            # 只创建 (q_len, seq_len) 的 offset causal mask。
            # 单 token decode 时 cache 里只有过去和自己，不需要创建 L×L mask。
            past_len = seq_len - q_len
            query_positions = torch.arange(
                past_len, seq_len, device=x.device
            ).unsqueeze(-1)
            key_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            causal_mask = key_positions <= query_positions
            scores = scores.masked_fill(
                ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)

        x = x + out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


class CachedGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_seq_len=512):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [CachedTransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, kv_caches=None, start_pos=0):
        B, T = x.shape
        if start_pos < 0 or start_pos + T > self.max_seq_len:
            raise ValueError(
                f"位置范围 [{start_pos}, {start_pos + T}) 超过 "
                f"max_seq_len={self.max_seq_len}"
            )
        if kv_caches is not None and len(kv_caches) != len(self.blocks):
            raise ValueError(
                f"kv_caches 有 {len(kv_caches)} 层，但模型有 {len(self.blocks)} 层"
            )

        pos = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = block(h, cache)
            new_caches.append(new_cache)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, new_caches

    @torch.no_grad()
    def generate_no_cache(self, prompt, max_new_tokens=20):
        """无缓存生成：每步重新计算全部"""
        tokens = prompt.clone()
        if max_new_tokens <= 0:
            return tokens

        for _ in range(max_new_tokens):
            logits, _ = self.forward(tokens)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

    @torch.no_grad()
    def generate_with_cache(self, prompt, max_new_tokens=20):
        """有缓存生成：只计算新 token"""
        if max_new_tokens <= 0:
            return prompt.clone()

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
model.eval()
prompt = torch.randint(0, vocab_size, (1, 20))

prompt_len = prompt.size(1)
new_tokens = 50


def median_runtime(fn, repeats=5):
    """先预热一次，再取多次运行的中位数，减少一次性抖动。"""
    fn()
    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - start)
    return result, statistics.median(durations)


out_no_cache, time_no_cache = median_runtime(
    lambda: model.generate_no_cache(prompt, max_new_tokens=new_tokens)
)
out_with_cache, time_with_cache = median_runtime(
    lambda: model.generate_with_cache(prompt, max_new_tokens=new_tokens)
)

match = (out_no_cache == out_with_cache).all().item()
assert match, "KV Cache 只能改变计算路径，不应该改变 greedy 生成结果"

# 下面统计每层、每个 head 的核心工作量；它比玩具 CPU 计时更稳定，也更能解释省在哪。
no_cache_score_work = sum(
    (prompt_len + t) ** 2 for t in range(new_tokens)
)
cache_score_work = prompt_len**2 + sum(
    prompt_len + t for t in range(1, new_tokens)
)
no_cache_kv_tokens = sum(prompt_len + t for t in range(new_tokens))
cache_kv_tokens = prompt_len + new_tokens - 1

print(f"无缓存生成: {time_no_cache:.4f}s")
print(f"有缓存生成: {time_with_cache:.4f}s")
print(f"加速比:     {time_no_cache/time_with_cache:.2f}x")
print(f"结果一致:   {match}")
print(
    f"K/V 投影处理的 token 数: {no_cache_kv_tokens:,} → {cache_kv_tokens:,} "
    f"({no_cache_kv_tokens / cache_kv_tokens:.1f}x)"
)
print(
    f"Attention score 元素数: {no_cache_score_work:,} → {cache_score_work:,} "
    f"({no_cache_score_work / cache_score_work:.1f}x)"
)
print("\n→ 计数展示理论节省；真实加速还受 FFN、内存访问、batch 和硬件影响")
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
    if temperature <= 0:
        raise ValueError(f"temperature 必须 > 0，实际是 {temperature}")
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item(), probs


def top_k_sample(logits, k=3, temperature=1.0):
    if not 1 <= k <= logits.size(-1):
        raise ValueError(f"k 必须在 [1, {logits.size(-1)}]，实际是 {k}")
    if temperature <= 0:
        raise ValueError(f"temperature 必须 > 0，实际是 {temperature}")

    top_k_logits, top_k_indices = logits.topk(k)
    probs = F.softmax(top_k_logits / temperature, dim=-1)
    idx = torch.multinomial(probs, 1).item()
    return top_k_indices[idx].item(), probs, top_k_indices


def top_p_sample(logits, p=0.9, temperature=1.0):
    if not 0 < p <= 1:
        raise ValueError(f"p 必须在 (0, 1]，实际是 {p}")
    if temperature <= 0:
        raise ValueError(f"temperature 必须 > 0，实际是 {temperature}")

    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = probs.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)

    # mask 看的是“当前 token 之前”的累计概率，保证跨过阈值的那个 token 被保留。
    mask = cumsum - sorted_probs >= p
    sorted_probs[mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()

    idx = torch.multinomial(sorted_probs, 1).item()
    return sorted_indices[idx].item(), sorted_probs, sorted_indices


print("1. 贪心采样:")
idx = greedy_sample(logits)
print(f"   选择: {tokens[idx]} (总是选概率最大的)")

print("\n2. Temperature 采样:")
for temp in [0.5, 1.0, 2.0]:
    sampled_idx, probs = temperature_sample(logits, temperature=temp)
    top3 = probs.topk(3)
    desc = ", ".join(f"{tokens[i]}={p:.3f}" for p, i in zip(top3.values, top3.indices))
    print(f"   T={temp}: {desc}，本次采到 {tokens[sampled_idx]}")
print("   → T 越小越确定，T 越大越随机")

print("\n3. Top-K 采样 (K=3):")
sampled_idx, top3_probs, top3_indices = top_k_sample(logits, k=3)
desc = ", ".join(
    f"{tokens[i]}={p:.3f}" for p, i in zip(top3_probs, top3_indices)
)
print(f"   截断后重新归一化: [{desc}]，本次采到 {tokens[sampled_idx]}")

print("\n4. Top-P 采样 (P=0.9):")
sampled_idx, nucleus_probs, sorted_indices = top_p_sample(logits, p=0.9)
kept = nucleus_probs > 0
selected = [
    (tokens[token_idx], prob)
    for token_idx, prob in zip(sorted_indices[kept], nucleus_probs[kept])
]
desc = ", ".join(f"{token}={prob:.3f}" for token, prob in selected)
print(f"   最小 nucleus 重新归一化: [{desc}]，本次采到 {tokens[sampled_idx]}")
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
    if max_new_tokens <= 0:
        return tokens

    logits, caches = model.forward(tokens)

    for step in range(max_new_tokens):
        next_logits = logits[:, -1, :]

        if strategy == "greedy":
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        elif strategy == "temperature":
            temp = kwargs.get("temperature", 1.0)
            if temp <= 0:
                raise ValueError(f"temperature 必须 > 0，实际是 {temp}")
            probs = F.softmax(next_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
        elif strategy == "top_k":
            k = kwargs.get("k", 10)
            temp = kwargs.get("temperature", 1.0)
            if not 1 <= k <= next_logits.size(-1):
                raise ValueError(
                    f"k 必须在 [1, {next_logits.size(-1)}]，实际是 {k}"
                )
            if temp <= 0:
                raise ValueError(f"temperature 必须 > 0，实际是 {temp}")
            top_k_logits, top_k_indices = next_logits.topk(k)
            probs = F.softmax(top_k_logits / temp, dim=-1)
            idx = torch.multinomial(probs, 1)
            next_token = top_k_indices.gather(1, idx)
        elif strategy == "top_p":
            p = kwargs.get("p", 0.9)
            temp = kwargs.get("temperature", 1.0)
            if not 0 < p <= 1:
                raise ValueError(f"p 必须在 (0, 1]，实际是 {p}")
            if temp <= 0:
                raise ValueError(f"temperature 必须 > 0，实际是 {temp}")
            probs = F.softmax(next_logits / temp, dim=-1)
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            mask = (cumsum - sorted_probs) >= p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            idx = torch.multinomial(sorted_probs, 1)
            next_token = sorted_indices.gather(1, idx)
        else:
            raise ValueError(f"未知采样策略: {strategy}")

        tokens = torch.cat([tokens, next_token], dim=1)
        if step == max_new_tokens - 1:
            break
        logits, caches = model.forward(next_token, caches, start_pos=tokens.size(1)-1)

    return tokens


prompt = torch.randint(0, vocab_size, (1, 5))

print("同一个 prompt，不同策略生成的 token 序列：\n")
print("（模型没有训练，只观察确定性与随机性；这些 token id 不能用于比较文本质量。）\n")
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
  - top_k 固定候选数；top_p 根据概率分布动态决定候选数
""")


# ============================================================
# Part 5: Greedy 版投机解码 (Speculative Decoding)
# ============================================================

print("=" * 60)
print("Part 5: Greedy 版投机解码")
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
    如果都猜对 → 接受 3 个草稿 token，还可顺手拿到 1 个 bonus token
    如果猜错 → 接受错位前的草稿，并用大模型在错位处的 token 纠正

为什么有效：
  - 大部分 token 是可预测的（"天气"后面很可能是"真好"）
  - draft 足够便宜，且和 target 的预测一致率足够高
  - target 一次前向可并行验证多个位置，减少串行的大模型调用

重要边界：
  - 下面实现的是 greedy 教学版，验收标准是“结果与 target greedy 完全一致”
  - 正式的随机采样版还要按 min(1, p/q) 接受，并从修正分布采样；
    不能把下面的 argmax 比对直接用于 sampling，否则会改变目标分布
""")


class DraftModel(nn.Module):
    """小模型（Draft Model），参数少，速度快"""

    def __init__(self, vocab_size, d_model=16, max_seq_len=512):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, start_pos=0):
        B, T = x.shape
        if start_pos < 0 or start_pos + T > self.max_seq_len:
            raise ValueError(
                f"位置范围 [{start_pos}, {start_pos + T}) 超过 "
                f"max_seq_len={self.max_seq_len}"
            )
        pos = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos)
        h = F.gelu(self.fc(h))
        return self.head(h)


@torch.no_grad()
def speculative_decode(
    target_model, draft_model, prompt,
    max_new_tokens=30, num_speculative=4,
):
    """
    batch=1 的 greedy 投机解码教学实现。

    每轮让 draft 猜最多 num_speculative 个 token，再用 target 的一次完整前向
    并行验收。为突出验收控制流，这里没有实现生产版的 prefix KV Cache 复用。
    """
    if prompt.size(0) != 1:
        raise ValueError("教学版 speculative_decode 只支持 batch=1")
    if max_new_tokens <= 0:
        return prompt.clone(), 0.0, 0
    if num_speculative <= 0:
        raise ValueError("num_speculative 必须 > 0")

    tokens = prompt.clone()
    accepted_count = 0
    total_drafted = 0
    target_calls = 0

    while tokens.size(1) - prompt.size(1) < max_new_tokens:
        remaining = max_new_tokens - (tokens.size(1) - prompt.size(1))
        draft_count = min(num_speculative, remaining)
        draft_tokens = tokens.clone()
        for _ in range(draft_count):
            draft_logits = draft_model(draft_tokens)
            next_token = draft_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            draft_tokens = torch.cat([draft_tokens, next_token], dim=1)

        # 若本轮不会再生成 bonus token，验证最后一个 draft 只需要它前一位置的
        # logits，无需把最后一个 draft 再送进 target。这与普通 greedy 的边界语义一致：
        # 最后返回的 token 可以尚未被模型处理。
        needs_bonus = remaining > draft_count
        target_input = draft_tokens if needs_bonus else draft_tokens[:, :-1]
        target_logits, _ = target_model(target_input)
        target_calls += 1

        n_accepted = 0
        start_pos = tokens.size(1)
        for i in range(draft_count):
            pos = start_pos + i
            target_token = target_logits[:, pos - 1, :].argmax(dim=-1, keepdim=True)
            draft_token = draft_tokens[:, pos:pos + 1]

            if torch.equal(target_token, draft_token):
                n_accepted += 1
            else:
                tokens = torch.cat(
                    [
                        tokens,
                        draft_tokens[:, start_pos:start_pos + n_accepted],
                        target_token,
                    ],
                    dim=1,
                )
                break
        else:
            accepted_draft = draft_tokens[:, start_pos:]
            if needs_bonus:
                bonus_token = target_logits[:, -1, :].argmax(
                    dim=-1, keepdim=True
                )
                tokens = torch.cat([tokens, accepted_draft, bonus_token], dim=1)
            else:
                tokens = torch.cat([tokens, accepted_draft], dim=1)

        accepted_count += n_accepted
        total_drafted += draft_count

    accept_rate = accepted_count / max(total_drafted, 1)
    return tokens, accept_rate, target_calls


target = CachedGPT(vocab_size=vocab_size, n_layers=2, d_model=64)
draft = DraftModel(vocab_size=vocab_size, d_model=16)
target.eval()
draft.eval()

# 两个未经训练的随机模型几乎不会预测一致，演示会退化成接受率 0%。
# 为隔离并展示“验收控制流”，这里让两个输出头都固定偏好同一个 token。
# 这是 100% 接受率的可控上界实验，不是对真实模型质量或端到端性能的模拟。
preferred_token = 7
with torch.no_grad():
    target.head.weight.zero_()
    target.head.bias.zero_()
    target.head.bias[preferred_token] = 1.0
    draft.head.weight.zero_()
    draft.head.bias.zero_()
    draft.head.bias[preferred_token] = 1.0

prompt = torch.randint(0, vocab_size, (1, 10))

start = time.perf_counter()
out_normal = target.generate_no_cache(prompt, max_new_tokens=30)
time_normal = time.perf_counter() - start

start = time.perf_counter()
out_spec, accept_rate, n_calls = speculative_decode(
    target, draft, prompt, max_new_tokens=30, num_speculative=4
)
time_spec = time.perf_counter() - start

same_output = torch.equal(out_normal, out_spec)
assert same_output, "greedy 投机解码必须与 target 的普通 greedy 输出完全一致"

target_params = sum(p.numel() for p in target.parameters())
draft_params = sum(p.numel() for p in draft.parameters())

print(f"大模型参数: {target_params:,}")
print(f"小模型参数: {draft_params:,} ({draft_params/target_params*100:.1f}%)")
print(f"控制实验:   target / draft 都固定偏好 token {preferred_token}")
print(f"\n普通生成:   {time_normal:.4f}s, target 调用 30 次")
print(f"投机解码:   {time_spec:.4f}s, target 调用 {n_calls} 次")
print(f"猜测接受率: {accept_rate:.1%}")
print(f"结果一致:   {same_output}")
print("""
→ 这组受控实验展示了高接受率如何减少串行 target 调用
  玩具 CPU 耗时不代表真实服务；实际收益取决于接受率、draft 成本、batch 和硬件
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
  K: (batch_size, n_kv_heads, seq_len, head_dim)
  V: (batch_size, n_kv_heads, seq_len, head_dim)
  单层 = 2 × batch_size × n_kv_heads × seq_len × head_dim × bytes_per_element

关键：MHA 中 n_kv_heads = n_query_heads；
     GQA / MQA 让多组 Query 共享更少的 K、V head，显存按 n_kv_heads 计算。
""")

configs = [
    # 名称, 层数, Query heads, KV heads, head_dim, context
    ("GPT-2 (124M, MHA)", 12, 12, 12, 64, 1024),
    ("LLaMA-2 7B (MHA)", 32, 32, 32, 128, 4096),
    ("LLaMA-2 70B (GQA)", 80, 64, 8, 128, 4096),
]

batch_size = 1
bytes_per_element = 2  # FP16

for name, n_layers, n_query_heads, n_kv_heads, head_dim, max_seq_len in configs:
    kv_size = (
        2
        * n_layers
        * batch_size
        * n_kv_heads
        * max_seq_len
        * head_dim
        * bytes_per_element
    )
    kv_size_gb = kv_size / (1024**3)
    print(f"  {name}:")
    print(f"    Query heads={n_query_heads}, KV heads={n_kv_heads}")
    print(f"    seq_len={max_seq_len}, KV Cache = {kv_size_gb:.2f} GB (batch=1)")
    kv_batch32 = kv_size_gb * 32
    print(f"    batch=32 时: {kv_batch32:.1f} GB")
    print()

print("""
优化 KV Cache 内存的方法：
  1. GQA (Grouped Query Attention): K, V 用更少的 head → 内存直降
  2. 量化 KV Cache: FP16 → INT8 → 内存减半
  3. 滑动窗口: 只缓存最近 N 个 token 的 KV
  4. PagedAttention (vLLM): 分页管理、按需分配，减少碎片和预留浪费

注意：PagedAttention 不改变“每个有效 token 需要多少 K/V 数据”的理论下限，
     它优化的是 KV Cache 的分配、复用与碎片问题。
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
  先把 draft 输出头偏好的 token 从 7 改成 8，观察接受率从 100% 降到 0%
  再思考真实场景中如何通过蒸馏提高 draft / target 一致率
  注意：draft 参数更多不等于预测自然更接近，关键是“一致率 / draft 成本”的权衡

练习 4：Beam Search
  实现 beam_size=3 的 Beam Search
  对比贪心搜索和 Beam Search 的生成质量
  提示：维护 beam_size 个候选序列，每步扩展并保留 top-beam_size
""")

print("=" * 60)
print("恭喜完成第 4 课！")
print("你现在能解释 KV Cache、采样策略和 greedy 投机解码的核心机制。")
print("下一步：按需阅读 MQA / GQA / PagedAttention，再进入第 5 课分布式训练专题。")
print("=" * 60)
