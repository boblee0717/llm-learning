"""
======================================================
第 2 课：LoRA 微调 —— 用 0.1% 的参数微调大模型
======================================================

核心问题：GPT-3 有 175B 参数，全部微调需要几百 GB 显存，怎么办？
答案：冻结原始权重，只训练两个小矩阵。这就是 LoRA。

数学原理：
  原始权重 W (d × d)，参数量 = d²
  LoRA: W' = W + BA
    B: (d × r)  ← r 远小于 d
    A: (r × d)
  参数量 = 2dr << d²

  例如 d=4096, r=8:
    原始参数: 4096² = 16,777,216 (16M)
    LoRA 参数: 2 × 4096 × 8 = 65,536 (65K)
    压缩比: 256x！

学习目标：
1. 理解低秩分解的数学原理
2. 从零实现 LoRA 层
3. 理解 rank 对效果的影响
4. 掌握参数冻结与选择性训练

运行方式：python3 02_lora.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Part 1: 低秩分解的直觉
# ============================================================

print("=" * 60)
print("Part 1: 低秩分解的直觉")
print("=" * 60)
print("""
核心洞察：微调时，权重的"变化量" ΔW 往往是低秩的。
也就是说，虽然 ΔW 是一个大矩阵，但它可以用两个小矩阵的乘积来近似。

类比：
  原始模型 = 一本百科全书 (W)
  微调 = 只在几页上做批注 (ΔW)
  LoRA = 把批注写在一张小便签纸上 (B × A)
""")

d = 64
r = 4  # rank

W = torch.randn(d, d)

delta_W_full = torch.randn(d, d)
full_rank = torch.linalg.matrix_rank(delta_W_full).item()
print(f"随机矩阵 ({d}×{d}) 的秩: {full_rank}")

B = torch.randn(d, r)
A = torch.randn(r, d)
delta_W_low_rank = B @ A
low_rank = torch.linalg.matrix_rank(delta_W_low_rank).item()
print(f"低秩矩阵 B({d}×{r}) × A({r}×{d}) 的秩: {low_rank}")

print(f"\n参数量对比:")
print(f"  全参数 ΔW: {d*d:,} = {d}×{d}")
print(f"  LoRA B+A:  {d*r + r*d:,} = {d}×{r} + {r}×{d}")
print(f"  压缩比:    {d*d / (2*d*r):.0f}x")
print()


# ============================================================
# Part 2: 从零实现 LoRA 层
# ============================================================

print("=" * 60)
print("Part 2: 从零实现 LoRA 层")
print("=" * 60)


class LoRALinear(nn.Module):
    """
    给一个已有的 Linear 层加上 LoRA 旁路。

    前向传播：
      h = W·x + (B·A)·x · (α/r)

    其中：
      W — 原始权重（冻结，不训练）
      B — 低秩矩阵，初始化为 0
      A — 低秩矩阵，随机初始化
      α — 缩放因子
      r — 秩
    """

    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        original_output = self.original(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output


original = nn.Linear(64, 64)
lora_layer = LoRALinear(original, rank=4, alpha=1.0)

x = torch.randn(2, 10, 64)
output = lora_layer(x)

orig_params = sum(p.numel() for p in [original.weight, original.bias])
lora_params = lora_layer.lora_A.numel() + lora_layer.lora_B.numel()

print(f"原始 Linear 参数量: {orig_params:,} (冻结)")
print(f"LoRA 新增参数量:    {lora_params:,} (可训练)")
print(f"参数比例:           {lora_params/orig_params*100:.2f}%")
print(f"输出 shape:         {output.shape}")
print()


# ============================================================
# Part 3: 给模型添加 LoRA
# ============================================================

print("=" * 60)
print("Part 3: 给模型添加 LoRA")
print("=" * 60)


class MiniTransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)

        q = self.q_proj(h).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.out_proj(attn)

        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPTForLoRA(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, seq_len=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [MiniTransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


def apply_lora(model, rank=4, alpha=1.0, target_modules=("q_proj", "v_proj")):
    """对模型中指定的 Linear 层添加 LoRA"""
    lora_params = []
    replaced = 0

    for name, module in model.named_modules():
        for attr_name in target_modules:
            if hasattr(module, attr_name):
                original_linear = getattr(module, attr_name)
                if isinstance(original_linear, nn.Linear):
                    lora_linear = LoRALinear(original_linear, rank=rank, alpha=alpha)
                    setattr(module, attr_name, lora_linear)
                    lora_params.extend([lora_linear.lora_A, lora_linear.lora_B])
                    replaced += 1

    for param in model.parameters():
        param.requires_grad = False
    for param in lora_params:
        param.requires_grad = True

    return lora_params


model = MiniGPTForLoRA(vocab_size=26, n_layers=4)

total_before = sum(p.numel() for p in model.parameters())
print(f"原始模型总参数: {total_before:,}")

lora_params = apply_lora(model, rank=4, alpha=1.0, target_modules=("q_proj", "v_proj"))

total_after = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = total_after - trainable

print(f"添加 LoRA 后:")
print(f"  总参数:     {total_after:,}")
print(f"  冻结参数:   {frozen:,}")
print(f"  可训练参数: {trainable:,} ({trainable/total_after*100:.2f}%)")
print(f"  LoRA 层数:  {len(lora_params)//2}")
print()


# ============================================================
# Part 4: LoRA 微调实战
# ============================================================

print("=" * 60)
print("Part 4: LoRA 微调实战")
print("=" * 60)

vocab_size = 26
seq_len = 16

def make_pattern_data(pattern_fn, n_samples=500):
    """生成简单模式的数据"""
    data = []
    for _ in range(n_samples):
        seq = [pattern_fn(i) for i in range(seq_len + 1)]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        data.append((x, y))
    return data

repeating_pattern = lambda i: i % vocab_size
data = make_pattern_data(repeating_pattern)
loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

model = MiniGPTForLoRA(vocab_size=vocab_size, n_layers=2, seq_len=seq_len)

print("--- 1. 先做\"预训练\" (全参数) ---")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for epoch in range(5):
    total_loss = 0
    for x, y in loader:
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    if epoch in [0, 4]:
        print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

reverse_pattern = lambda i: (vocab_size - 1 - i) % vocab_size
new_data = make_pattern_data(reverse_pattern)
new_loader = torch.utils.data.DataLoader(new_data, batch_size=32, shuffle=True)

print("\n--- 2. 用 LoRA 微调到新任务 ---")
lora_params = apply_lora(model, rank=4, alpha=1.0)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

optimizer_lora = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3
)

for epoch in range(10):
    total_loss = 0
    for x, y in new_loader:
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer_lora.step()
        optimizer_lora.zero_grad()
        total_loss += loss.item()
    avg_loss = total_loss / len(new_loader)
    if epoch in [0, 4, 9]:
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

print("\n→ LoRA 用很少的参数就能适配新任务！")
print()


# ============================================================
# Part 5: Rank 的影响
# ============================================================

print("=" * 60)
print("Part 5: Rank 对效果的影响")
print("=" * 60)
print("""
rank 越大 → 表达能力越强 → 参数越多 → 越接近全参数微调
rank 越小 → 参数越少 → 可能不够用

实践中常用的 rank:
  - rank=4~8:   简单任务（格式微调、语言适配）
  - rank=16~32: 中等任务（领域适配）
  - rank=64+:   复杂任务（新能力学习）
""")

for rank in [1, 4, 8, 16]:
    model_test = MiniGPTForLoRA(vocab_size=vocab_size, n_layers=2, seq_len=seq_len)
    lora_p = apply_lora(model_test, rank=rank)
    trainable = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_test.parameters())
    print(f"  rank={rank:2d}: 可训练参数 {trainable:5,} / {total:,} ({trainable/total*100:.1f}%)")
print()


# ============================================================
# Part 6: LoRA 权重的合并与保存
# ============================================================

print("=" * 60)
print("Part 6: LoRA 权重的合并")
print("=" * 60)
print("""
训练完成后，可以把 LoRA 权重合并回原始权重：
  W_merged = W + B·A · (α/r)

合并后的模型跟原始模型大小完全一样，推理时没有额外开销。
""")


def merge_lora(model):
    """把 LoRA 权重合并回原始 Linear"""
    merged = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta_w = (module.lora_A @ module.lora_B) * module.scaling
                module.original.weight.data += delta_w.T
            merged += 1
    return merged


model_merge = MiniGPTForLoRA(vocab_size=vocab_size, n_layers=2, seq_len=seq_len)
apply_lora(model_merge, rank=4)

x_test = torch.randint(0, vocab_size, (1, seq_len))
with torch.no_grad():
    out_before = model_merge(x_test)

n_merged = merge_lora(model_merge)
print(f"已合并 {n_merged} 个 LoRA 层")
print("合并后模型可以像普通模型一样保存和推理，没有额外开销")
print()


# ============================================================
# 练习
# ============================================================

print("=" * 60)
print("动手练习")
print("=" * 60)
print("""
练习 1：给 FFN 也加 LoRA
  修改 apply_lora，让 target_modules 也包含 FFN 中的 Linear 层
  对比只加在 Q/V vs 加在所有 Linear 层的参数量和效果

练习 2：不同 alpha 的影响
  固定 rank=4，分别用 alpha=0.5, 1, 2, 4 训练
  观察收敛速度和最终 loss 的变化

练习 3：LoRA 合并验证
  先用 LoRA 模型跑推理得到输出 A
  合并 LoRA 权重后再跑推理得到输出 B
  验证 A 和 B 是否完全一致（应该一致）
""")

print("=" * 60)
print("恭喜完成第 2 课！")
print("下一课我们将学习模型量化 —— 让大模型跑在消费级显卡上")
print("=" * 60)
