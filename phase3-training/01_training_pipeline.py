"""
======================================================
第 1 课：完整训练流程 —— 工业级模型训练的实战
======================================================

上一阶段我们从零搭建了 GPT，但训练循环比较简陋。
真正训练大模型时，还需要考虑：
- 如何高效加载数据？
- 如何用 FP16 省显存加速？
- 显存不够时怎么办？（梯度累积）
- 如何保存和恢复训练？

学习目标：
1. 掌握 PyTorch DataLoader 的使用
2. 理解混合精度训练（AMP）
3. 实现梯度累积
4. 实现完整的 checkpoint 机制

运行方式：python3 01_training_pipeline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import os
import time


# ============================================================
# Part 1: 数据集与 DataLoader
# ============================================================

print("=" * 60)
print("Part 1: 数据集与 DataLoader")
print("=" * 60)


class TextDataset(Dataset):
    """把文本转换成训练数据：每个样本是 (输入序列, 目标序列)"""

    def __init__(self, text, seq_len, vocab_size=50):
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)

        data = [self.char_to_idx[c] for c in text]
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

    def decode(self, indices):
        return "".join(self.idx_to_char[i.item()] for i in indices)


text = "to be or not to be that is the question " * 100
seq_len = 32
dataset = TextDataset(text, seq_len)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
)

batch_x, batch_y = next(iter(dataloader))
print(f"数据集大小: {len(dataset)} 个样本")
print(f"每个 batch: x.shape={batch_x.shape}, y.shape={batch_y.shape}")
print(f"  x = 输入序列 (batch_size, seq_len)")
print(f"  y = 目标序列 (x 右移一位)")
print(f"\n示例:")
print(f"  输入: '{dataset.decode(batch_x[0])}'")
print(f"  目标: '{dataset.decode(batch_y[0])}'")
print()


# ============================================================
# Part 2: 定义一个小型 GPT 模型
# ============================================================

print("=" * 60)
print("Part 2: 定义模型")
print("=" * 60)


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, seq_len=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, h, tgt_mask=causal_mask, memory_mask=causal_mask)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits


model = MiniGPT(vocab_size=dataset.vocab_size, seq_len=seq_len)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"可训练参数: {trainable_params:,}")
print()


# ============================================================
# Part 3: 标准训练循环（五步曲）
# ============================================================

print("=" * 60)
print("Part 3: 标准训练循环")
print("=" * 60)
print("""
每一步训练都包含 5 个操作：
  1. forward  — 前向传播，计算预测
  2. loss     — 计算损失（交叉熵）
  3. backward — 反向传播，计算梯度
  4. step     — 更新参数
  5. zero_grad— 清零梯度（否则会累积）
""")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备: {device}")

model = MiniGPT(vocab_size=dataset.vocab_size, seq_len=seq_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        logits = model(batch_x)                           # 1. forward
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch_y.view(-1),
        )                                                 # 2. loss
        loss.backward()                                   # 3. backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()                                  # 4. step
        optimizer.zero_grad()                             # 5. zero_grad

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


print("--- 标准训练 (3 epochs) ---")
for epoch in range(3):
    start = time.time()
    loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
    elapsed = time.time() - start
    print(f"  Epoch {epoch+1}: loss={loss:.4f}, 耗时={elapsed:.2f}s")
print()


# ============================================================
# Part 4: 混合精度训练（AMP）
# ============================================================

print("=" * 60)
print("Part 4: 混合精度训练 (AMP)")
print("=" * 60)
print("""
FP32 (32位浮点) → FP16 (16位浮点)
- 显存减半
- 计算速度翻倍（在支持的 GPU 上）
- 精度基本不受影响

关键：用 GradScaler 防止 FP16 下梯度下溢（太小变成 0）
""")

model_amp = MiniGPT(vocab_size=dataset.vocab_size, seq_len=seq_len).to(device)
optimizer_amp = torch.optim.AdamW(model_amp.parameters(), lr=3e-4)
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))


def train_one_epoch_amp(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(batch_x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch_y.view(-1),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


print("--- AMP 训练 (3 epochs) ---")
for epoch in range(3):
    start = time.time()
    loss = train_one_epoch_amp(model_amp, dataloader, optimizer_amp, scaler, device)
    elapsed = time.time() - start
    print(f"  Epoch {epoch+1}: loss={loss:.4f}, 耗时={elapsed:.2f}s")

print("\n→ 在 GPU 上 AMP 通常快 1.5-2x，在 CPU 上差别不大")
print()


# ============================================================
# Part 5: 梯度累积 —— 小显存模拟大 batch
# ============================================================

print("=" * 60)
print("Part 5: 梯度累积")
print("=" * 60)
print("""
问题：大模型训练需要大 batch_size（如 512），但显存只够跑 batch=16
解决：累积多个小 batch 的梯度，再一次性更新

等效关系：
  batch_size=16, accumulation_steps=4
  ≡ batch_size=64 (但显存只占 16 的量)
""")

model_accum = MiniGPT(vocab_size=dataset.vocab_size, seq_len=seq_len).to(device)
optimizer_accum = torch.optim.AdamW(model_accum.parameters(), lr=3e-4)
accumulation_steps = 4


def train_one_epoch_accum(model, dataloader, optimizer, device, accum_steps):
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    for i, (batch_x, batch_y) in enumerate(dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        logits = model(batch_x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch_y.view(-1),
        )
        loss = loss / accum_steps
        loss.backward()

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        num_batches += 1

    return total_loss / num_batches


print("--- 梯度累积训练 (3 epochs, accum_steps=4) ---")
for epoch in range(3):
    loss = train_one_epoch_accum(
        model_accum, dataloader, optimizer_accum, device, accumulation_steps
    )
    print(f"  Epoch {epoch+1}: loss={loss:.4f} (等效 batch_size={16*accumulation_steps})")
print()


# ============================================================
# Part 6: 学习率调度器
# ============================================================

print("=" * 60)
print("Part 6: 学习率调度器")
print("=" * 60)
print("""
大模型训练标配：Warmup + Cosine Decay
  - Warmup: 前几百步学习率从 0 线性增长到最大值
    → 避免训练初期参数剧烈震荡
  - Cosine Decay: 之后学习率按余弦曲线下降到接近 0
    → 后期精细调整参数
""")


def get_lr(step, total_steps, max_lr=3e-4, warmup_steps=100):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


total_steps = 1000
sample_steps = [0, 50, 100, 200, 500, 800, 1000]
print("学习率变化示例 (warmup=100, total=1000):")
for s in sample_steps:
    lr = get_lr(s, total_steps)
    bar = "█" * int(lr / 3e-4 * 30)
    print(f"  step {s:4d}: lr={lr:.6f}  {bar}")
print()


# ============================================================
# Part 7: 模型保存与恢复（Checkpoint）
# ============================================================

print("=" * 60)
print("Part 7: Checkpoint 保存与恢复")
print("=" * 60)
print("""
训练大模型可能要跑几天甚至几周。
如果中途断电或 OOM，没有 checkpoint 就得从头来。

Checkpoint 需要保存：
  - model.state_dict()     — 模型参数
  - optimizer.state_dict() — 优化器状态（动量等）
  - epoch / step           — 训练进度
  - loss                   — 当前损失
""")

checkpoint_dir = "/tmp/llm_learning_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"  ✓ Checkpoint 已保存: {path}")


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"  ✓ 从 Checkpoint 恢复: epoch={epoch}, loss={loss:.4f}")
    return epoch, loss


ckpt_path = os.path.join(checkpoint_dir, "mini_gpt.pt")
save_checkpoint(model, optimizer, epoch=3, loss=0.5, path=ckpt_path)

model_restored = MiniGPT(vocab_size=dataset.vocab_size, seq_len=seq_len).to(device)
optimizer_restored = torch.optim.AdamW(model_restored.parameters(), lr=3e-4)
load_checkpoint(model_restored, optimizer_restored, ckpt_path)
print()


# ============================================================
# Part 8: 把所有技巧整合到一起
# ============================================================

print("=" * 60)
print("Part 8: 完整训练流程（整合所有技巧）")
print("=" * 60)


def full_training_loop(
    model, train_loader, device,
    max_epochs=5,
    max_lr=3e-4,
    warmup_steps=50,
    accum_steps=2,
    use_amp=True,
    checkpoint_dir=checkpoint_dir,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    total_steps = max_epochs * len(train_loader)
    global_step = 0
    best_loss = float("inf")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            lr = get_lr(global_step, total_steps, max_lr, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(batch_x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch_y.view(-1),
                )
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps
            num_batches += 1
            global_step += 1

        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch+1}/{max_epochs}: loss={avg_loss:.4f}, lr={lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                os.path.join(checkpoint_dir, "best_model.pt"),
            )

    print(f"\n训练完成！最佳 loss: {best_loss:.4f}")


model_full = MiniGPT(vocab_size=dataset.vocab_size, seq_len=seq_len).to(device)
full_training_loop(model_full, dataloader, device, max_epochs=5)
print()


# ============================================================
# 练习
# ============================================================

print("=" * 60)
print("动手练习")
print("=" * 60)
print("""
练习 1：Early Stopping
  在 full_training_loop 中加入早停机制：
  如果连续 3 个 epoch 验证集 loss 没有下降，就停止训练。
  提示：需要把 dataset 分成 train/val 两部分

练习 2：学习率调度实验
  修改 get_lr 函数，实现线性衰减（而非余弦），对比训练曲线。

练习 3：梯度累积验证
  设置 batch_size=4, accum_steps=8 和 batch_size=32, accum_steps=1
  验证最终 loss 是否接近（理论上应该接近）。
""")

print("=" * 60)
print("恭喜完成第 1 课！")
print("下一课我们将学习 LoRA —— 用 0.1% 的参数微调大模型")
print("=" * 60)
