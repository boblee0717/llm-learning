"""
第 8 课：毕业项目 —— 把 7 课零件拼成一个 phase3 风格训练脚本 🎓
================================================================
前 7 课你把 PyTorch 拆成了零件逐个吃透：tensor、autograd、Module、
损失/优化器、DataLoader、训练循环、调试/显存。但「会每个零件」和
「能把它们组装成一个能跑的训练脚本」是两回事——phase3 第 1 课一上来
就默认你会后者。这节课就把所有零件按真实工程顺序串成【一条连续的训练流】，
训练一个迷你 token 语言模型，作为进入 phase3 前的毕业验收。

这节课没有新知识点，全是「整合」。你会看到前 7 课的每个零件各就各位：
  · 第 5 课 SeqDataset + DataLoader + random_split   → 数据管道
  · 第 3 课 nn.Module + nn.Embedding（LLM 输入第一层）→ 模型
  · 第 4 课 AdamW + param groups + warmup/cosine      → 优化器与调度
  · 第 6 课 五步曲 + 梯度累积 + 裁剪 + eval + early stop + checkpoint → 训练循环
  · 第 7 课 参数量 / 显存 / FLOPs 估算                → 训练前的「能不能训」体检
  · 第 2 课 autograd（loss.backward 背后）、第 1 课 tensor 形状（(B,T,C)）贯穿全程

核心问题：
- 一个真实训练脚本的代码顺序是怎样的？（体检 → 数据 → 模型 → 优化器 → 循环 → 存盘）
- token id (B,T) 怎么经 nn.Embedding 变成 (B,T,C) 喂进模型？（LLM 的输入第一层）
- 语言建模的 loss 怎么算？为什么要把 (B,T,V) 摊平成 (B*T,V) 再喂 cross_entropy？

与大模型的关系：
- 这就是 nanoGPT / 任何 LLM 训练脚本的最小骨架。phase3 在此之上换更大的模型、
  更多的数据、多卡与 AMP，骨架不变。能独立写出这一版，你就接得住 phase3。

前置：本专项第 1~7 课全部
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 产物锚定脚本自身目录


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# 全局复现性（第 6 课）：固定种子 → 同样的代码同样的结果
torch.manual_seed(1337)

# 任务设定：一个可学的 next-token 玩具语言建模任务。
#   语料 = 0,1,2,...,vocab-1,0,1,2,... 的循环序列（next token = (cur+1) % vocab）。
#   规律确定、模型必然能学会 → loss 会从 ln(vocab) 一路降到接近 0，训练曲线干净好看。
VOCAB_SIZE = 16
BLOCK_SIZE = 8
N_EMBD = 32


# ============================================================
section("Part 1: 训练前体检 —— 模型多大、要多少显存/算力（第 7 课）")
# ============================================================
# 真实工程的第一步不是写循环，而是「这模型能不能训」。先把账算清楚。
# 这里把模型先建出来，量一下规模，再决定 batch / 步数。

class MiniLM(nn.Module):
    """迷你 token 语言模型：Embedding（输入第一层）+ MLP + LM head。
    nn.Embedding 是 LLM 的输入第一层：把 token id (B,T) 查表成向量 (B,T,C)。
    它本质是一张可训练的查找表，weight 形状 = (vocab_size, n_embd)。"""

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)   # (vocab, C) 查找表
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd), nn.Tanh(),
            nn.Linear(n_embd, n_embd), nn.Tanh(),
        )
        self.head = nn.Linear(n_embd, vocab_size)       # 投影回词表，输出每个位置的 logits

    def forward(self, idx):
        # idx: (B, T) 的 token id（long）
        x = self.embed(idx)        # (B, T) → (B, T, C)   ← 查表，token id 变向量
        x = self.mlp(x)            # (B, T, C) → (B, T, C)
        logits = self.head(x)      # (B, T, C) → (B, T, vocab)  每个位置预测下一个 token
        return logits


model = MiniLM(VOCAB_SIZE, N_EMBD)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量：total={total:,}  trainable={trainable:,}")

# 显存估算（第 7 课：训练态每参数约 16 字节 = fp32 权重4+梯度4+Adam m4+v4）
mem_gb = total * 16 / 1e9
print(f"训练态显存粗估 ≈ {mem_gb * 1024:.3f} MB（每参数 16 字节，未含激活）")
print("  → 这种玩具规模 CPU 随便跑；换成 7B 就是 ~112GB，那才需要多卡/ZeRO/量化。")

# 算力估算（第 7 课：C ≈ 6ND）
planned_tokens = 50_000
flops = 6 * total * planned_tokens
print(f"算力粗估：训练 {planned_tokens:,} tokens ≈ {flops:.2e} FLOPs（C≈6ND）")
print("→ 体检通过，开始搭数据管道。\n")


# ============================================================
section("Part 2: 数据管道 —— SeqDataset + random_split + DataLoader（第 5 课）")
# ============================================================
# 第 5 课你写过 SeqDataset（滑窗 next-token）和 DataLoader。这里直接用上：
# 把循环语料切成训练/验证两份，各包一个 DataLoader。

class SeqDataset(Dataset):
    """语言建模滑窗数据集：样本 i 的 x=data[i:i+B]，y=x 右移一位。"""

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + 1 + self.block_size]
        return x, y


data = torch.arange(0, 1024) % VOCAB_SIZE        # 循环 token 语料（next=(cur+1)%vocab）
full_ds = SeqDataset(data, BLOCK_SIZE)

gen = torch.Generator().manual_seed(42)          # 固定划分（第 5/6 课：复现性）
n_val = int(len(full_ds) * 0.2)
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

BATCH_SIZE = 16
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"语料 {len(data)} tokens → 样本 {len(full_ds)}（train {n_train} / val {n_val}）")
print(f"train_loader {len(train_loader)} 个 batch，val_loader {len(val_loader)} 个 batch")
print("（真实工程把 num_workers 调大并预取；本课 CPU 小数据用 0，见第 5 课多进程说明）\n")


# ============================================================
section("Part 3: 优化器 + 调度 —— AdamW + param groups + warmup/cosine（第 4 课）")
# ============================================================
# 第 4 课的标准件全用上：矩阵权重做 weight decay、bias/LayerNorm/Embedding 不做；
# 学习率走 warmup（稳住初期）+ cosine（后期精调）。

def build_param_groups(model, weight_decay=0.1):
    """ndim>=2 的权重做 weight decay，其余（bias 等一维参数）不做。"""
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_lambda(step, warmup, total, min_ratio=0.1):
    if step < warmup:                                  # 线性 warmup
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))  # cosine 衰减到 min_ratio
    return min_ratio + (1 - min_ratio) * cosine


TOTAL_STEPS = 600
WARMUP_STEPS = 60
opt = torch.optim.AdamW(build_param_groups(model), lr=3e-3)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lr_lambda=lambda s: lr_lambda(s, WARMUP_STEPS, TOTAL_STEPS))
print(f"AdamW：decay 组 {len(opt.param_groups[0]['params'])} 个张量、"
      f"no-decay 组 {len(opt.param_groups[1]['params'])} 个")
print(f"调度：warmup {WARMUP_STEPS} 步 + cosine 衰减到 {TOTAL_STEPS} 步\n")


# ============================================================
section("Part 4: 训练循环 —— 五步曲 + 累积 + 裁剪 + eval + early stop（第 6 课）")
# ============================================================
# 把第 6 课的零件按真实顺序串起来，跑一条连续训练流（不再是各自独立的 demo）。

ACCUM_STEPS = 2          # 梯度累积：2 个 micro-batch 攒成 1 个有效大 batch
MAX_NORM = 1.0           # 梯度裁剪阈值（LLM 常用 1.0）
EVAL_EVERY = 100         # 每隔多少步评估一次 val
PATIENCE = 4             # early stopping 容忍次数


def lm_loss(logits, y):
    """语言建模 loss：把 (B,T,V) 摊平成 (B*T,V)、y 摊平成 (B*T,) 再喂 cross_entropy。"""
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), y.reshape(B * T))


@torch.no_grad()
def evaluate(model, loader):
    was_training = model.training
    model.eval()
    losses = [lm_loss(model(xb), yb).item() for xb, yb in loader]
    if was_training:
        model.train()
    return sum(losses) / len(losses)


class EarlyStopper:
    def __init__(self, patience=4, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad = float("inf"), 0

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best, self.bad = val_loss, 0
            return False, True          # (是否该停, 是否刷新了最优)
        self.bad += 1
        return self.bad >= self.patience, False


ckpt_path = os.path.join(SCRIPT_DIR, "_capstone_ckpt.pt")
stopper = EarlyStopper(patience=PATIENCE)
print(f"开训前 val loss = {evaluate(model, val_loader):.4f}（≈ ln({VOCAB_SIZE})={math.log(VOCAB_SIZE):.3f} 随机基线）")

model.train()
step = 0
train_iter = iter(train_loader)
stop = False
while step < TOTAL_STEPS and not stop:
    # --- 梯度累积：攒 ACCUM_STEPS 个 micro-batch 再 step 一次 ---
    opt.zero_grad()
    running = 0.0
    for _ in range(ACCUM_STEPS):
        try:
            xb, yb = next(train_iter)
        except StopIteration:                  # 一个 epoch 跑完，重起迭代器
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        loss = lm_loss(model(xb), yb) / ACCUM_STEPS   # 除以 accum（第 6 课的关键坑）
        loss.backward()
        running += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)   # 裁剪防爆炸
    opt.step()
    sched.step()
    step += 1

    # --- 定期评估 + early stopping + 保存最优 checkpoint ---
    if step % EVAL_EVERY == 0:
        val = evaluate(model, val_loader)
        stop, improved = stopper.step(val)
        if improved:                          # 刷新最优 → 存盘（断点续训用，第 6 课）
            torch.save({"step": step, "model": model.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "best_val": stopper.best}, ckpt_path)
        tag = "  ← 保存最优" if improved else ("  ← early stop!" if stop else "")
        print(f"  step {step:>3d}  train={running:.4f}  val={val:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}{tag}")

print(f"\n训练结束于 step {step}，最优 val loss = {stopper.best:.4f}"
      f"（从 ~{math.log(VOCAB_SIZE):.2f} 降下来，模型学会了 next-token）")


# ============================================================
section("Part 5: 断点续训 —— 从最优 checkpoint 恢复（第 6 课）")
# ============================================================
# 模拟「断电重启」：新建一套全新对象，从盘里恢复 model/opt/sched/step，验证续得上。

model2 = MiniLM(VOCAB_SIZE, N_EMBD)
opt2 = torch.optim.AdamW(build_param_groups(model2), lr=3e-3)
sched2 = torch.optim.lr_scheduler.LambdaLR(
    opt2, lr_lambda=lambda s: lr_lambda(s, WARMUP_STEPS, TOTAL_STEPS))

ckpt = torch.load(ckpt_path)                  # GPU 时传 map_location
model2.load_state_dict(ckpt["model"])
opt2.load_state_dict(ckpt["opt"])
sched2.load_state_dict(ckpt["sched"])
print(f"从 step {ckpt['step']} 恢复，载入后 val loss = {evaluate(model2, val_loader):.4f}"
      f"（与存盘时的最优 {ckpt['best_val']:.4f} 一致 → 续训不丢进度）")
os.remove(ckpt_path)
print("已删除 checkpoint 文件，目录保持干净\n")


# ============================================================
section("Part 6: 验收 —— 让模型生成一段，看它真学会了没")
# ============================================================
# 语言模型的终极检验：给个起始 token，自回归地一个个往后预测，看是否符合
# 「next = (cur+1) % vocab」的规律。这就是 phase3 推理课要深入的「生成」雏形。

@torch.no_grad()
def generate(model, start_token, n_new):
    model.eval()
    seq = [start_token]
    cur = torch.tensor([[start_token]])           # (1, 1)
    for _ in range(n_new):
        logits = model(cur)                       # (1, t, V)
        nxt = int(logits[0, -1].argmax())         # 取最后一个位置 argmax 当下一个 token
        seq.append(nxt)
        cur = torch.tensor([seq])                 # 追加后再喂（玩具实现，未用 KV cache）
    return seq


gen_seq = generate(model, start_token=0, n_new=15)
expected = [(i) % VOCAB_SIZE for i in range(16)]
print(f"从 token 0 生成：{gen_seq}")
print(f"理论正确序列    ：{expected}")
print(f"完全一致：{gen_seq == expected} → 模型学会了 next-token 规律 ✓")


# ============================================================
section("小结 —— 你毕业了 🎓")
# ============================================================
print("""
本课关键结论：
  1. 真实训练脚本的顺序：体检(参数/显存/算力) → 数据管道 → 模型 → 优化器+调度
     → 训练循环(五步曲+累积+裁剪+定期eval+early stop+存最优) → 续训 → 验收生成。
  2. nn.Embedding 是 LLM 输入第一层：token id (B,T) 查表成 (B,T,C)，本质是可训练查找表。
  3. 语言建模 loss：logits (B,T,V) 摊平成 (B*T,V)、target (B,T) 摊平成 (B*T,) 喂 cross_entropy。
  4. 前 7 课的每个零件在这里各就各位，骨架与 nanoGPT/任何 LLM 训练脚本一致。
  5. checkpoint 存 model+opt+sched+step，断电能续；early stopping 取泛化最好的点。

你已经能独立写出工程化训练脚本，正式具备进入 phase3 的能力：
  → phase3：在更大的模型 / 更多数据 / 多卡 / AMP 上，把这套骨架放大。
     训练 → LoRA 微调 → 量化 → RLHF → 推理优化。出发！
""")
