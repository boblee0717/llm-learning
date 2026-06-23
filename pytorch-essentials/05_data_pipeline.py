"""
第 5 课：数据管道 Dataset / DataLoader（torch 版）
===================================================
phase2 第 5 课你手写过 get_batch：从一长串 token 里随机取若干起点，
切出 (batch, block_size) 的 x 和右移一位的 y。那是「数据怎么喂给模型」
的原理。这节课把它换成 PyTorch 的标准管道：Dataset 负责「单个样本怎么取」，
DataLoader 负责「攒成 batch、打乱、并行加载」，collate_fn 负责「变长样本
怎么 pad 成一个矩形 batch」。

核心问题：
- 自定义 Dataset 要实现哪两个方法？__len__ / __getitem__ 各管什么？
- DataLoader 的 batch_size / shuffle / drop_last 分别是什么含义？
- 变长序列（每条样本长度不同）怎么 pad 成一个 batch？collate_fn 干什么？
- train/val 怎么切？random_split 替你做了什么？
- 我手写的 get_batch 和 DataLoader 取出来的 batch 等价吗？DataLoader 多做了什么？

与大模型的关系：
- LLM 预训练/微调的数据喂入全靠这套管道：IterableDataset 流式读语料、
  collate_fn 做 padding + 构造 attention mask、DataLoader 多进程预取。
- 你 phase2 手写的 get_batch 正是 DataLoader 在语言建模场景下做的事。

前置：phase2 第 5 课（手写 get_batch），本专项第 1 课（tensor 形状）
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码

import torch
from torch.utils.data import Dataset, DataLoader, random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 锚定脚本自身目录


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
section("Part 1: 自定义 Dataset —— __len__ / __getitem__")
# ============================================================
# Dataset 是一个「按索引取单个样本」的抽象。你只需实现两个方法：
#   __len__(self)        → 数据集一共有多少个样本（DataLoader 靠它知道边界）
#   __getitem__(self, i) → 返回第 i 个样本（通常是 (x, y) 一对张量）
# 注意：Dataset 自己【不】负责攒 batch、打乱、并行——那是 DataLoader 的活。
#
# 对照 phase2：你手写 get_batch 时是「一次性随机取一批起点」；这里换成
# 「先把所有滑窗样本编好号，DataLoader 再来按号取」。职责拆开了。

# 用一串连续整数当「语料 token」，做语言建模 next-token 预测：
#   样本 i 的 x = data[i : i+block_size]
#   样本 i 的 y = data[i+1 : i+block_size+1]   ← y 是 x 右移一位（预测下一个 token）
data = torch.arange(0, 100)        # 假装是 100 个 token 的语料
block_size = 8                     # 上下文长度（一个样本看多长）


class SeqDataset(Dataset):
    """语言建模滑窗数据集：每个样本是 (x, y)，y 是 x 右移一位。"""

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # 能切出的滑窗起点个数。最后一个样本的 y 需要 +1，所以是 len - block_size。
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + 1 + self.block_size]
        return x, y


ds = SeqDataset(data, block_size)
print(f"数据集长度 len(ds) = {len(ds)}（= 100 - block_size 8）")
x0, y0 = ds[0]
print(f"第 0 个样本  x = {x0.tolist()}")
print(f"第 0 个样本  y = {y0.tolist()}   ← 正是 x 右移一位")
print("→ Dataset 只回答『第 i 个样本是什么』，不管 batch / shuffle。")


# ============================================================
section("Part 2: DataLoader —— batch_size / shuffle / drop_last")
# ============================================================
# DataLoader 把 Dataset 包起来，负责：
#   batch_size  —— 每次吐出多少个样本攒成一个 batch（自动 stack 成 (B, ...)）
#   shuffle     —— True 则每个 epoch 打乱样本顺序（训练集要打乱，验证集不用）
#   drop_last   —— True 则丢掉最后凑不满 batch_size 的尾巴（保证每个 batch 形状一致）
#   num_workers —— 用几个子进程并行加载。Windows 下用 0（见下方说明）。
#
# 默认的「攒 batch」逻辑（default_collate）会把同形状的样本 stack：
#   batch_size 个 (block_size,) 的 x  →  (batch_size, block_size)

batch_size = 4
loader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,     # 92 个样本 / batch 4 = 23 个整 batch，无尾巴可丢；演示语义
    num_workers=0,      # ← Windows + 脚本顶层执行：必须 0，否则多进程会重复执行模块
)

# Windows 多进程坑说明：
#   num_workers>0 时 DataLoader 用 spawn 起子进程，子进程会重新 import 主模块。
#   若 DataLoader 的迭代写在模块顶层（不在 if __name__=="__main__" 里），
#   子进程 import 时会再次执行 → 无限递归 / 报错。
#   解决：要么 num_workers=0（本课做法），要么把训练代码放进
#         if __name__ == "__main__": 守护块里。CPU 小数据用 0 完全够。

xb, yb = next(iter(loader))
print(f"一个 batch：xb.shape = {tuple(xb.shape)}, yb.shape = {tuple(yb.shape)}")
print(f"  → (batch_size={batch_size}, block_size={block_size})，DataLoader 自动 stack 好了")

n_batches = len(loader)
print(f"len(loader) = {n_batches} 个 batch（{len(ds)} 样本 // {batch_size}，drop_last 丢余数）")
print("shuffle=True：每个 epoch 顺序都不同，下面跑两个 epoch 看首个样本的 x[0]：")
for epoch in range(2):
    first_x = next(iter(loader))[0][0]
    print(f"  epoch {epoch}  本 epoch 第一个样本起点 token = {first_x[0].item()}")


# ============================================================
section("Part 3: collate_fn —— 变长序列 padding")
# ============================================================
# 上面所有样本长度都一样（都是 block_size），能直接 stack。但真实语料里
# 每条句子长度不同，没法直接堆成矩形矩阵。collate_fn 就是你自定义的
# 「怎么把一批样本合成一个 batch」的函数——典型操作就是 pad 到本 batch 最长。
#
# 对照语言建模：HuggingFace 的 DataCollator 就是干这个的，pad 之后还会
# 顺手生成 attention_mask（标记哪些位置是真 token、哪些是 pad）。

# 造一批变长样本（长度 3/5/2/4）
var_batch = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5, 6, 7, 8]),
    torch.tensor([9, 10]),
    torch.tensor([11, 12, 13, 14]),
]


def pad_collate(batch, pad_value=0):
    """把一批变长 1D 张量 pad 到本 batch 内最大长度，返回 (padded, lengths)。"""
    lengths = torch.tensor([len(seq) for seq in batch])
    max_len = int(lengths.max())
    padded = torch.full((len(batch), max_len), pad_value, dtype=batch[0].dtype)
    for row, seq in enumerate(batch):
        padded[row, : len(seq)] = seq        # 前面填真值，后面留 pad_value
    return padded, lengths


padded, lengths = pad_collate(var_batch)
print(f"变长样本长度：{[len(s) for s in var_batch]}")
print(f"pad 到 max_len={int(lengths.max())} 后：\n{padded}")
print(f"各样本真实长度 lengths = {lengths.tolist()}（pad 位置可据此造 mask）")
print("注：等价于 nn.utils.rnn.pad_sequence(batch, batch_first=True)，这里手写以见原理。")

# 把 collate_fn 挂到 DataLoader 上（用 list of 变长张量当 dataset 演示）：
var_ds = [torch.arange(1, n + 2) for n in range(1, 6)]   # 长度 2,3,4,5,6 的样本
var_loader = DataLoader(var_ds, batch_size=3, shuffle=False,
                        collate_fn=pad_collate, num_workers=0)
vb, vl = next(iter(var_loader))
print(f"\n挂上 collate_fn 后，DataLoader 吐出的第一个 batch：shape={tuple(vb.shape)}")
print(vb)


# ============================================================
section("Part 4: train / val 切分 —— random_split")
# ============================================================
# 训练前要把数据切成训练集 / 验证集（验证集用来监控泛化、early stop）。
# random_split 替你做的事：按比例随机划分索引，返回两个 Subset（仍是 Dataset，
# 仍能丢给 DataLoader）。给定 generator 种子可复现同一份划分。

val_ratio = 0.2
n_total = len(ds)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val

gen = torch.Generator().manual_seed(42)      # 固定种子 → 划分可复现
train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

print(f"总样本 {n_total} → train {len(train_ds)} / val {len(val_ds)}（val_ratio={val_ratio}）")
print(f"切出来的还是 Dataset（Subset），可直接喂 DataLoader：")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)  # 验证集不打乱、不丢尾
print(f"  train_loader: {len(train_loader)} 个 batch（shuffle=True, drop_last=True）")
print(f"  val_loader  : {len(val_loader)} 个 batch（shuffle=False，验证集要稳定）")


# ============================================================
section("Part 5: 手写 get_batch vs DataLoader —— 对照")
# ============================================================
# phase2 你手写的 get_batch（语言建模随机采样）大致是：
#   ix = torch.randint(len(data) - block_size, (batch_size,))   # 随机起点
#   x  = torch.stack([data[i   : i+block_size]   for i in ix])
#   y  = torch.stack([data[i+1 : i+1+block_size] for i in ix])
# 它「有放回随机采样」一个 batch，永远不遍历完、也没有 epoch 概念。

def manual_get_batch(data, batch_size, block_size, generator=None):
    """手写随机采样一个 batch（对照 phase2 的 get_batch）。"""
    max_start = len(data) - block_size
    ix = torch.randint(max_start, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


g = torch.Generator().manual_seed(0)
mx, my = manual_get_batch(data, batch_size=4, block_size=block_size, generator=g)
print(f"手写 get_batch:  x.shape = {tuple(mx.shape)}, y.shape = {tuple(my.shape)}")
print(f"  和 DataLoader 取出的 batch 形状完全一致：{tuple(xb.shape)}")

# 校验：手写产出的 (x, y) 确实满足 y 是 x 右移一位（语言建模约束）
ok = torch.equal(my[:, :-1], mx[:, 1:])
print(f"  手写 batch 满足 y == x 右移一位：{ok}")

print("""
DataLoader 比手写 get_batch 多替你做了什么：
  1. epoch 语义：shuffle 后【不放回】遍历完整个数据集才算一个 epoch
     （手写 get_batch 是有放回随机采样，没有"走完一遍"的概念）。
  2. drop_last / 自动 stack / 自动转 batch 形状，不用你手 stack。
  3. collate_fn 钩子：变长 padding、造 mask 等都能插进去。
  4. num_workers 多进程预取：加载和训练重叠，喂数据不卡 GPU。
  5. Sampler 可换：分布式训练用 DistributedSampler 自动切分到各卡。
→ 小实验/语言建模随机采样，手写 get_batch 足够轻便；
  工程化训练（有 epoch、变长、多进程、分布式）就该用 DataLoader。
""")


# ============================================================
section("小结")
# ============================================================
print("""
本课关键结论：
  1. Dataset 只管「第 i 个样本是什么」：实现 __len__ 和 __getitem__ 两个方法。
  2. DataLoader 管「攒 batch」：batch_size 决定大小，shuffle 每 epoch 打乱，
     drop_last 丢凑不满的尾巴，num_workers 控制并行（Windows 用 0 或加 __main__ 守护）。
  3. collate_fn 是「怎么把一批样本合成 batch」的钩子；变长序列在这里 pad 到本 batch 最长。
  4. random_split 按比例随机切 train/val，给定 generator 种子可复现划分。
  5. 手写 get_batch = 有放回随机采样一个 batch；DataLoader 多给你 epoch 语义、
     自动 stack、collate 钩子、多进程预取、可换 Sampler。语言建模两者 batch 形状等价。

下一课：训练循环工程化 —— 五步曲 + early stopping + checkpoint + AMP + 梯度累积。
""")
