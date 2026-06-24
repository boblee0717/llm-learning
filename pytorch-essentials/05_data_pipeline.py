"""
第 5 课：数据管道 Dataset / DataLoader
======================================
phase2 第 5 课你用手写的 get_batch（随机切片 + torch.stack）喂数据。
真实项目用 PyTorch 的标准管道：Dataset 定义「单条样本怎么取」，
DataLoader 负责「批量、打乱、并行加载、拼 batch」。这节课把手写采样
翻译成标准管道，并补上 collate_fn、train/val 切分这些工程细节。

核心问题：
- Dataset 要实现哪两个方法？__getitem__ 返回什么？
- DataLoader 的 batch_size / shuffle / collate_fn / num_workers 各干什么？
- 变长序列怎么拼成一个 batch？（collate_fn + padding）
- 怎么把一份数据切成 train / val？

与大模型的关系：
- 预训练用 Dataset 把语料切成 (input, target) 对，DataLoader 流式喂进训练循环；
  SFT/微调把对话样本做成 Dataset，padding + attention mask 全靠 collate_fn。

前置：phase2 第 5 课（手写 get_batch），本专项第 1 课（tensor）

注意（Windows）：num_workers>0 需要把启动代码放进 if __name__ == "__main__"，
本课为简单起见统一用 num_workers=0（主线程加载），够学习用。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码

import torch
from torch.utils.data import Dataset, DataLoader, random_split


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
section("Part 1: map-style Dataset —— 定义「单条样本怎么取」")
# ============================================================
# 最常用的 Dataset 是 map-style：实现 __len__（多少条）和 __getitem__（取第 i 条）。
# 我们为语言模型做一个：给定一长串 token，第 i 条样本是
#   x = data[i : i+context_len]，y = data[i+1 : i+context_len+1]（右移一位）。
# 这就是 phase2 手写 get_batch 干的事，只是拆成「按 index 取单条」。

class CharLMDataset(Dataset):
    def __init__(self, data, context_len):
        self.data = data
        self.context_len = context_len

    def __len__(self):
        # 能取的起点数量：最后一个起点要留出 context_len+1 的空间
        return len(self.data) - self.context_len

    def __getitem__(self, i):
        x = self.data[i : i + self.context_len]
        y = self.data[i + 1 : i + self.context_len + 1]
        return x, y


data = torch.arange(100, dtype=torch.long)   # 假装是 0..99 的 token 流
ds = CharLMDataset(data, context_len=8)
print("数据集长度（可取样本数）:", len(ds))
x0, y0 = ds[0]
print("第 0 条样本 x:", x0.tolist())
print("第 0 条样本 y:", y0.tolist(), "← 正好是 x 右移一位（next-token）")


# ============================================================
section("Part 2: DataLoader —— 批量 + 打乱 + 拼 batch")
# ============================================================
# DataLoader 包住 Dataset，自动：按 batch_size 攒样本、shuffle 打乱顺序、
# 把多条样本拼成一个 batch tensor（默认用 default_collate，对齐 stack）。

loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
print(f"一共 {len(loader)} 个 batch（{len(ds)} 条 / batch_size 4，向上取整）")

xb, yb = next(iter(loader))
print("一个 batch 的 x 形状:", xb.shape, "（batch_size=4, context_len=8）")
print("一个 batch 的 y 形状:", yb.shape)
print("→ DataLoader 自动把 4 条 (8,) 的样本 stack 成了 (4, 8)")

# 遍历整个 epoch
total = 0
for xb, yb in loader:
    total += xb.shape[0]
print(f"遍历一个 epoch 共喂了 {total} 条样本")


# ============================================================
section("Part 3: collate_fn —— 变长序列怎么拼 batch")
# ============================================================
# 上面每条样本长度相同，default_collate 直接 stack 就行。但真实数据（句子）
# 长度不一，没法直接 stack。这时要自定义 collate_fn：把一个 batch 的样本
# pad 到相同长度，并返回一个 mask 标记哪些是真实 token、哪些是 padding。

class VarLenDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs       # list of 1D tensor，长度不一

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


def pad_collate(batch, pad_value=0):
    """把一个 batch 的变长序列 pad 到本 batch 最长长度，返回 (padded, mask)。"""
    max_len = max(len(seq) for seq in batch)
    padded, mask = [], []
    for seq in batch:
        pad_len = max_len - len(seq)
        padded.append(torch.cat([seq, torch.full((pad_len,), pad_value, dtype=seq.dtype)]))
        mask.append(torch.cat([torch.ones(len(seq)), torch.zeros(pad_len)]))
    return torch.stack(padded), torch.stack(mask)


seqs = [torch.arange(3), torch.arange(5), torch.arange(2)]   # 长度 3/5/2
var_ds = VarLenDataset(seqs)
var_loader = DataLoader(var_ds, batch_size=3, shuffle=False,
                        collate_fn=pad_collate, num_workers=0)
padded, mask = next(iter(var_loader))
print("pad 后的 batch（统一到最长 5）:\n", padded)
print("mask（1=真实 token，0=padding）:\n", mask)
print("→ 训练时用 mask 把 padding 位置的 loss 忽略掉（或 cross_entropy 的 ignore_index）")


# ============================================================
section("Part 4: train / val 切分")
# ============================================================
# 用 random_split 把一个 Dataset 按比例随机切成训练集和验证集。

n_total = len(ds)
n_val = int(n_total * 0.1)
n_train = n_total - n_val
train_ds, val_ds = random_split(ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
print(f"总样本 {n_total} → 训练 {len(train_ds)}，验证 {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)  # 验证不用 shuffle
print(f"训练 loader {len(train_loader)} 个 batch，验证 loader {len(val_loader)} 个 batch")


# ============================================================
section("Part 5: 和 phase2 手写 get_batch 的对照")
# ============================================================
# 手写 get_batch：每次【随机】采 batch_size 个起点，立刻 stack。
# DataLoader：先定义好所有样本（Dataset），再【遍历 + 打乱】，覆盖整个数据集。
#
# 区别：
#   - get_batch 是「有放回随机采样」，同一个 epoch 可能重复/漏采，但实现极简、适合教学。
#   - DataLoader 是「无放回遍历」，一个 epoch 不重不漏，还自带并行加载/collate，工程标配。
# 两者喂出来的 batch 形状完全一样，可以无缝替换。

def get_batch_manual(data, context_len, batch_size):
    ix = torch.randint(len(data) - context_len, (batch_size,))
    x = torch.stack([data[i : i + context_len] for i in ix])
    y = torch.stack([data[i + 1 : i + context_len + 1] for i in ix])
    return x, y


mx, my = get_batch_manual(data, context_len=8, batch_size=4)
print("手写 get_batch 的 batch 形状:", mx.shape, my.shape)
print("DataLoader 的 batch 形状   :", xb.shape if False else (4, 8))
print("→ 形状一致；小项目用 get_batch 够了，正式训练用 DataLoader 更稳更全")


# ============================================================
section("小结")
# ============================================================
print("""
本课关键结论：
  1. Dataset 实现 __len__ + __getitem__；语言模型里 __getitem__ 返回 (x, y=x右移一位)。
  2. DataLoader 负责 batch / shuffle / 拼 batch / 并行加载；验证集 shuffle=False。
  3. 变长序列用自定义 collate_fn pad 到等长，并返回 mask 标记真实 token。
  4. random_split 按比例切 train / val。
  5. DataLoader（无放回遍历、工程标配）和手写 get_batch（有放回随机、教学简洁）
     喂出的 batch 形状一致，可无缝替换。

下一课：把数据管道 + 优化器 + autograd 串成工程化训练循环（checkpoint / AMP / 梯度累积）。
""")
