"""
======================================================
PyTorch 专项 / 第 5 课（自写版）：数据管道 Dataset / DataLoader
======================================================

用法：
1. 运行：python 05_data_pipeline_self_write.py
2. 按 TODO-1 到 TODO-5 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错
   （没填的 TODO 会返回 None，校验提示「未实现」，这是正常的）

目标：
- 写自定义 SeqDataset：__len__ / __getitem__ 返回语言建模 next-token 滑窗样本
- 写 make_loader：配置好的 DataLoader（shuffle / drop_last / num_workers）
- 写 pad_collate：把变长样本 pad 到 batch 内最大长度
- 用 random_split 做 train/val 切分
- 手写 manual_get_batch 随机采样（对照 phase2），验证形状与右移约束

对照：本课主课 05_data_pipeline.py，以及 phase2 第 5 课（手写 get_batch）。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")  # ValidationError 走 stderr，也要防乱码

import torch
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(0)


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


class ValidationError(Exception):
    pass


def require_not_none(name, value):
    if value is None:
        raise ValidationError(f"{name} 未实现：结果是 None。")


def require_true(name, cond, hint=""):
    if not cond:
        raise ValidationError(f"{name} 条件不满足：{hint}")


def require_shape(name, actual, expected_shape):
    require_not_none(name, actual)
    if tuple(actual.shape) != tuple(expected_shape):
        raise ValidationError(
            f"{name} 形状不对：actual={tuple(actual.shape)}, expected={tuple(expected_shape)}"
        )


def require_close(name, actual, expected, atol=1e-5):
    require_not_none(name, actual)
    if not torch.allclose(actual, expected, atol=atol):
        raise ValidationError(
            f"{name} 数值不对\nactual=\n{actual}\nexpected=\n{expected}"
        )


# ============================================================
section("TODO-1：自定义 SeqDataset（__len__ / __getitem__）")
# ============================================================
# 语言建模 next-token 滑窗数据集。给定一长串 token data 和 block_size：
#   样本 i 的 x = data[i : i+block_size]
#   样本 i 的 y = data[i+1 : i+1+block_size]   ← y 是 x 右移一位（预测下一个 token）
# 需要实现两个方法：
#   __len__     → 能切出多少个样本？最后一个样本的 y 要 +1，所以是 len(data) - block_size
#   __getitem__ → 返回第 i 个样本 (x, y)
#
# 提示：x = self.data[i : i+self.block_size]
#       y = self.data[i+1 : i+1+self.block_size]


class SeqDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # TODO-1a: 返回样本总数（= len(data) - block_size）
        return None

    def __getitem__(self, i):
        # TODO-1b: 返回第 i 个样本 (x, y)，y 是 x 右移一位
        return None


_data = torch.arange(0, 50)
_bs = 8
_ds = SeqDataset(_data, _bs)
_n = _ds.__len__()
require_not_none("TODO-1 __len__", _n)
require_true("TODO-1 数据集长度", _n == 50 - _bs,
             f"应为 {50 - _bs}，实际 {_n}")
_x0, _y0 = _ds[0]
require_shape("TODO-1 x 形状", _x0, (_bs,))
require_shape("TODO-1 y 形状", _y0, (_bs,))
require_close("TODO-1 y 是 x 右移一位", _y0[:-1], _x0[1:])
require_close("TODO-1 x 内容", _x0, torch.arange(0, _bs))
print("SeqDataset OK：len =", len(_ds), " x0 =", _x0.tolist())


# ============================================================
section("TODO-2：make_loader（配置好的 DataLoader）")
# ============================================================
# 返回一个配置好的 DataLoader：
#   shuffle=True       —— 训练集每个 epoch 打乱
#   drop_last=True     —— 丢掉凑不满 batch_size 的尾巴（保证 batch 形状一致）
#   num_workers=0      —— Windows + 脚本顶层执行用 0，避免多进程坑
# （Windows 下 num_workers>0 需把训练代码放进 if __name__=="__main__" 守护块。）
#
# 提示：return DataLoader(dataset, batch_size=batch_size,
#                         shuffle=True, drop_last=True, num_workers=0)


def make_loader(dataset, batch_size):
    # TODO-2: 返回配置好的 DataLoader（shuffle=True, drop_last=True, num_workers=0）
    return None


_loader = make_loader(_ds, batch_size=4)
require_not_none("TODO-2 make_loader", _loader)
require_true("TODO-2 返回 DataLoader", isinstance(_loader, DataLoader),
             "应返回 torch.utils.data.DataLoader")
require_true("TODO-2 batch_size", _loader.batch_size == 4, "batch_size 应为 4")
require_true("TODO-2 drop_last", _loader.drop_last is True, "drop_last 应为 True")
require_true("TODO-2 num_workers", _loader.num_workers == 0,
             "num_workers 应为 0（Windows 安全）")
_xb, _yb = next(iter(_loader))
require_shape("TODO-2 batch x 形状", _xb, (4, _bs))
require_true("TODO-2 batch 数量（drop_last 丢余数）",
             len(_loader) == (50 - _bs) // 4,
             f"应为 {(50 - _bs) // 4} 个 batch")
print("make_loader OK：batch xb.shape =", tuple(_xb.shape), " 共", len(_loader), "个 batch")


# ============================================================
section("TODO-3：pad_collate（变长序列 padding）")
# ============================================================
# 把一批【变长】1D 张量 pad 到本 batch 内最大长度，返回 (padded, lengths)：
#   padded  —— (batch, max_len) 张量，每行前面是真值，后面补 pad_value(=0)
#   lengths —— (batch,) 张量，每个样本的真实长度（用来造 mask）
#
# 提示：
#   lengths = torch.tensor([len(seq) for seq in batch])
#   max_len = int(lengths.max())
#   padded  = torch.full((len(batch), max_len), pad_value, dtype=batch[0].dtype)
#   循环把每条 seq 填到 padded[row, :len(seq)]


def pad_collate(batch, pad_value=0):
    # TODO-3: pad 到本 batch 最大长度，返回 (padded, lengths)
    return None


_var = [torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6, 7, 8]),
        torch.tensor([9, 10])]
_res = pad_collate(_var)
require_not_none("TODO-3 pad_collate", _res)
_padded, _lengths = _res
require_shape("TODO-3 padded 形状", _padded, (3, 5))
require_close("TODO-3 lengths", _lengths, torch.tensor([3, 5, 2]))
require_close("TODO-3 第 0 行（pad 后）", _padded[0], torch.tensor([1, 2, 3, 0, 0]))
require_close("TODO-3 第 2 行（pad 后）", _padded[2], torch.tensor([9, 10, 0, 0, 0]))
print("pad_collate OK：\n", _padded, "\nlengths =", _lengths.tolist())


# ============================================================
section("TODO-4：split_train_val（random_split 切分）")
# ============================================================
# 用 random_split 把 dataset 按 val_ratio 切成 (train, val)，返回这两个子集。
# 给定 seed 构造 generator 使划分可复现。
#   n_val   = int(len(dataset) * val_ratio)
#   n_train = len(dataset) - n_val
#
# 提示：
#   gen = torch.Generator().manual_seed(seed)
#   return random_split(dataset, [n_train, n_val], generator=gen)


def split_train_val(dataset, val_ratio, seed):
    # TODO-4: 用 random_split 返回 (train, val)，generator 用 seed 固定
    return None


_res4 = split_train_val(_ds, val_ratio=0.25, seed=42)
require_not_none("TODO-4 split_train_val", _res4)
_train, _val = _res4
require_true("TODO-4 val 数量", len(_val) == int(len(_ds) * 0.25),
             f"应为 {int(len(_ds) * 0.25)}，实际 {len(_val)}")
require_true("TODO-4 train+val=总数", len(_train) + len(_val) == len(_ds),
             "切分后两部分之和应等于原数据集大小")
# 复现性：同 seed 再切一次，val 的第一个样本应一致
_train2, _val2 = split_train_val(_ds, val_ratio=0.25, seed=42)
require_close("TODO-4 同 seed 划分可复现", _val[0][0], _val2[0][0])
print("split_train_val OK：train", len(_train), "/ val", len(_val), "（同 seed 可复现）")


# ============================================================
section("TODO-5：manual_get_batch（手写随机采样，对照 phase2）")
# ============================================================
# 手写随机采样一个 batch（语言建模），对照 phase2 的 get_batch：
#   随机取 batch_size 个起点 ix（范围 0 .. len(data)-block_size）
#   x = stack(data[i : i+block_size])
#   y = stack(data[i+1 : i+1+block_size])   ← y 是 x 右移一位
# 用传入的 generator 让采样可复现。
#
# 提示：
#   max_start = len(data) - block_size
#   ix = torch.randint(max_start, (batch_size,), generator=generator)
#   x  = torch.stack([data[i   : i+block_size]   for i in ix])
#   y  = torch.stack([data[i+1 : i+1+block_size] for i in ix])


def manual_get_batch(data, batch_size, block_size, generator=None):
    # TODO-5: 随机采样一个 batch，返回 (x, y)，y 是 x 右移一位
    return None


_g = torch.Generator().manual_seed(0)
_res5 = manual_get_batch(_data, batch_size=4, block_size=_bs, generator=_g)
require_not_none("TODO-5 manual_get_batch", _res5)
_mx, _my = _res5
require_shape("TODO-5 x 形状", _mx, (4, _bs))
require_shape("TODO-5 y 形状", _my, (4, _bs))
require_close("TODO-5 y 是 x 右移一位", _my[:, :-1], _mx[:, 1:])
require_true("TODO-5 起点在合法范围内", int(_mx[:, 0].max()) <= len(_data) - _bs,
             "采样起点不能超过 len(data) - block_size")
print("manual_get_batch OK：x.shape =", tuple(_mx.shape), " y 是 x 右移一位 ✓")


# ============================================================
section("全部 TODO 校验通过 ✓")
# ============================================================
print("""
你已经手写完成：
  1. SeqDataset：__len__ / __getitem__ 切语言建模 next-token 滑窗样本
  2. make_loader：配置好的 DataLoader（shuffle / drop_last / num_workers=0）
  3. pad_collate：把变长样本 pad 到本 batch 最大长度，并返回真实 lengths
  4. split_train_val：random_split 按比例切 train/val，同 seed 可复现
  5. manual_get_batch：手写随机采样一个 batch（对照 phase2 的 get_batch）

复盘三问：
  * Dataset 的 __len__ 为什么是 len(data) - block_size 而不是 len(data)？
  * collate_fn 在 DataLoader 流程里何时被调用？它的输入是什么、输出又喂给谁？
  * 手写 get_batch 与 DataLoader 取 batch 的本质区别（有放回采样 vs epoch 遍历）是什么？
""")
