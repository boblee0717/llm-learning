"""
======================================================
PyTorch 专项 / 第 5 课（自写版）：Dataset / DataLoader
======================================================

用法：
1. 运行：python 05_data_pipeline_self_write.py
2. 按 TODO-1 到 TODO-5 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错

目标：
- 实现语言模型 Dataset 的 __len__ 与 __getitem__（next-token 样本）
- 用 DataLoader 批量取数据
- 手写 collate_fn 把变长序列 pad 成 batch + 生成 mask
- 用 random_split 切 train / val 并各建一个 DataLoader
- 遍历 DataLoader 统计样本数（验证无放回遍历覆盖全集）

对照：本课主课 05_data_pipeline.py，phase2 第 5 课（手写 get_batch）。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import Dataset, DataLoader, random_split


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
        raise ValidationError(f"{name} 数值不对\nactual=\n{actual}\nexpected=\n{expected}")


# ============================================================
section("TODO-1 / TODO-2：语言模型 Dataset")
# ============================================================
# 实现一个 map-style Dataset：
#   __len__   ：可取的样本数 = len(data) - context_len
#   __getitem__：第 i 条样本 x = data[i:i+context_len]，y = data[i+1:i+context_len+1]
#               （y 是 x 右移一位，即 next-token 目标），返回 (x, y)


class LMDataset(Dataset):
    def __init__(self, data, context_len):
        self.data = data
        self.context_len = context_len

    def __len__(self):
        # TODO-1: 返回可取的样本数
        return None

    def __getitem__(self, i):
        # TODO-2: 返回第 i 条样本 (x, y)，y 是 x 右移一位
        return None


_data = torch.arange(100, dtype=torch.long)
_ds = LMDataset(_data, context_len=8)
require_true("TODO-1 数据集长度", len(_ds) == 92, f"应为 92（100-8），实际 {len(_ds)}")
_item = _ds[0]
require_not_none("TODO-2 __getitem__", _item)
require_true("TODO-2 返回 (x, y)", isinstance(_item, tuple) and len(_item) == 2, "应 return x, y")
_x0, _y0 = _item
require_shape("TODO-2 x 形状", _x0, (8,))
require_close("TODO-2 y = x 右移一位", _y0.float(), (_x0 + 1).float())
print("LMDataset OK：len=92，第 0 条 y 恰好是 x 右移一位")


# ============================================================
section("TODO-3：变长序列 collate_fn pad_collate")
# ============================================================
# 把一个 batch（list of 1D tensor，长度不一）pad 到本 batch 最长长度，
# 返回 (padded, mask)：
#   padded: (batch, max_len)，短序列右侧补 pad_value
#   mask  : (batch, max_len)，真实 token 处为 1，padding 处为 0
#
# 提示：
#   max_len = max(len(seq) for seq in batch)
#   每条 seq：右侧 cat 上 (max_len-len(seq)) 个 pad_value；mask 同理 cat 1...1,0...0
#   最后 torch.stack 成 (batch, max_len)


def pad_collate(batch, pad_value=0):
    # TODO-3: 返回 (padded, mask)
    return None


_seqs = [torch.arange(3), torch.arange(5), torch.arange(2)]
_res = pad_collate(_seqs, pad_value=0)
require_not_none("TODO-3 pad_collate", _res)
_padded, _mask = _res
require_shape("TODO-3 padded 形状", _padded, (3, 5))   # 最长是 5
require_shape("TODO-3 mask 形状", _mask, (3, 5))
require_close("TODO-3 mask 正确",
              _mask, torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]).float())
require_close("TODO-3 第一条 pad 内容", _padded[0].float(),
              torch.tensor([0, 1, 2, 0, 0]).float())
print("pad_collate OK：变长序列 pad 到 5，mask 标记真实 token")


# ============================================================
section("TODO-4：切 train / val 并建 DataLoader make_loaders")
# ============================================================
# 用 random_split 把 ds 按 val_ratio 切成 train / val，各建一个 DataLoader，
# 返回 (train_loader, val_loader)。
#   - 训练集 shuffle=True，验证集 shuffle=False
#   - num_workers=0（Windows 友好）
#   - 用 generator=torch.Generator().manual_seed(42) 让切分可复现
#
# 提示：
#   n_val = int(len(ds) * val_ratio); n_train = len(ds) - n_val
#   train_ds, val_ds = random_split(ds, [n_train, n_val], generator=...)
#   再各 DataLoader(...)


def make_loaders(ds, val_ratio, batch_size):
    # TODO-4: 返回 (train_loader, val_loader)
    return None


_res2 = make_loaders(_ds, val_ratio=0.1, batch_size=8)
require_not_none("TODO-4 make_loaders", _res2)
_train_loader, _val_loader = _res2
require_true("TODO-4 训练集大小", len(_train_loader.dataset) == 83,
             f"应为 83（92 - int(92*0.1)=92-9）, 实际 {len(_train_loader.dataset)}")
require_true("TODO-4 验证集大小", len(_val_loader.dataset) == 9,
             f"应为 9，实际 {len(_val_loader.dataset)}")
require_true("TODO-4 验证集不打乱",
             isinstance(_val_loader.sampler, torch.utils.data.SequentialSampler),
             "验证集应 shuffle=False（SequentialSampler）")
print(f"make_loaders OK：训练 {len(_train_loader.dataset)} / 验证 {len(_val_loader.dataset)}")


# ============================================================
section("TODO-5：遍历 DataLoader 统计样本数 count_samples")
# ============================================================
# 遍历整个 loader，累加每个 batch 的样本数（batch 的第 0 维），返回总数。
# 用来验证「无放回遍历」恰好覆盖整个数据集（不重不漏）。
#
# 提示：for xb, yb in loader: total += xb.shape[0]


def count_samples(loader):
    # TODO-5: 返回遍历一个 epoch 喂过的样本总数
    return None


_cnt = count_samples(_train_loader)
require_not_none("TODO-5 count_samples", _cnt)
require_true("TODO-5 覆盖整个训练集", _cnt == len(_train_loader.dataset),
             f"一个 epoch 应覆盖全部 {len(_train_loader.dataset)} 条，实际 {_cnt}")
print(f"count_samples OK：一个 epoch 恰好喂了 {_cnt} 条（无放回，不重不漏）")


# ============================================================
section("全部 TODO 校验通过 ✓")
# ============================================================
print("""
你已经手写完成：
  1. Dataset.__len__ / __getitem__（next-token 样本）
  2. DataLoader 取 batch
  3. collate_fn 把变长序列 pad 成 batch + mask
  4. random_split 切 train/val 并建两个 DataLoader（验证集不 shuffle）
  5. 遍历 DataLoader 验证无放回遍历覆盖全集

复盘三问：
  * Dataset 和 DataLoader 各自负责什么？为什么要拆成两层？
  * 变长序列为什么需要 collate_fn？mask 在训练里怎么用？
  * 手写 get_batch 和 DataLoader 在「采样方式」上有什么本质区别？
""")
