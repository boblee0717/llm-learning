"""
重置 PyTorch 专项 第 5 课自写练习脚本。

用途：
  python reset_exercises_05.py

效果：
  - 将 05_data_pipeline_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path

TARGET_FILE = Path(__file__).with_name("05_data_pipeline_self_write.py")


def replace_block(text, start_marker, end_marker, replacement, label, search_from=None):
    base = 0
    if search_from is not None:
        base = text.find(search_from)
        if base == -1:
            raise RuntimeError(f"重置失败: {label} 找不到范围锚点 {search_from!r}")
    start = text.find(start_marker, base)
    if start == -1:
        raise RuntimeError(f"重置失败: {label} 找不到起点 {start_marker!r}")
    end = text.find(end_marker, start)
    if end == -1:
        raise RuntimeError(f"重置失败: {label} 找不到终点 {end_marker!r}")
    return text[:start] + replacement + text[end:]


BLANK_BLOCKS = [
    (
        None,
        "    def __len__(self):",
        "_data = torch.arange(0, 50)",
        "    def __len__(self):\n"
        "        # TODO-1a: 返回样本总数（= len(data) - block_size）\n"
        "        return None\n"
        "\n"
        "    def __getitem__(self, i):\n"
        "        # TODO-1b: 返回第 i 个样本 (x, y)，y 是 x 右移一位\n"
        "        return None\n"
        "\n\n",
        "TODO-1 SeqDataset",
    ),
    (
        None,
        "def make_loader(dataset, batch_size):",
        "_loader = make_loader(_ds",
        "def make_loader(dataset, batch_size):\n"
        "    # TODO-2: 返回配置好的 DataLoader（shuffle=True, drop_last=True, num_workers=0）\n"
        "    return None\n"
        "\n\n",
        "TODO-2 make_loader",
    ),
    (
        None,
        "def pad_collate(batch, pad_value=0):",
        "_var = [torch.tensor([1, 2, 3])",
        "def pad_collate(batch, pad_value=0):\n"
        "    # TODO-3: pad 到本 batch 最大长度，返回 (padded, lengths)\n"
        "    return None\n"
        "\n\n",
        "TODO-3 pad_collate",
    ),
    (
        None,
        "def split_train_val(dataset, val_ratio, seed):",
        "_res4 = split_train_val(_ds",
        "def split_train_val(dataset, val_ratio, seed):\n"
        "    # TODO-4: 用 random_split 返回 (train, val)，generator 用 seed 固定\n"
        "    return None\n"
        "\n\n",
        "TODO-4 split_train_val",
    ),
    (
        None,
        "def manual_get_batch(data, batch_size, block_size, generator=None):",
        "_g = torch.Generator().manual_seed(0)",
        "def manual_get_batch(data, batch_size, block_size, generator=None):\n"
        "    # TODO-5: 随机采样一个 batch，返回 (x, y)，y 是 x 右移一位\n"
        "    return None\n"
        "\n\n",
        "TODO-5 manual_get_batch",
    ),
]


def main() -> int:
    if not TARGET_FILE.exists():
        print(f"未找到目标文件: {TARGET_FILE}")
        return 1

    text = TARGET_FILE.read_text(encoding="utf-8")
    for search_from, start_marker, end_marker, replacement, label in BLANK_BLOCKS:
        text = replace_block(text, start_marker, end_marker, replacement, label, search_from)
    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
