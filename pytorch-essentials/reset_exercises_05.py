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
        "class LMDataset(Dataset):",
        "    def __len__(self):",
        "    def __getitem__(self, i):",
        "    def __len__(self):\n"
        "        # TODO-1: 返回可取的样本数\n"
        "        return None\n"
        "\n",
        "TODO-1 LMDataset.__len__",
    ),
    (
        "class LMDataset(Dataset):",
        "    def __getitem__(self, i):",
        "_data = torch.arange(100, dtype=torch.long)",
        "    def __getitem__(self, i):\n"
        "        # TODO-2: 返回第 i 条样本 (x, y)，y 是 x 右移一位\n"
        "        return None\n"
        "\n\n",
        "TODO-2 LMDataset.__getitem__",
    ),
    (
        None,
        "def pad_collate(batch, pad_value=0):",
        "_seqs = [torch.arange(3)",
        "def pad_collate(batch, pad_value=0):\n"
        "    # TODO-3: 返回 (padded, mask)\n"
        "    return None\n"
        "\n\n",
        "TODO-3 pad_collate",
    ),
    (
        None,
        "def make_loaders(ds, val_ratio, batch_size):",
        "_res2 = make_loaders(_ds",
        "def make_loaders(ds, val_ratio, batch_size):\n"
        "    # TODO-4: 返回 (train_loader, val_loader)\n"
        "    return None\n"
        "\n\n",
        "TODO-4 make_loaders",
    ),
    (
        None,
        "def count_samples(loader):",
        "_cnt = count_samples(_train_loader)",
        "def count_samples(loader):\n"
        "    # TODO-5: 返回遍历一个 epoch 喂过的样本总数\n"
        "    return None\n"
        "\n\n",
        "TODO-5 count_samples",
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
