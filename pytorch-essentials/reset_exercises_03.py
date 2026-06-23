"""
重置 PyTorch 专项 第 3 课自写练习脚本。

用途：
  python reset_exercises_03.py

效果：
  - 将 03_nn_module_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path

TARGET_FILE = Path(__file__).with_name("03_nn_module_self_write.py")


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
        "class MyLinear(nn.Module):",
        "    def forward(self, x):",
        "_lin = MyLinear(4, 3)",
        "    def forward(self, x):\n"
        "        # TODO-1: 实现 y = x @ Wᵀ + b\n"
        "        return None\n"
        "\n\n",
        "TODO-1 MyLinear.forward",
    ),
    (
        "class ScratchLayerNorm(nn.Module):",
        "    def forward(self, x):",
        "_ln = ScratchLayerNorm(8)",
        "    def forward(self, x):\n"
        "        # TODO-2: 实现 LayerNorm 前向\n"
        "        return None\n"
        "\n\n",
        "TODO-2 ScratchLayerNorm.forward",
    ),
    (
        "class MaskedModule(nn.Module):",
        '        # TODO-3: 把因果掩码登记为名为 "mask" 的 buffer',
        "    def forward(self, x):",
        '        # TODO-3: 把因果掩码登记为名为 "mask" 的 buffer\n'
        "        #   提示：mask = torch.triu(torch.ones(context_len, context_len), diagonal=1)\n"
        '        #         然后 self.register_buffer("mask", mask)\n'
        "        pass  # ← 实现后删掉这行\n"
        "\n",
        "TODO-3 register_buffer",
    ),
    (
        None,
        "def copy_weights(src, dst):",
        "_src = MyLinear(4, 3)",
        "def copy_weights(src, dst):\n"
        "    # TODO-4: 把 src 的权重复制进 dst，返回 dst\n"
        "    return None\n"
        "\n\n",
        "TODO-4 copy_weights",
    ),
    (
        None,
        "def count_trainable(model):",
        "_net = nn.Sequential(",
        "def count_trainable(model):\n"
        "    # TODO-5: 返回可训练参数总数（int）\n"
        "    return None\n"
        "\n\n",
        "TODO-5 count_trainable",
    ),
    (
        None,
        "def freeze_all(module):",
        "_freeze_net = nn.Linear(4, 3)",
        "def freeze_all(module):\n"
        "    # TODO-6: 冻结所有参数，返回被冻结的参数总数\n"
        "    return None\n"
        "\n\n",
        "TODO-6 freeze_all",
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
