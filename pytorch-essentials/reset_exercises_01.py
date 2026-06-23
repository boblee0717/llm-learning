"""
重置 PyTorch 专项 第 1 课自写练习脚本。

用途：
  python reset_exercises_01.py

效果：
  - 将 01_tensor_basics_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path

TARGET_FILE = Path(__file__).with_name("01_tensor_basics_self_write.py")


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
        "def outer_via_broadcast(a, b):",
        "_a = torch.arange(1, 4",
        "def outer_via_broadcast(a, b):\n"
        "    # TODO-1: 用广播返回 (m, n) 的外积\n"
        "    return None\n"
        "\n\n",
        "TODO-1 outer_via_broadcast",
    ),
    (
        None,
        "def split_heads(x, n_head):",
        "_x = torch.randn(2, 5, 12)",
        "def split_heads(x, n_head):\n"
        "    # TODO-2: 返回 (B, n_head, T, d_k)\n"
        "    return None\n"
        "\n\n",
        "TODO-2 split_heads",
    ),
    (
        None,
        "def merge_heads(x):",
        "_merged = merge_heads(_heads)",
        "def merge_heads(x):\n"
        "    # TODO-3: 返回 (B, T, C)，记得 transpose 后要 contiguous 再 view\n"
        "    return None\n"
        "\n\n",
        "TODO-3 merge_heads",
    ),
    (
        None,
        "def flatten_contiguous(x):",
        "_t = torch.arange(6)",
        "def flatten_contiguous(x):\n"
        "    # TODO-4: 返回展平后的一维连续张量\n"
        "    return None\n"
        "\n\n",
        "TODO-4 flatten_contiguous",
    ),
    (
        None,
        "def detach_to_numpy(t):",
        "_g = torch.randn(3, requires_grad=True)",
        "def detach_to_numpy(t):\n"
        "    # TODO-5: 返回 numpy 数组\n"
        "    return None\n"
        "\n\n",
        "TODO-5 detach_to_numpy",
    ),
    (
        None,
        "def add_in_place(t, value):",
        "_orig = torch.ones(4)",
        "def add_in_place(t, value):\n"
        "    # TODO-6: 就地给 t 加上 value 并返回 t 本身\n"
        "    return None\n"
        "\n\n",
        "TODO-6 add_in_place",
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
