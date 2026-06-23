"""
重置 PyTorch 专项 第 2 课自写练习脚本。

用途：
  python reset_exercises_02.py

效果：
  - 将 02_autograd_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path

TARGET_FILE = Path(__file__).with_name("02_autograd_self_write.py")


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
        "def autograd_grad(W, x, target):",
        "_W = torch.randn(3, 4, requires_grad=True)",
        "def autograd_grad(W, x, target):\n"
        "    # TODO-1: 用 autograd 求 loss 对 W 的梯度并返回（.clone()）\n"
        "    return None\n"
        "\n\n",
        "TODO-1 autograd_grad",
    ),
    (
        None,
        "def analytic_grad(W, x, target):",
        "_g_manual = analytic_grad",
        "def analytic_grad(W, x, target):\n"
        "    # TODO-2: 不用 autograd，返回 loss 对 W 的解析梯度 (out, in)\n"
        "    return None\n"
        "\n\n",
        "TODO-2 analytic_grad",
    ),
    (
        None,
        "def numeric_grad(f, x, eps=1e-4):",
        "def _f(t):",
        "def numeric_grad(f, x, eps=1e-4):\n"
        "    # TODO-3: 返回 f 在 x 处的数值梯度（和 x 同形状）\n"
        "    return None\n"
        "\n\n",
        "TODO-3 numeric_grad",
    ),
    (
        None,
        "def to_constant(x):",
        "_xc = torch.randn(3, requires_grad=True)",
        "def to_constant(x):\n"
        "    # TODO-4: 返回 x 的 detach 版本（不带梯度，共享数据）\n"
        "    return None\n"
        "\n\n",
        "TODO-4 to_constant",
    ),
    (
        None,
        "def train_step(w, b, xs, ys, lr):",
        "torch.manual_seed(1)",
        "def train_step(w, b, xs, ys, lr):\n"
        "    # TODO-5: 执行一次训练 step，返回这一步的 loss（float）\n"
        "    return None\n"
        "\n\n",
        "TODO-5 train_step",
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
