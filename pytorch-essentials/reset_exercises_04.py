"""
重置 PyTorch 专项 第 4 课自写练习脚本。

用途：
  python reset_exercises_04.py

效果：
  - 将 04_loss_and_optim_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path

TARGET_FILE = Path(__file__).with_name("04_loss_and_optim_self_write.py")


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
        "def manual_cross_entropy(logits, target):",
        "_logits = torch.tensor([[2.0, 0.5, 0.1]",
        "def manual_cross_entropy(logits, target):\n"
        "    # TODO-1: 用 log_softmax + 按 target 取值 + 取负求平均，返回标量交叉熵\n"
        "    return None\n"
        "\n\n",
        "TODO-1 manual_cross_entropy",
    ),
    (
        None,
        "def train_step(model, x, y, opt):",
        "torch.manual_seed(0)",
        "def train_step(model, x, y, opt):\n"
        "    # TODO-2: 实现训练五步曲，返回 loss.item()\n"
        "    return None\n"
        "\n\n",
        "TODO-2 train_step",
    ),
    (
        None,
        "def build_param_groups(model):",
        "class _TinyNet(nn.Module):",
        "def build_param_groups(model):\n"
        "    # TODO-3: 返回 [decay_group(wd=0.1), no_decay_group(wd=0.0)]\n"
        "    return None\n"
        "\n\n",
        "TODO-3 build_param_groups",
    ),
    (
        None,
        "def lr_lambda(step, warmup, total, min_ratio=0.1):",
        "_warmup, _total = 100, 1000",
        "def lr_lambda(step, warmup, total, min_ratio=0.1):\n"
        "    # TODO-4: warmup 线性 + cosine 衰减，返回倍率（float）\n"
        "    return None\n"
        "\n\n",
        "TODO-4 lr_lambda",
    ),
    (
        None,
        "def fit_one(model, xs, ys, steps):",
        "torch.manual_seed(1)",
        "def fit_one(model, xs, ys, steps):\n"
        "    # TODO-5: AdamW + LambdaLR 跑完整训练，返回最终 loss(float)\n"
        "    return None\n"
        "\n\n",
        "TODO-5 fit_one",
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
