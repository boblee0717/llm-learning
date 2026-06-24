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
        "def cross_entropy_manual(logits, target):",
        "_logits = torch.tensor(",
        "def cross_entropy_manual(logits, target):\n"
        "    # TODO-1: 返回标量交叉熵损失\n"
        "    return None\n"
        "\n\n",
        "TODO-1 cross_entropy_manual",
    ),
    (
        None,
        "def warmup_cosine_lr(step, warmup_steps, total_steps, min_ratio=0.1):",
        "_warm, _total = 100, 1000",
        "def warmup_cosine_lr(step, warmup_steps, total_steps, min_ratio=0.1):\n"
        "    # TODO-2: 返回该 step 的 lr 倍率（float）\n"
        "    return None\n"
        "\n\n",
        "TODO-2 warmup_cosine_lr",
    ),
    (
        None,
        "def build_param_groups(model, wd):",
        "_net = nn.Sequential(nn.Linear(8, 16)",
        "def build_param_groups(model, wd):\n"
        "    # TODO-3: 返回 [decay_group, no_decay_group]\n"
        "    return None\n"
        "\n\n",
        "TODO-3 build_param_groups",
    ),
    (
        None,
        "def train_one_step(model, x, y, opt, loss_fn):",
        "torch.manual_seed(1)",
        "def train_one_step(model, x, y, opt, loss_fn):\n"
        "    # TODO-4: 执行一次训练 step，返回 loss（float）\n"
        "    return None\n"
        "\n\n",
        "TODO-4 train_one_step",
    ),
    (
        None,
        "def train_loop(model, x, y, base_lr, warmup, total, loss_fn):",
        "torch.manual_seed(2)",
        "def train_loop(model, x, y, base_lr, warmup, total, loss_fn):\n"
        "    # TODO-5: 跑 total 步训练，返回最终 loss（float）\n"
        "    return None\n"
        "\n\n",
        "TODO-5 train_loop",
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
