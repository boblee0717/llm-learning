"""
重置 PyTorch 专项 第 8 课自写练习脚本。

用途：
  python reset_exercises_08.py

效果：
  - 将 08_capstone_train_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path

TARGET_FILE = Path(__file__).with_name("08_capstone_train_self_write.py")


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
        "def build_param_groups(model, weight_decay=0.1):",
        "_pg_model = MiniLM(VOCAB_SIZE, N_EMBD)",
        "def build_param_groups(model, weight_decay=0.1):\n"
        '    # TODO-1: 返回 [{"params":..., "weight_decay":weight_decay}, {"params":..., "weight_decay":0.0}]\n'
        "    return None\n"
        "\n\n",
        "TODO-1 build_param_groups",
    ),
    (
        None,
        "def lm_loss(logits, y):",
        "_lg = torch.randn(2, 4, VOCAB_SIZE)",
        "def lm_loss(logits, y):\n"
        "    # TODO-2: 摊平后返回 F.cross_entropy(...)\n"
        "    return None\n"
        "\n\n",
        "TODO-2 lm_loss",
    ),
    (
        None,
        "def evaluate(model, loader):",
        "_ev_model = MiniLM(VOCAB_SIZE, N_EMBD)",
        "def evaluate(model, loader):\n"
        "    # TODO-3: 返回平均 val loss（float），评估后恢复 train 状态\n"
        "    return None\n"
        "\n\n",
        "TODO-3 evaluate",
    ),
    (
        None,
        "def accumulate_and_step(model, micro_batches, opt, accum_steps, max_norm=1.0):",
        "_as_model = MiniLM(VOCAB_SIZE, N_EMBD)",
        "def accumulate_and_step(model, micro_batches, opt, accum_steps, max_norm=1.0):\n"
        "    # TODO-4: 梯度累积 + 裁剪 + step，返回平均 loss（float）\n"
        "    return None\n"
        "\n\n",
        "TODO-4 accumulate_and_step",
    ),
    (
        None,
        "def fit(model, train_loader, val_loader, total_steps, accum_steps=2):",
        "torch.manual_seed(0)",
        "def fit(model, train_loader, val_loader, total_steps, accum_steps=2):\n"
        "    # TODO-5: 完整训练循环，返回最优 val loss（float）\n"
        "    return None\n"
        "\n\n",
        "TODO-5 fit",
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
