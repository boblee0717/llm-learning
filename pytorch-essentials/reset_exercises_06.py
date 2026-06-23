"""
重置 PyTorch 专项 第 6 课自写练习脚本。

用途：
  python reset_exercises_06.py

效果：
  - 将 06_training_loop_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path

TARGET_FILE = Path(__file__).with_name("06_training_loop_self_write.py")


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
        "def evaluate(model, xs, ys):",
        "_m = make_model()",
        "def evaluate(model, xs, ys):\n"
        "    # TODO-1: eval() + no_grad 算平均 loss，返回 float，结束恢复 train()\n"
        "    return None\n"
        "\n\n",
        "TODO-1 evaluate",
    ),
    (
        None,
        "def train_step_accum(model, batches, opt, accum_steps):",
        "torch.manual_seed(1)",
        "def train_step_accum(model, batches, opt, accum_steps):\n"
        "    # TODO-2: 梯度累积一步，返回平均 loss\n"
        "    return None\n"
        "\n\n",
        "TODO-2 train_step_accum",
    ),
    (
        None,
        "def clip_and_step(model, opt, max_norm):",
        "torch.manual_seed(2)",
        "def clip_and_step(model, opt, max_norm):\n"
        "    # TODO-3: 裁剪梯度范数后 step，返回裁剪前总范数(float)\n"
        "    return None\n"
        "\n\n",
        "TODO-3 clip_and_step",
    ),
    (
        None,
        "def save_ckpt(path, model, opt, step):",
        "torch.manual_seed(3)",
        "def save_ckpt(path, model, opt, step):\n"
        "    # TODO-4a: 保存 {model, opt, step} 到 path\n"
        "    return None\n"
        "\n\n"
        "def load_ckpt(path, model, opt):\n"
        "    # TODO-4b: 从 path 恢复 model/opt，返回 step(int)\n"
        "    return None\n"
        "\n\n",
        "TODO-4 save_ckpt/load_ckpt",
    ),
    (
        None,
        "    def step(self, val_loss):",
        "_stopper = EarlyStopper(patience=3)",
        "    def step(self, val_loss):\n"
        "        # TODO-5: 更新 best/bad，返回是否应停止(bool)\n"
        "        return None\n"
        "\n\n",
        "TODO-5 EarlyStopper.step",
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
