"""
重置 PyTorch 专项 第 7 课自写练习脚本。

用途：
  python reset_exercises_07.py

效果：
  - 将 07_debug_profile_memory_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path

TARGET_FILE = Path(__file__).with_name("07_debug_profile_memory_self_write.py")


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
        "def count_params(model):",
        "_m = nn.Sequential(",
        "def count_params(model):\n"
        "    # TODO-1: 返回 (total, trainable)\n"
        "    return None\n"
        "\n\n",
        "TODO-1 count_params",
    ),
    (
        None,
        "def estimate_memory_gb(num_params, bytes_per_param=16):",
        "_gb = estimate_memory_gb(",
        "def estimate_memory_gb(num_params, bytes_per_param=16):\n"
        "    # TODO-2: 返回训练显存估算（GB），用 /1e9 口径\n"
        "    return None\n"
        "\n\n",
        "TODO-2 estimate_memory_gb",
    ),
    (
        None,
        "def estimate_flops(num_params, num_tokens):",
        "_c = estimate_flops(",
        "def estimate_flops(num_params, num_tokens):\n"
        "    # TODO-3: 返回 6*N*D\n"
        "    return None\n"
        "\n\n",
        "TODO-3 estimate_flops",
    ),
    (
        None,
        "def has_nan_or_inf(t):",
        "_good = torch.tensor(",
        "def has_nan_or_inf(t):\n"
        "    # TODO-4: 含 nan 或 inf 返回 True，否则 False\n"
        "    return None\n"
        "\n\n",
        "TODO-4 has_nan_or_inf",
    ),
    (
        None,
        "def safe_log(t, eps=1e-8):",
        "_p = torch.tensor([0.0, 0.5, 1.0])",
        "def safe_log(t, eps=1e-8):\n"
        "    # TODO-5: 返回 log(t + eps)，避免 log(0)=-inf\n"
        "    return None\n"
        "\n\n",
        "TODO-5 safe_log",
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
