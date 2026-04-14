"""
重置 phase2 第1课自写练习脚本。

用途：
  python3 reset_exercises_01.py

效果：
  - 将 01_word_embeddings_self_write.py 中的 TODO 实现恢复为待填写状态
  - 保留讲解、打印与校验模块
"""

from __future__ import annotations

from pathlib import Path
import re


TARGET_FILE = Path(__file__).with_name("01_word_embeddings_self_write.py")


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"重置失败: {label} 匹配数量为 {count}（期望 1）")
    return new_text


def reset_position_encoding_block(text: str) -> str:
    replacement = (
        "def sinusoidal_position_encoding(max_len: int, d_model: int) -> np.ndarray:\n"
        "    \"\"\"\n"
        "    TODO-4:\n"
        "    按公式实现正弦位置编码\n"
        "      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))\n"
        "      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))\n"
        "    \"\"\"\n"
        "    # pe = ...\n"
        "    # position = ...\n"
        "    # div_term = ...\n"
        "    # pe[:, 0::2] = ...\n"
        "    # pe[:, 1::2] = ...\n"
        "    # return pe\n"
        "    raise NotImplementedError(\"TODO-4 未完成：请实现 sinusoidal_position_encoding\")\n"
    )
    start = text.find("def sinusoidal_position_encoding(")
    if start == -1:
        raise RuntimeError("重置失败: 找不到 sinusoidal_position_encoding 函数定义")

    end = text.find("\n\nmax_len = ", start)
    if end == -1:
        raise RuntimeError("重置失败: 找不到位置编码函数后续边界（max_len = ...）")

    return text[:start] + replacement + text[end + 2 :]


def main() -> int:
    if not TARGET_FILE.exists():
        print(f"未找到目标文件: {TARGET_FILE}")
        return 1

    text = TARGET_FILE.read_text(encoding="utf-8")

    replacements = [
        (r"^embedding_matrix\s*=.*$", "embedding_matrix = None", "TODO-1"),
        (r"^cat_embedding\s*=.*$", "cat_embedding = None", "TODO-2"),
        (r"^sim_cat_dog\s*=.*$", "sim_cat_dog = None", "TODO-3-1"),
        (r"^sim_cat_love\s*=.*$", "sim_cat_love = None", "TODO-3-2"),
        (r"^final_embedding\s*=.*$", "final_embedding = None", "TODO-5"),
        (r"^pos01_sim\s*=.*$", "pos01_sim = None", "TODO-6-1"),
        (r"^pos049_sim\s*=.*$", "pos049_sim = None", "TODO-6-2"),
    ]

    for pattern, replacement, label in replacements:
        text = replace_once(text, pattern, replacement, label)

    text = reset_position_encoding_block(text)

    TARGET_FILE.write_text(text, encoding="utf-8")
    print(f"已重置练习文件: {TARGET_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
