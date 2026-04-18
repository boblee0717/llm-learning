"""
位置编码可视化：直观展示
- (cos, sin) 在单位圆上随位置旋转
- 位移 k 等价于一个只依赖 k 的固定旋转
- 编码是唯一且有界的
"""

import numpy as np
import matplotlib.pyplot as plt


def build_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    pos = np.arange(max_len)[:, None]
    i = np.arange(0, d_model, 2)
    omega = 1.0 / (10000 ** (i / d_model))

    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(pos * omega)
    pe[:, 1::2] = np.cos(pos * omega)
    return pe, omega


def main() -> None:
    d_model = 64
    max_len = 100

    pe, omega = build_positional_encoding(max_len, d_model)

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(pe, aspect="auto", cmap="RdBu")
    ax1.set_title("Positional Encoding heatmap\nrow = pos, col = dim")
    ax1.set_xlabel("dimension")
    ax1.set_ylabel("position")
    fig.colorbar(im, ax=ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    pairs_to_plot = [0, 4, 16]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3)
    for idx, c in zip(pairs_to_plot, colors):
        s = pe[:, 2 * idx]
        co = pe[:, 2 * idx + 1]
        ax2.plot(co, s, "-o", color=c, markersize=3,
                 label=f"pair i={idx}, omega={omega[idx]:.4f}")
    ax2.set_aspect("equal")
    ax2.set_title("(cos, sin) trajectories on the unit circle\npos increases -> rotation")
    ax2.set_xlabel("cos(omega*pos)  = PE[:, 2i+1]")
    ax2.set_ylabel("sin(omega*pos)  = PE[:, 2i]")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    k = 5
    pair_i = 4
    s = pe[:, 2 * pair_i]
    c = pe[:, 2 * pair_i + 1]
    ax3.plot(c, s, "o-", color="lightgray", markersize=3, label="all positions")
    for start, color in [(10, "red"), (30, "blue"), (50, "green")]:
        ax3.annotate(
            "",
            xy=(c[start + k], s[start + k]),
            xytext=(c[start], s[start]),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )
    ax3.set_aspect("equal")
    ax3.set_title(f"Shift by k={k} is the SAME rotation\n(regardless of starting pos)")
    ax3.set_xlabel("cos")
    ax3.set_ylabel("sin")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(2, 2, 4)
    norms = np.linalg.norm(pe, axis=1, keepdims=True)
    sim = (pe @ pe.T) / (norms * norms.T)
    im2 = ax4.imshow(sim, cmap="viridis")
    ax4.set_title("Cosine similarity between positions\n(bright band = close positions)")
    ax4.set_xlabel("pos")
    ax4.set_ylabel("pos")
    fig.colorbar(im2, ax=ax4)

    plt.tight_layout()
    out_path = "phase2-transformer/01b_positional_encoding_viz.png"
    plt.savefig(out_path, dpi=120)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
