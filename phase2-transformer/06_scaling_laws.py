"""
第 6 课：Scaling Law 与 Compute-Optimal 训练
================================================
回答「该搭多大模型 / 用多少数据 / 烧多少 compute」的工程问题。

核心问题：
- 模型 loss 是怎么随参数量 N、数据量 D、训练 compute C 变化的？
- 给定 compute 预算，N 和 D 应该各自取多少？
- 为什么 Chinchilla 70B 能打 Gopher 280B？
- Llama 3-8B 训了 15T tokens（远超 Chinchilla）为什么不是浪费？

与大模型的关系：
- GPT-3 175B + 300B tokens 是按 Kaplan 2020 算出来的「compute-optimal」
- Chinchilla 2022 推翻了 Kaplan：N 和 D 应该按 1:20 同步扩
- 工业界（Llama 3 / DeepSeek-V3）实际还会 over-train 更多 tokens
  —— 因为推理 cost 不在 Chinchilla 的优化目标里

前置阅读：
- papers/notes/scaling_laws_kaplan_2020.md
- papers/notes/chinchilla_compute_optimal_2022.md

这节课不真的训模型（小尺寸上拟合不出有意义的幂律，需要 100+ 模型 × 几万 GPU-hours）。
我们的目标是把两篇论文的结论用代码「翻译」成可手算的工程估算：
  1. 三条幂律的形状
  2. C ≈ 6ND 公式
  3. compute_optimal_chinchilla(C) 工具函数
  4. 把工业模型放在 N-D 平面上对比
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能跑通，图存到本地 PNG
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================================
# Part 1: Kaplan 2020 的三条幂律
# ============================================================

print("=" * 60)
print("Part 1: Kaplan 2020 的三条幂律")
print("=" * 60)

# 三条幂律来自论文 §3.1（每一条的适用条件：另两个变量不能成为瓶颈）
#
#   L(N) = (N_c / N) ** alpha_N   # N: 非 embedding 参数量
#   L(D) = (D_c / D) ** alpha_D   # D: 训练 token 数
#   L(C) = (C_c / C) ** alpha_C   # C: 训练总 compute（PF-days）
#
# loss 单位是 nats per token（自然对数下的交叉熵），perplexity = exp(loss)

KAPLAN = {
    "L_N": {"N_c": 8.8e13, "alpha": 0.076, "desc": "参数量 N 是瓶颈"},
    "L_D": {"D_c": 5.4e13, "alpha": 0.095, "desc": "数据量 D 是瓶颈"},
    "L_C": {"C_c": 3.1e8, "alpha": 0.050, "desc": "compute C 是瓶颈（PF-days）"},
}


def loss_vs_N(N):
    """L(N) = (N_c / N) ** alpha_N，仅在数据足够、训练充分时有效"""
    return (KAPLAN["L_N"]["N_c"] / N) ** KAPLAN["L_N"]["alpha"]


def loss_vs_D(D):
    """L(D) = (D_c / D) ** alpha_D"""
    return (KAPLAN["L_D"]["D_c"] / D) ** KAPLAN["L_D"]["alpha"]


def loss_vs_C(C_pf_days):
    """L(C) = (C_c / C) ** alpha_C；C 单位是 PF-days"""
    return (KAPLAN["L_C"]["C_c"] / C_pf_days) ** KAPLAN["L_C"]["alpha"]


for key, cfg in KAPLAN.items():
    print(f"  {key:5s} | {cfg['desc']:24s} | alpha = {cfg['alpha']:.3f}")

# 翻倍带来 loss 下降的直觉
print("\n  「翻倍 X 带来的 loss 相对变化」")
print(f"  N 翻倍 → loss × 2 ** -{KAPLAN['L_N']['alpha']} = {2 ** -KAPLAN['L_N']['alpha']:.4f}  ({(1 - 2 ** -KAPLAN['L_N']['alpha']) * 100:.1f}% 下降)")
print(f"  D 翻倍 → loss × 2 ** -{KAPLAN['L_D']['alpha']} = {2 ** -KAPLAN['L_D']['alpha']:.4f}  ({(1 - 2 ** -KAPLAN['L_D']['alpha']) * 100:.1f}% 下降)")
print(f"  C 翻倍 → loss × 2 ** -{KAPLAN['L_C']['alpha']} = {2 ** -KAPLAN['L_C']['alpha']:.4f}  ({(1 - 2 ** -KAPLAN['L_C']['alpha']) * 100:.1f}% 下降)")

# 三个常见尺寸的 L(N) 预测
print("\n  三个常见模型尺寸下的 L(N) 预测：")
for name, N in [("GPT-2 small (124M)", 124e6), ("GPT-3 (175B)", 175e9), ("Chinchilla (70B)", 70e9)]:
    print(f"  {name:24s}  N = {N:>10.2e}   L(N) ≈ {loss_vs_N(N):.3f} nats/token  (ppl ≈ {np.exp(loss_vs_N(N)):.2f})")

print("""
  提醒：
  - L(N) 是「假设 D 和 C 都足够」的极限值，真实训练会高于这条曲线
  - Kaplan 估的 N 不含 embedding；用 N ≈ 12 · L · d_model² 估更稳
  - 这些指数后来被 Chinchilla 部分修正（特别是「N 应该比 D 长得快」这条）
""")


# ============================================================
# Part 2: 画三条幂律曲线
# ============================================================

print("=" * 60)
print("Part 2: 画三条幂律曲线（log-log 直线）")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# L(N)
N_range = np.logspace(6, 12, 100)
axes[0].loglog(N_range, loss_vs_N(N_range), "b-", linewidth=2)
axes[0].set_xlabel("Parameters N (non-embedding)")
axes[0].set_ylabel("Loss (nats/token)")
axes[0].set_title(f"L(N) — alpha = {KAPLAN['L_N']['alpha']}")
axes[0].grid(True, which="both", alpha=0.3)
# 标几个实际模型
for name, N, color in [
    ("GPT-2 small", 124e6, "orange"),
    ("GPT-3", 175e9, "red"),
    ("Chinchilla", 70e9, "green"),
]:
    axes[0].scatter([N], [loss_vs_N(N)], color=color, s=80, zorder=5, label=name)
axes[0].legend(loc="upper right", fontsize=8)

# L(D)
D_range = np.logspace(8, 13, 100)
axes[1].loglog(D_range, loss_vs_D(D_range), "g-", linewidth=2)
axes[1].set_xlabel("Training tokens D")
axes[1].set_ylabel("Loss (nats/token)")
axes[1].set_title(f"L(D) — alpha = {KAPLAN['L_D']['alpha']}")
axes[1].grid(True, which="both", alpha=0.3)

# L(C)
C_range = np.logspace(0, 5, 100)  # PF-days
axes[2].loglog(C_range, loss_vs_C(C_range), "r-", linewidth=2)
axes[2].set_xlabel("Compute C (PF-days)")
axes[2].set_ylabel("Loss (nats/token)")
axes[2].set_title(f"L(C) — alpha = {KAPLAN['L_C']['alpha']}")
axes[2].grid(True, which="both", alpha=0.3)
# GPT-3 ≈ 3640 PF-days
axes[2].scatter([3640], [loss_vs_C(3640)], color="red", s=80, zorder=5, label="GPT-3 (3640 PF-days)")
axes[2].legend(loc="upper right", fontsize=8)

plt.suptitle("Kaplan 2020: three power laws (straight lines on log-log)", fontsize=12)
plt.tight_layout()
out_path = "phase2-transformer/scaling_laws_part2.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"  三条幂律图已保存到 {out_path}")
print("  关键观察：log-log 图上是直线 = 幂律。斜率 = 指数 α。")
plt.close(fig)


# ============================================================
# Part 3: C ≈ 6ND 与 compute_optimal_chinchilla(C)
# ============================================================

print("\n" + "=" * 60)
print("Part 3: C ≈ 6ND 与 Chinchilla compute-optimal")
print("=" * 60)


def training_flops(N, D):
    """
    估算训练 compute。
    每个 token 训练一次需要 ≈ 6N FLOPs（前向 2N + 反向 4N）
    总 compute = 6 · N · D
    """
    return 6.0 * N * D


def compute_optimal_chinchilla(C):
    """
    给定 compute budget C（FLOPs），按 Chinchilla 2022 推 (N*, D*)。

    Chinchilla 三种 approach 都给出 N : D ≈ 1 : 1（同步扩）+ 1:20 经验比。
    最简口径：N* ∝ C^0.5、D* ∝ C^0.5，且满足 D = 20·N、C = 6·N·D。

    代入：C = 6 · N · (20·N) = 120 · N²
        → N* = sqrt(C / 120)
        → D* = 20 · N*

    （论文用的指数其实是 0.46 / 0.54，不是精确 0.5，但 0.5 更好记，
     量级一致；想精确就用论文 Table 3 给的拟合系数。）
    """
    N_star = np.sqrt(C / 120.0)
    D_star = 20.0 * N_star
    return N_star, D_star


def chinchilla_optimal_tokens(N):
    """已知 N，按 Chinchilla 1:20 口径推荐的训练 token 数。"""
    return 20.0 * N


# 演示 1：拿到一个模型的 (N, D)，秒算训练 FLOPs
print("\n  演示 1：用 C ≈ 6ND 估训练 FLOPs")
print(f"  {'Model':<22s} {'N':>10s} {'D':>10s} {'C (FLOPs)':>15s} {'D/N':>8s}")
models = [
    ("GPT-3", 175e9, 300e9),
    ("Gopher", 280e9, 300e9),
    ("Chinchilla", 70e9, 1.4e12),
    ("Llama 2-7B", 7e9, 2e12),
    ("Llama 3-8B", 8e9, 15e12),
    ("Llama 3-70B", 70e9, 15e12),
    ("Qwen2.5-7B", 7e9, 18e12),
    ("DeepSeek-V3 (active)", 37e9, 14.8e12),
]
for name, N, D in models:
    C = training_flops(N, D)
    print(f"  {name:<22s} {N:>10.2e} {D:>10.2e} {C:>15.2e} {D/N:>8.0f}")

# 演示 2：给 C 推 N* 和 D*
print("\n  演示 2：给定 compute budget，按 Chinchilla 推最优 (N*, D*)")
print(f"  {'C (FLOPs)':>15s} {'N* (params)':>15s} {'D* (tokens)':>15s} {'Description':>30s}")
for C, desc in [
    (3.15e23, "GPT-3 同款 compute"),
    (5.76e23, "Chinchilla / Gopher 同款"),
    (7.2e23, "Llama 3-8B 同款"),
    (1e24, "1024 × H100 × 30天 @ MFU=40%"),
    (1e25, "下一代旗舰级"),
]:
    N_star, D_star = compute_optimal_chinchilla(C)
    print(f"  {C:>15.2e} {N_star:>15.2e} {D_star:>15.2e}   {desc:>30s}")

# 演示 3：给定 N 反推 D
print("\n  演示 3：给定 N，按 Chinchilla 1:20 口径推 D")
print(f"  {'N':>10s} {'D = 20·N':>15s}  {'实际工业模型 D':>20s}")
print(f"  {'7B':>10s} {chinchilla_optimal_tokens(7e9):>15.2e}  {'Llama 3-8B 训了 15T （over 100×）'}")
print(f"  {'70B':>10s} {chinchilla_optimal_tokens(70e9):>15.2e}  {'Llama 3-70B 训了 15T （over 10×）'}")
print(f"  {'175B':>10s} {chinchilla_optimal_tokens(175e9):>15.2e}  {'GPT-3 只训了 300B （under 12×）'}")
print(f"  {'280B':>10s} {chinchilla_optimal_tokens(280e9):>15.2e}  {'Gopher 只训了 300B （under 19×）'}")


# ============================================================
# Part 4: 把工业模型放在 N-D 平面上对比
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 工业模型在 N-D 平面上的分布")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 7))

# 画 Chinchilla optimal 线：D = 20·N
N_line = np.logspace(8, 12.5, 100)
D_line = 20.0 * N_line
ax.loglog(N_line, D_line, "k--", linewidth=1.5, label="Chinchilla optimal (D = 20·N)", alpha=0.7)

# 画几条 IsoFLOP 等 compute 曲线：C = 6ND = const → D = C/(6N)
for C, label in [
    (1e22, "C = 1e22"),
    (1e23, "C = 1e23"),
    (1e24, "C = 1e24"),
    (1e25, "C = 1e25"),
]:
    N_iso = np.logspace(8, 12.5, 100)
    D_iso = C / (6.0 * N_iso)
    ax.loglog(N_iso, D_iso, "gray", linewidth=0.8, alpha=0.4)
    # 标 label 在曲线右端附近
    ax.text(3e12, C / (6 * 3e12), label, fontsize=8, color="gray", alpha=0.8)

# 标各个模型
models_to_plot = [
    ("GPT-3", 175e9, 300e9, "red", "o"),
    ("Gopher", 280e9, 300e9, "darkred", "o"),
    ("MT-NLG", 530e9, 270e9, "firebrick", "o"),
    ("Chinchilla", 70e9, 1.4e12, "green", "s"),
    ("Llama 2-7B", 7e9, 2e12, "steelblue", "^"),
    ("Llama 2-70B", 70e9, 2e12, "navy", "^"),
    ("Llama 3-8B", 8e9, 15e12, "deepskyblue", "^"),
    ("Llama 3-70B", 70e9, 15e12, "blue", "^"),
    ("Qwen2.5-7B", 7e9, 18e12, "purple", "D"),
    ("DeepSeek-V3 (active)", 37e9, 14.8e12, "orange", "*"),
]
for name, N, D, color, marker in models_to_plot:
    ax.scatter([N], [D], color=color, s=140, marker=marker, edgecolors="black", linewidth=0.5, zorder=5, label=name)
    ax.annotate(name, (N, D), xytext=(8, 5), textcoords="offset points", fontsize=8)

ax.set_xlabel("Parameters N", fontsize=11)
ax.set_ylabel("Training tokens D", fontsize=11)
ax.set_title("Industrial models on the N-D plane\n(dashed = Chinchilla optimal; grey = IsoFLOP)", fontsize=12)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left", fontsize=8, ncol=2)

ax.text(
    3e11, 3e11,
    "GPT-3 / Gopher / MT-NLG\nseverely UNDER-trained\n(below Chinchilla line)",
    fontsize=9, color="darkred", ha="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.7),
)
ax.text(
    1e10, 1e13,
    "Llama 3 / Qwen2.5\naggressively OVER-trained\n(small N, lots of tokens)",
    fontsize=9, color="darkblue", ha="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.7),
)

plt.tight_layout()
out_path = "phase2-transformer/scaling_laws_part4.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"  N-D 平面图已保存到 {out_path}")
plt.close(fig)


# ============================================================
# Part 5: 复盘：3 个手算题（答案给在最后）
# ============================================================

print("\n" + "=" * 60)
print("Part 5: 复盘 —— 3 个手算题")
print("=" * 60)

print("""
  Q1：手上有 1024 张 H100 跑 30 天（MFU = 40%），按 Chinchilla 该训多大？

  Q2：GPT-3 175B 训了 300B tokens。如果给定 GPT-3 的训练 compute，
      按 Chinchilla 改训多大的模型 + 多少 tokens？

  Q3：Llama 3-8B 训了 15T tokens，按 Chinchilla 只需要 ~160B。
      多训 ~94 倍是不是浪费？为什么？

  -------- 答案 --------
""")

# Q1 答案
single_h100 = 1e15           # FLOPs/s (BF16)
n_gpu = 1024
days = 30
mfu = 0.40
total_compute = n_gpu * single_h100 * days * 86400 * mfu
N_star, D_star = compute_optimal_chinchilla(total_compute)
print(f"  Q1: C = {n_gpu} × 1e15 × {days}天 × {mfu} × 86400 = {total_compute:.2e} FLOPs")
print(f"      Chinchilla optimal: N* ≈ {N_star/1e9:.1f} B, D* ≈ {D_star/1e12:.2f} T")
print(f"      → 大致是 Chinchilla 70B 量级\n")

# Q2 答案
N_gpt3, D_gpt3 = 175e9, 300e9
C_gpt3 = training_flops(N_gpt3, D_gpt3)
N_star, D_star = compute_optimal_chinchilla(C_gpt3)
print(f"  Q2: GPT-3 compute = 6 × 175B × 300B = {C_gpt3:.2e} FLOPs")
print(f"      Chinchilla optimal: N* ≈ {N_star/1e9:.1f} B, D* ≈ {D_star/1e12:.2f} T")
print(f"      → 同样 compute 应该训 ~{N_star/1e9:.0f}B 模型 × ~{D_star/1e12:.1f}T tokens")
print(f"      → 这就是 Chinchilla 70B 用同样 compute 打败 Gopher 280B 的原理\n")

# Q3 答案
print("  Q3: 不浪费。Chinchilla 只优化「训练 compute-optimal」，")
print("      没把推理 cost 算进去。Llama 3-8B 训得越久，能力越接近 70B 模型，")
print("      但推理时 cost 只有 70B 的 ~1/10 —— 多花 5-10 倍训练 compute，")
print("      在 inference-heavy 场景下长期回本。这叫 inference-aware scaling。")


# ============================================================
# 关键 takeaway
# ============================================================

print("\n" + "=" * 60)
print("第 6 课关键 takeaway")
print("=" * 60)
print("""
  1. 三条幂律 L(N) / L(D) / L(C) 在 log-log 图上是直线 —— 这是大模型「可外推」的基础
  2. C ≈ 6·N·D 是估训练 compute 的工程口算神器（前向 2N + 反向 4N）
  3. Chinchilla 修正 Kaplan：N 和 D 应该「按 1:20 同步扩」，而不是「优先扩 N」
  4. GPT-3 在 Chinchilla 视角下严重欠训；Chinchilla 70B 用同 compute 打败 Gopher 280B
  5. Chinchilla 1:20 是「训练 optimal」，不是「推理 optimal」
     —— 工业界（Llama 3 / DeepSeek-V3）实际跑到 1:200 ~ 1:2000，因为推理 cost 重要

  下一步：
  - phase4 第 1 课会用这套框架解 DeepSeek-V3 的 671B-total / 37B-active
  - phase4 第 8 课会引入「测试时 compute」这第四个 scaling 变量
""")
