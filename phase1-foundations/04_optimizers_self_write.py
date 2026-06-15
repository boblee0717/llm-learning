"""
======================================================
第 4 课（自写版）：从零实现优化器
======================================================

用法：
1) 先运行原版 04_optimizers.py，理解每种优化器在解决什么问题
2) 运行本文件：python3 04_optimizers_self_write.py
3) 按 TODO-1 到 TODO-6 逐个补全（每个都是一个"单步更新"函数）
4) 每完成一个 TODO 就运行一次，查看校验报错

目标：
- 手写 SGD 单步更新
- 手写 Momentum（动量）单步更新
- 手写 RMSprop（自适应学习率）单步更新
- 手写 Adam（动量 + 自适应 + 偏差修正）单步更新
- 手写 AdamW（解耦权重衰减）单步更新
- 估算 Adam 训练的显存代价（理解优化器状态为何吃显存）

说明：所有更新函数都设计成"纯函数"——传入当前参数/状态，返回更新后的参数/状态，
      不在函数内部依赖全局变量。这样校验和复用都更干净。
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# ------------------------------------------------------------------
# 校验工具
# ------------------------------------------------------------------

class ValidationError(Exception):
    """统一的练习校验错误。"""


def require_not_none(name: str, value, hint: str) -> None:
    if value is None:
        raise ValidationError(f"{name} 错误：结果是 None。{hint}")


def require_close(name: str, actual, expected, hint: str = "", atol: float = 1e-8) -> None:
    try:
        if not np.allclose(actual, expected, atol=atol):
            raise ValidationError(
                f"{name} 错误：数值不正确。\nactual={actual}\nexpected={expected}\n{hint}"
            )
    except TypeError as err:
        raise ValidationError(f"{name} 错误：类型不正确。{hint}\n底层错误: {err}") from err


def require_true(name: str, condition: bool, hint: str) -> None:
    if not condition:
        raise ValidationError(f"{name} 错误：{hint}")


# ------------------------------------------------------------------
# 实验场景：各向异性的碗（和主课一致）
# ------------------------------------------------------------------

CURVATURE = np.array([1.0, 100.0])
START = np.array([-5.0, -5.0])
STEPS = 300


def grad_fn(w):
    return CURVATURE * w


def loss_fn(w):
    return 0.5 * np.sum(CURVATURE * w ** 2)


# ------------------------------------------------------------------
# 参考实现（仅用于自动校验，不要偷看 :)）
# ------------------------------------------------------------------

def _ref_sgd(p, g, lr):
    return p - lr * g


def _ref_momentum(p, g, v, lr, beta):
    v = beta * v + g
    return p - lr * v, v


def _ref_rmsprop(p, g, s, lr, beta, eps):
    s = beta * s + (1 - beta) * g ** 2
    return p - lr * g / (np.sqrt(s) + eps), s


def _ref_adam(p, g, m, v, t, lr, b1, b2, eps):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    return p - lr * mh / (np.sqrt(vh) + eps), m, v


def _ref_adamw(p, g, m, v, t, lr, b1, b2, eps, wd):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    return p - lr * (mh / (np.sqrt(vh) + eps) + wd * p), m, v


# ============================================================
# 第一部分：SGD —— 朴素梯度下降（TODO-1）
# ============================================================

section("第一部分：SGD")


def sgd_step(params, grads, lr):
    """
    TODO-1:
    实现 SGD 的单步更新（就是第 2 课那条公式）：
      params_new = params - lr * grads

    参数都是 np.ndarray，返回更新后的 params。
    """
    raise NotImplementedError("TODO-1 未完成：请实现 sgd_step")


# ============================================================
# 第二部分：Momentum 动量（TODO-2）
# ============================================================

section("第二部分：Momentum 动量")


def momentum_step(params, grads, velocity, lr, beta=0.9):
    """
    TODO-2:
    实现 Momentum 单步更新（PyTorch SGD 的 momentum 约定）：
      velocity = beta * velocity + grads
      params   = params - lr * velocity

    返回元组 (params_new, velocity_new)。
    注意：velocity 是"状态"，要返回新的 velocity 供下一步使用。
    """
    raise NotImplementedError("TODO-2 未完成：请实现 momentum_step")


# ============================================================
# 第三部分：RMSprop 自适应学习率（TODO-3）
# ============================================================

section("第三部分：RMSprop")


def rmsprop_step(params, grads, sq_avg, lr, beta=0.9, eps=1e-8):
    """
    TODO-3:
    实现 RMSprop 单步更新：
      sq_avg = beta * sq_avg + (1 - beta) * grads**2
      params = params - lr * grads / (sqrt(sq_avg) + eps)

    返回元组 (params_new, sq_avg_new)。
    提示：用 np.sqrt(sq_avg)。
    """
    raise NotImplementedError("TODO-3 未完成：请实现 rmsprop_step")


# ============================================================
# 第四部分：Adam（TODO-4）
# ============================================================

section("第四部分：Adam")


def adam_step(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    TODO-4:
    实现 Adam 单步更新（动量 + 自适应 + 偏差修正）：
      m = beta1 * m + (1 - beta1) * grads
      v = beta2 * v + (1 - beta2) * grads**2
      m_hat = m / (1 - beta1**t)      # t 从 1 开始
      v_hat = v / (1 - beta2**t)
      params = params - lr * m_hat / (sqrt(v_hat) + eps)

    返回元组 (params_new, m_new, v_new)。
    注意：m、v 都是状态，t 是当前步数（用于偏差修正），由调用方传入。
    """
    raise NotImplementedError("TODO-4 未完成：请实现 adam_step")


# ============================================================
# 第五部分：AdamW 解耦权重衰减（TODO-5）
# ============================================================

section("第五部分：AdamW")


def adamw_step(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    """
    TODO-5:
    实现 AdamW 单步更新。和 Adam 唯一的区别：权重衰减"解耦"出来，
    直接作用在参数上，不进梯度、也不被 sqrt(v_hat) 缩放：
      m, v, m_hat, v_hat 的算法同 Adam
      params = params - lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * params )

    返回元组 (params_new, m_new, v_new)。
    """
    raise NotImplementedError("TODO-5 未完成：请实现 adamw_step")


# ============================================================
# 第六部分：优化器的显存代价（TODO-6）
# ============================================================

section("第六部分：Adam 训练的显存代价")


def adam_training_bytes_per_param():
    """
    TODO-6:
    返回"混合精度 + Adam"训练时，每个参数占用的字节数（经典的 16 字节拆解）：
      fp16 权重 (2) + fp16 梯度 (2) + fp32 主权重 (4) + fp32 一阶矩 m (4) + fp32 二阶矩 v (4)

    返回一个整数（应当等于 16）。
    想一想：这 16 字节里，优化器状态(m + v)占了几个字节？（答案：8，是权重本身的好几倍）
    """
    raise NotImplementedError("TODO-6 未完成：请返回每参数字节数")


# ============================================================
# 把单步函数跑成完整训练，便于观察（不需要修改）
# ============================================================

def _run(step_kind):
    """用学生实现的单步函数，在各向异性碗上跑 STEPS 步，返回 (final_w, final_loss)。"""
    w = START.copy()
    if step_kind == "sgd":
        for _ in range(STEPS):
            w = sgd_step(w, grad_fn(w), lr=0.018)
    elif step_kind == "momentum":
        v = np.zeros_like(w)
        for _ in range(STEPS):
            w, v = momentum_step(w, grad_fn(w), v, lr=0.004)
    elif step_kind == "rmsprop":
        s = np.zeros_like(w)
        for _ in range(STEPS):
            w, s = rmsprop_step(w, grad_fn(w), s, lr=0.02)
    elif step_kind == "adam":
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        for t in range(1, STEPS + 1):
            w, m, v = adam_step(w, grad_fn(w), m, v, t, lr=0.2)
    elif step_kind == "adamw":
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        for t in range(1, STEPS + 1):
            w, m, v = adamw_step(w, grad_fn(w), m, v, t, lr=0.2, weight_decay=0.01)
    return w, loss_fn(w)


_run_errors = {}
_run_results = {}
for _kind in ["sgd", "momentum", "rmsprop", "adam", "adamw"]:
    try:
        _run_results[_kind] = _run(_kind)
    except Exception as err:  # noqa: BLE001
        _run_errors[_kind] = err

print("各优化器跑 300 步后的结果（需要先填完对应 TODO）：")
for _kind in ["sgd", "momentum", "rmsprop", "adam", "adamw"]:
    if _kind in _run_results:
        w, l = _run_results[_kind]
        print(f"  {_kind:<10} final loss = {l:.3e}   w = {np.round(w, 4)}")
    else:
        print(f"  {_kind:<10} 执行失败 -> {_run_errors[_kind]}")
print()


# ============================================================
# 自动校验
# ============================================================

def validate_all() -> None:
    rng = np.random.default_rng(0)
    p0 = rng.standard_normal(4)
    g_seq = [rng.standard_normal(4) for _ in range(5)]  # 固定一段梯度序列做对拍

    # --- TODO-1: SGD ---
    try:
        out = sgd_step(p0.copy(), g_seq[0], lr=0.1)
    except Exception as err:
        raise ValidationError(f"TODO-1 错误：sgd_step 执行失败。底层错误: {err}") from err
    require_not_none("TODO-1", out, "sgd_step 返回了 None。")
    require_close("TODO-1", out, _ref_sgd(p0, g_seq[0], 0.1), "SGD 更新结果不正确。")

    # --- TODO-2: Momentum ---
    try:
        p, v = p0.copy(), np.zeros(4)
        for g in g_seq:
            p, v = momentum_step(p, g, v, lr=0.05, beta=0.9)
    except Exception as err:
        raise ValidationError(f"TODO-2 错误：momentum_step 执行失败。底层错误: {err}") from err
    rp, rv = p0.copy(), np.zeros(4)
    for g in g_seq:
        rp, rv = _ref_momentum(rp, g, rv, 0.05, 0.9)
    require_close("TODO-2", p, rp, "Momentum 多步更新后 params 不正确。")
    require_close("TODO-2", v, rv, "Momentum 的 velocity 状态不正确（注意要返回新 velocity）。")

    # --- TODO-3: RMSprop ---
    try:
        p, s = p0.copy(), np.zeros(4)
        for g in g_seq:
            p, s = rmsprop_step(p, g, s, lr=0.05, beta=0.9, eps=1e-8)
    except Exception as err:
        raise ValidationError(f"TODO-3 错误：rmsprop_step 执行失败。底层错误: {err}") from err
    rp, rs = p0.copy(), np.zeros(4)
    for g in g_seq:
        rp, rs = _ref_rmsprop(rp, g, rs, 0.05, 0.9, 1e-8)
    require_close("TODO-3", p, rp, "RMSprop 多步更新后 params 不正确。")
    require_close("TODO-3", s, rs, "RMSprop 的 sq_avg 状态不正确。")

    # --- TODO-4: Adam ---
    try:
        p, m, v = p0.copy(), np.zeros(4), np.zeros(4)
        for t, g in enumerate(g_seq, start=1):
            p, m, v = adam_step(p, g, m, v, t, lr=0.05)
    except Exception as err:
        raise ValidationError(f"TODO-4 错误：adam_step 执行失败。底层错误: {err}") from err
    rp, rm, rv = p0.copy(), np.zeros(4), np.zeros(4)
    for t, g in enumerate(g_seq, start=1):
        rp, rm, rv = _ref_adam(rp, g, rm, rv, t, 0.05, 0.9, 0.999, 1e-8)
    require_close("TODO-4", p, rp, "Adam 多步更新后 params 不正确（检查偏差修正 1-beta**t）。")
    require_close("TODO-4", m, rm, "Adam 的一阶矩 m 不正确。")
    require_close("TODO-4", v, rv, "Adam 的二阶矩 v 不正确。")

    # --- TODO-5: AdamW ---
    try:
        p, m, v = p0.copy(), np.zeros(4), np.zeros(4)
        for t, g in enumerate(g_seq, start=1):
            p, m, v = adamw_step(p, g, m, v, t, lr=0.05, weight_decay=0.1)
    except Exception as err:
        raise ValidationError(f"TODO-5 错误：adamw_step 执行失败。底层错误: {err}") from err
    rp, rm, rv = p0.copy(), np.zeros(4), np.zeros(4)
    for t, g in enumerate(g_seq, start=1):
        rp, rm, rv = _ref_adamw(rp, g, rm, rv, t, 0.05, 0.9, 0.999, 1e-8, 0.1)
    require_close("TODO-5", p, rp, "AdamW 多步更新后 params 不正确（weight_decay 应直接乘在 params 上）。")
    # 额外检查：weight_decay=0 时 AdamW 应等价于 Adam
    pa, ma, va = p0.copy(), np.zeros(4), np.zeros(4)
    pw, mw, vw = p0.copy(), np.zeros(4), np.zeros(4)
    for t, g in enumerate(g_seq, start=1):
        pa, ma, va = adam_step(pa, g, ma, va, t, lr=0.05)
        pw, mw, vw = adamw_step(pw, g, mw, vw, t, lr=0.05, weight_decay=0.0)
    require_close("TODO-5", pw, pa, "weight_decay=0 时，AdamW 应当与 Adam 完全等价。")

    # --- TODO-6: 显存代价 ---
    try:
        bpp = adam_training_bytes_per_param()
    except Exception as err:
        raise ValidationError(f"TODO-6 错误：adam_training_bytes_per_param 执行失败。底层错误: {err}") from err
    require_not_none("TODO-6", bpp, "函数返回了 None。")
    require_true("TODO-6", int(bpp) == 16,
                 f"每参数应为 16 字节（2+2+4+4+4），你返回了 {bpp}。")

    # --- 端到端：用学生函数把碗优化下去 ---
    require_true("SGD-训练", _run_results.get("sgd", (None, 1e9))[1] < 1e-2,
                 "用你的 sgd_step 跑 300 步后 loss 应明显下降。")
    require_true("Adam-训练", _run_results.get("adam", (None, 1e9))[1] < 1e-3,
                 "用你的 adam_step 跑 300 步后 loss 应收敛到很小。")


section("自写版骨架就绪")
print("你可以按 TODO-1 到 TODO-6 逐个补全。")
print("开始自动校验...")
validate_all()
print("校验通过：你当前实现正确。恭喜完成第 4 课！")
