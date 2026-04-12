"""
======================================================
第 2 课（自写版）：梯度下降 —— 模型如何学习
======================================================

用法：
1) 运行：python3 02_gradient_descent_self_write.py
2) 按 TODO-1 到 TODO-6 逐个补全
3) 每完成一个 TODO 就运行一次，查看校验报错

目标：
- 手写 MSE 损失
- 手写梯度
- 手写参数更新
- 手写训练循环
- 用数值梯度做正确性检查
- 对比不同学习率的效果
"""

import numpy as np


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


class ValidationError(Exception):
    """统一的练习校验错误。"""


def require_not_none(name: str, value, hint: str) -> None:
    if value is None:
        raise ValidationError(f"{name} 错误：结果是 None。{hint}")


def require_close(name: str, actual, expected, hint: str = "", atol: float = 1e-6) -> None:
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


def shape_or_none(x):
    return None if x is None else getattr(x, "shape", None)


def reference_mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)


def reference_gradients(X: np.ndarray, y_true: np.ndarray, w: float, b: float):
    n = len(X)
    y_pred = w * X + b
    error = y_pred - y_true
    dw = (2 / n) * np.sum(error * X)
    db = (2 / n) * np.sum(error)
    return dw, db


def reference_train(X, y_true, learning_rate=0.02, epochs=200):
    w, b = 0.0, 0.0
    loss_history = []
    for _ in range(epochs):
        y_pred = w * X + b
        loss = reference_mse_loss(y_pred, y_true)
        dw, db = reference_gradients(X, y_true, w, b)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        loss_history.append(loss)
    return w, b, np.array(loss_history)


section("第一部分：数据准备")
np.random.seed(42)
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_true = 2 * X + 1

print(f"训练输入 X: {X}")
print(f"真实输出 y: {y_true}")
print("目标规律：y = 2x + 1")
print()


section("第二部分：损失函数（TODO-1）")


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    TODO-1:
    实现 MSE（均方误差）
      MSE = mean((y_pred - y_true)^2)
    """
    # return ...
    raise NotImplementedError("TODO-1 未完成：请实现 mse_loss")


w0, b0 = 0.0, 0.0
y_pred0 = w0 * X + b0
loss0_error = None
loss0 = None
try:
    loss0 = mse_loss(y_pred0, y_true)
except Exception as err:  # noqa: BLE001
    loss0_error = err

print(f"初始参数: w={w0}, b={b0}")
print(f"初始预测: {y_pred0}")
print(f"初始损失: {loss0 if loss0 is not None else f'执行失败 -> {loss0_error}'}")
print()


section("第三部分：梯度计算（TODO-2）")


def compute_gradients(X: np.ndarray, y_true: np.ndarray, w: float, b: float):
    """
    TODO-2:
    已知：
      y_pred = w * X + b
      error  = y_pred - y_true

    推导结果：
      dw = (2/n) * sum(error * X)
      db = (2/n) * sum(error)
    """
    # y_pred = ...
    # error = ...
    # dw = ...
    # db = ...
    raise NotImplementedError("TODO-2 未完成：请实现梯度计算")


grads_error = None
dw0, db0 = None, None
try:
    dw0, db0 = compute_gradients(X, y_true, w0, b0)
except Exception as err:  # noqa: BLE001
    grads_error = err

print(f"初始梯度: dw={dw0}, db={db0 if grads_error is None else f'执行失败 -> {grads_error}'}")
print("提示：如果梯度是负数，参数更新时会增大（因为 w = w - lr * dw）")
print()


section("第四部分：单步参数更新（TODO-3）")


def gradient_step(w: float, b: float, dw: float, db: float, learning_rate: float):
    """
    TODO-3:
    按梯度下降规则更新参数：
      w_new = w - learning_rate * dw
      b_new = b - learning_rate * db
    """
    # w_new = ...
    # b_new = ...
    # return w_new, b_new
    raise NotImplementedError("TODO-3 未完成：请实现单步更新")


step_error = None
w1, b1 = None, None
try:
    if dw0 is not None and db0 is not None:
        w1, b1 = gradient_step(w0, b0, dw0, db0, learning_rate=0.02)
except Exception as err:  # noqa: BLE001
    step_error = err

print(f"一步更新后: w={w1}, b={b1 if step_error is None else f'执行失败 -> {step_error}'}")
print()


section("第五部分：完整训练循环（TODO-4）")


def train_linear_model(
    X: np.ndarray,
    y_true: np.ndarray,
    learning_rate: float = 0.02,
    epochs: int = 200,
    print_every: int = 40,
):
    """
    TODO-4:
    手写训练循环，返回 (w, b, loss_history)

    建议步骤：
    1) 初始化 w,b = 0
    2) 每轮计算 y_pred 和 loss
    3) 计算 dw, db
    4) 参数更新
    5) 记录 loss_history
    """
    # w, b = ...
    # loss_history = ...
    # for epoch in range(epochs):
    #     ...
    # return ...
    raise NotImplementedError("TODO-4 未完成：请实现训练循环")


train_error = None
w_final, b_final, loss_history = None, None, None
try:
    w_final, b_final, loss_history = train_linear_model(X, y_true, learning_rate=0.02, epochs=200)
except Exception as err:  # noqa: BLE001
    train_error = err

print(
    "训练结果: "
    f"{(w_final, b_final) if train_error is None else f'执行失败 -> {train_error}'}"
)
if loss_history is not None:
    print(
        f"loss_history: len={len(loss_history)}, "
        f"first={loss_history[0]:.4f}, last={loss_history[-1]:.6f}"
    )
print()


section("第六部分：梯度正确性检查（TODO-5，选做）")


def numerical_gradient_w(X: np.ndarray, y_true: np.ndarray, w: float, b: float, eps: float = 1e-5):
    """
    TODO-5（选做）:
    用有限差分近似 dw（只检验 w 的梯度）：
      dw_num ≈ (L(w + eps, b) - L(w - eps, b)) / (2*eps)

    说明：
    - 这是进阶检查项，用来验证你手写的解析梯度是否可靠
    - 不影响主线学习（可先跳过，后续再补）
    """
    # return ...
    raise NotImplementedError("TODO-5 未完成：请实现数值梯度")


grad_check_error = None
dw_num = None
try:
    dw_num = numerical_gradient_w(X, y_true, w=0.8, b=0.1)
except Exception as err:  # noqa: BLE001
    grad_check_error = err

print(f"数值梯度 dw_num: {dw_num if grad_check_error is None else f'执行失败 -> {grad_check_error}'}")
if isinstance(grad_check_error, NotImplementedError):
    print("提示：TODO-5 是选做，当前未完成不会阻塞其他 TODO 的校验。")
print()


section("第七部分：学习率实验（TODO-6）")

# 你需要填充这个字典：
# key: 学习率(float)
# value: (w, b, final_loss)
lr_results = {}

# TODO-6:
# 1) 用 train_linear_model 分别测试 0.001 / 0.02 / 0.1
# 2) 把结果存到 lr_results
# 示例：
# w_lr, b_lr, hist_lr = train_linear_model(...)
# lr_results[0.02] = (w_lr, b_lr, hist_lr[-1])

print(f"lr_results={lr_results}")
print()


def validate_all() -> None:
    """统一校验：检查 TODO-1 到 TODO-6。"""
    ref_loss0 = reference_mse_loss(y_pred0, y_true)
    ref_dw0, ref_db0 = reference_gradients(X, y_true, w0, b0)

    if loss0_error is not None:
        raise ValidationError(f"TODO-1 错误：mse_loss 执行失败。底层错误: {loss0_error}")
    require_not_none("TODO-1", loss0, "请返回标量损失值。")
    require_close("TODO-1", loss0, ref_loss0, "初始 MSE 应等于 57.0。")

    if grads_error is not None:
        raise ValidationError(f"TODO-2 错误：梯度函数执行失败。底层错误: {grads_error}")
    require_not_none("TODO-2", dw0, "请返回 dw。")
    require_not_none("TODO-2", db0, "请返回 db。")
    require_close("TODO-2", dw0, ref_dw0, "初始 dw 应为 -50.0。")
    require_close("TODO-2", db0, ref_db0, "初始 db 应为 -14.0。")

    if step_error is not None:
        raise ValidationError(f"TODO-3 错误：单步更新执行失败。底层错误: {step_error}")
    require_not_none("TODO-3", w1, "请返回更新后的 w。")
    require_not_none("TODO-3", b1, "请返回更新后的 b。")
    require_close("TODO-3", w1, 1.0, "lr=0.02 时，w 应更新为 1.0。")
    require_close("TODO-3", b1, 0.28, "lr=0.02 时，b 应更新为 0.28。")

    if train_error is not None:
        raise ValidationError(f"TODO-4 错误：训练循环执行失败。底层错误: {train_error}")
    require_not_none("TODO-4", w_final, "请返回最终 w。")
    require_not_none("TODO-4", b_final, "请返回最终 b。")
    require_not_none("TODO-4", loss_history, "请返回 loss_history。")
    require_true("TODO-4", len(loss_history) == 200, "loss_history 长度应等于 epochs。")
    require_true(
        "TODO-4",
        loss_history[0] > loss_history[-1],
        "训练后损失应该下降（first_loss > last_loss）。",
    )
    ref_w, ref_b, _ = reference_train(X, y_true, learning_rate=0.02, epochs=200)
    require_close(
        "TODO-4",
        [w_final, b_final],
        [ref_w, ref_b],
        "有限轮数训练未必精确等于 (2,1)，应与同配置参考实现一致。",
        atol=1e-6,
    )

    # TODO-5 是选做：未实现时跳过；实现后会进行正确性校验
    if grad_check_error is None:
        require_not_none("TODO-5", dw_num, "请返回 dw 的数值近似。")
        dw_ref, _ = reference_gradients(X, y_true, w=0.8, b=0.1)
        require_close("TODO-5", dw_num, dw_ref, "数值梯度应接近解析梯度。", atol=1e-4)
    elif isinstance(grad_check_error, NotImplementedError):
        print("校验提示：TODO-5（选做）未完成，已跳过该项。")
    else:
        print(f"校验提示：TODO-5（选做）执行异常，已跳过。错误: {grad_check_error}")

    require_true("TODO-6", isinstance(lr_results, dict), "lr_results 应为字典。")
    for lr in (0.001, 0.02, 0.1):
        require_true("TODO-6", lr in lr_results, f"lr_results 缺少学习率 {lr} 的结果。")
        w_lr, b_lr, loss_lr = lr_results[lr]
        require_true("TODO-6", np.isscalar(loss_lr), f"学习率 {lr} 的 final_loss 应为标量。")
        require_true("TODO-6", np.isfinite(loss_lr), f"学习率 {lr} 的 final_loss 不能是 NaN/Inf。")
        _ = (w_lr, b_lr)  # 仅用于明确元组结构

    # 规律性检查：合适学习率应该优于过小学习率
    require_true(
        "TODO-6",
        lr_results[0.02][2] < lr_results[0.001][2],
        "通常 lr=0.02 的最终损失应小于 lr=0.001（收敛更快）。",
    )


section("自写版骨架就绪")
print("你可以按 TODO-1 到 TODO-6 逐个补全。")
print("开始自动校验...")
validate_all()
print("校验通过：你当前实现正确。")
