"""
======================================================
PyTorch 专项 / 第 2 课（自写版）：Autograd 自动微分
======================================================

用法：
1. 运行：python 02_autograd_self_write.py
2. 按 TODO-1 到 TODO-5 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错

目标（核心是「对拍」：让你确信 autograd 和手写梯度算的是同一件事）：
- 用 autograd 求线性层 + MSE 的梯度
- 不用 autograd，手写同一个梯度的解析式（对照 phase1）
- 用有限差分求数值梯度，三方对拍
- 用 detach 把张量截断成常数
- 手写一个完整的训练 step（forward → backward → no_grad 更新 → zero_grad）

对照：本课主课 02_autograd.py，phase1 第 2/3 课，phase0 第 4 课（数值梯度）。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")

import torch

torch.manual_seed(0)


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


class ValidationError(Exception):
    pass


def require_not_none(name, value):
    if value is None:
        raise ValidationError(f"{name} 未实现：结果是 None。")


def require_true(name, cond, hint=""):
    if not cond:
        raise ValidationError(f"{name} 条件不满足：{hint}")


def require_shape(name, actual, expected_shape):
    require_not_none(name, actual)
    if tuple(actual.shape) != tuple(expected_shape):
        raise ValidationError(
            f"{name} 形状不对：actual={tuple(actual.shape)}, expected={tuple(expected_shape)}"
        )


def require_close(name, actual, expected, atol=1e-4):
    require_not_none(name, actual)
    if not torch.allclose(actual, expected, atol=atol):
        raise ValidationError(
            f"{name} 数值不对\nactual=\n{actual}\nexpected=\n{expected}"
        )


# ============================================================
section("TODO-1：用 autograd 求线性层 + MSE 的梯度 autograd_grad")
# ============================================================
# 给定 W (out, in)（requires_grad=True）、x (in,)、target (out,)，
#   pred = W @ x
#   loss = sum((pred - target)²)
# 调用 backward 后返回 W.grad（克隆一份再返回，避免被后续清零影响）。
#
# 提示：
#   1. pred = W @ x
#   2. loss = ((pred - target) ** 2).sum()
#   3. loss.backward()
#   4. return W.grad.clone()


def autograd_grad(W, x, target):
    # TODO-1: 用 autograd 求 loss 对 W 的梯度并返回（.clone()）
    return None


_W = torch.randn(3, 4, requires_grad=True)
_x = torch.randn(4)
_target = torch.randn(3)
_g_auto = autograd_grad(_W, _x, _target)
require_shape("TODO-1 W.grad", _g_auto, (3, 4))
print("autograd_grad OK：拿到了 (3, 4) 的梯度")


# ============================================================
section("TODO-2：手写同一个梯度的解析式 analytic_grad")
# ============================================================
# 不许用 autograd！用 phase1 推过的链式法则手算：
#   loss = Σ_i (pred_i - target_i)²,  pred = W @ x
#   d_loss/d_W[i,j] = 2 * (pred_i - target_i) * x_j   → 这是一个外积
#
# 提示：pred = W @ x（可在 no_grad 下算）；
#   返回 2 * (pred - target).unsqueeze(1) * x.unsqueeze(0)   # (out,1)*(1,in)=(out,in)


def analytic_grad(W, x, target):
    # TODO-2: 不用 autograd，返回 loss 对 W 的解析梯度 (out, in)
    return None


_g_manual = analytic_grad(_W, _x, _target)
require_shape("TODO-2 解析梯度形状", _g_manual, (3, 4))
require_close("TODO-2 与 autograd 对拍一致", _g_manual, _g_auto)
print("analytic_grad OK：手写解析梯度和 autograd 逐位相等 ✓")


# ============================================================
section("TODO-3：有限差分数值梯度 numeric_grad")
# ============================================================
# 给定一个标量函数 f（输入向量、返回标量张量）和向量 x，用中心差分逐元素近似梯度：
#   grad[i] ≈ (f(x + eps·e_i) - f(x - eps·e_i)) / (2·eps)
# 返回和 x 同形状的梯度张量。
#
# 提示：
#   1. 在 torch.no_grad() 下做（纯数值，不需要建图）
#   2. 对每个 i：复制 x，第 i 个元素 +eps 算一次、-eps 算一次，相减除以 2eps
#   3. plus = x.clone(); plus[i] += eps   ...


def numeric_grad(f, x, eps=1e-4):
    # TODO-3: 返回 f 在 x 处的数值梯度（和 x 同形状）
    return None


def _f(t):
    return (t ** 3 + 2 * t).sum()   # f' = 3t² + 2


_t = torch.randn(5)
_num = numeric_grad(_f, _t)
require_shape("TODO-3 数值梯度形状", _num, (5,))
_analytic_f = 3 * _t ** 2 + 2       # 该函数的真实导数
require_close("TODO-3 数值梯度 ≈ 解析导数", _num, _analytic_f, atol=1e-2)
print("numeric_grad OK：中心差分逼近导数成功")


# ============================================================
section("TODO-4：把张量截断成常数 to_constant")
# ============================================================
# 给定 requires_grad=True 的张量 x，返回一个「当成常数」的版本：
# 不带梯度、但和 x 共享同一份数据。这是 detach 的典型用法
#（RLHF/蒸馏里把 reference model 的输出 detach 掉，不让梯度回传过去）。
#
# 提示：return x.detach()


def to_constant(x):
    # TODO-4: 返回 x 的 detach 版本（不带梯度，共享数据）
    return None


_xc = torch.randn(3, requires_grad=True)
_const = to_constant(_xc)
require_not_none("TODO-4 to_constant", _const)
require_true("TODO-4 结果不带梯度", _const.requires_grad is False,
             "detach 后 requires_grad 应为 False")
require_true("TODO-4 与原张量共享数据",
             _const.data_ptr() == _xc.data_ptr(),
             "detach 共享内存，不复制数据")
print("to_constant OK：detach 得到不带梯度、共享数据的常数视图")


# ============================================================
section("TODO-5：手写一个完整训练 step train_step")
# ============================================================
# 把前面串起来：对参数 w、b（都 requires_grad=True）做一次梯度下降。
#   1. pred = w * xs + b
#   2. loss = ((pred - ys) ** 2).mean()
#   3. loss.backward()
#   4. 在 torch.no_grad() 下：w -= lr*w.grad；b -= lr*b.grad
#   5. 清零：w.grad.zero_()；b.grad.zero_()
#   6. 返回 loss.item()（标量）
#
# 注意：第 4 步必须在 no_grad 里（更新参数本身不该被建图），
#       且要用 in-place 的 -=（保持 w/b 还是叶子张量）。


def train_step(w, b, xs, ys, lr):
    # TODO-5: 执行一次训练 step，返回这一步的 loss（float）
    return None


torch.manual_seed(1)
_true_w, _true_b = 2.5, -1.0
_xs = torch.linspace(-3, 3, 100)
_ys = _true_w * _xs + _true_b + 0.1 * torch.randn(100)
_w = torch.zeros(1, requires_grad=True)
_b = torch.zeros(1, requires_grad=True)

_first_loss = None
for _step in range(200):
    _l = train_step(_w, _b, _xs, _ys, lr=0.05)
    if _first_loss is None:
        _first_loss = _l

require_not_none("TODO-5 train_step 返回 loss", _first_loss)
require_true("TODO-5 loss 显著下降", _l < _first_loss * 0.1,
             f"初始 {_first_loss:.4f} → 最终 {_l:.4f}，下降不足，检查更新/清零步骤")
require_true("TODO-5 w 收敛到真值附近", abs(_w.item() - _true_w) < 0.1,
             f"w={_w.item():.3f}，真值 {_true_w}")
require_true("TODO-5 b 收敛到真值附近", abs(_b.item() - _true_b) < 0.1,
             f"b={_b.item():.3f}，真值 {_true_b}")
print(f"train_step OK：loss {_first_loss:.4f} → {_l:.4f}，"
      f"拟合出 w={_w.item():.3f} (真 {_true_w})，b={_b.item():.3f} (真 {_true_b})")


# ============================================================
section("全部 TODO 校验通过 ✓")
# ============================================================
print("""
你已经手写完成：
  1. 用 autograd 求线性层 + MSE 的梯度（forward → backward → .grad）
  2. 手写同一个梯度的解析式，并和 autograd 逐位对拍一致
  3. 中心差分数值梯度，三方互相验证
  4. detach 把张量截断成常数
  5. 完整训练 step（backward + no_grad 更新 + zero_grad）

复盘三问：
  * 为什么参数更新要写在 torch.no_grad() 里？不写会怎样？
  * 为什么每个 step 都要 zero_grad？不清零梯度会发生什么？
  * detach 和 no_grad 的区别是什么？各举一个用它的场景。
""")
