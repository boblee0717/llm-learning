"""
======================================================
PyTorch 专项 / 第 4 课（自写版）：损失函数与优化器
======================================================

用法：
1. 运行：python 04_loss_and_optim_self_write.py
2. 按 TODO-1 到 TODO-5 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错

目标：
- 手写 cross entropy（= NLL of log_softmax），和 F.cross_entropy 对拍
- 手写 warmup + cosine 学习率倍率函数（GPT 同款）
- 用 param groups 区分 decay / no-decay 参数（nanoGPT 标准写法）
- 手写训练五步曲 train_one_step
- 串成完整训练循环，验证 loss 收敛

对照：本课主课 04_loss_and_optim.py，phase1 第 2/4 课（损失 / 优化器）。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def require_close(name, actual, expected, atol=1e-5):
    require_not_none(name, actual)
    if not torch.allclose(actual, expected, atol=atol):
        raise ValidationError(
            f"{name} 数值不对\nactual=\n{actual}\nexpected=\n{expected}"
        )


# ============================================================
section("TODO-1：手写交叉熵 cross_entropy_manual")
# ============================================================
# 实现 F.cross_entropy 的等价计算（不许直接调 F.cross_entropy / F.nll_loss）：
#   1. log_probs = log_softmax(logits, dim=-1)        # (B, C)
#   2. 取每个样本【真实类别】的 log_prob：用 gather 或高级索引
#   3. 取负、再对 batch 求平均
#
# 提示：
#   log_probs = F.log_softmax(logits, dim=-1)
#   picked = log_probs[torch.arange(B), target]   # 每行取 target 那一列
#   return -picked.mean()


def cross_entropy_manual(logits, target):
    # TODO-1: 返回标量交叉熵损失
    return None


_logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 0.2, 3.0], [1.0, 1.0, 1.0]])
_target = torch.tensor([0, 2, 1])
_ce = cross_entropy_manual(_logits, _target)
require_not_none("TODO-1 cross_entropy_manual", _ce)
require_close("TODO-1 与 F.cross_entropy 对拍", _ce, F.cross_entropy(_logits, _target))
print(f"cross_entropy_manual OK：{_ce.item():.6f}（和 F.cross_entropy 一致）")


# ============================================================
section("TODO-2：warmup + cosine 学习率倍率 warmup_cosine_lr")
# ============================================================
# 返回 step 时刻的 lr 倍率（相对峰值的比例，范围大致 [min_ratio, 1]）：
#   - step < warmup_steps：线性 warmup，返回 step / warmup_steps
#   - 否则：cosine 衰减，progress = (step-warmup)/(total-warmup)，
#           cosine = 0.5*(1+cos(pi*progress))，
#           返回 min_ratio + (1-min_ratio)*cosine
#
# 提示：用 math.cos / math.pi；注意分母用 max(1, ...) 防止除零。


def warmup_cosine_lr(step, warmup_steps, total_steps, min_ratio=0.1):
    # TODO-2: 返回该 step 的 lr 倍率（float）
    return None


_warm, _total = 100, 1000
require_true("TODO-2 起点接近 0", abs(warmup_cosine_lr(0, _warm, _total) - 0.0) < 1e-6,
             "step=0 时倍率应为 0")
require_true("TODO-2 warmup 结束达峰值 1", abs(warmup_cosine_lr(_warm, _warm, _total) - 1.0) < 1e-6,
             "step=warmup_steps 时倍率应为 1")
require_true("TODO-2 warmup 中点为 0.5", abs(warmup_cosine_lr(50, _warm, _total) - 0.5) < 1e-6,
             "step=50 时应线性升到 0.5")
require_true("TODO-2 末尾降到 min_ratio 附近",
             abs(warmup_cosine_lr(_total, _warm, _total) - 0.1) < 1e-6,
             "step=total 时应降到 min_ratio=0.1")
print("warmup_cosine_lr OK：0 → 峰值 1（warmup 末）→ 余弦降到 0.1")


# ============================================================
section("TODO-3：param groups 区分 decay / no-decay build_param_groups")
# ============================================================
# 返回两个 param group 的列表，供 AdamW 使用：
#   - 二维及以上参数（矩阵权重，p.ndim >= 2）→ {"params": [...], "weight_decay": wd}
#   - 一维参数（bias / LayerNorm 的 γ/β，p.ndim < 2）→ {"params": [...], "weight_decay": 0.0}
#
# 提示：遍历 model.parameters()，按 p.ndim 分到两个列表，再各包成一个 dict。


def build_param_groups(model, wd):
    # TODO-3: 返回 [decay_group, no_decay_group]
    return None


_net = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 2))
_groups = build_param_groups(_net, wd=0.1)
require_not_none("TODO-3 build_param_groups", _groups)
require_true("TODO-3 返回两组", isinstance(_groups, list) and len(_groups) == 2,
             "应返回 [decay_group, no_decay_group]")
require_true("TODO-3 组0 做 decay", _groups[0]["weight_decay"] == 0.1)
require_true("TODO-3 组1 不做 decay", _groups[1]["weight_decay"] == 0.0)
# Linear weight 有 2 个（fc1/fc2），其余（2 个 bias + LayerNorm γ/β）是一维
require_true("TODO-3 decay 组是矩阵权重", len(_groups[0]["params"]) == 2,
             f"应有 2 个二维权重，实际 {len(_groups[0]['params'])}")
require_true("TODO-3 no-decay 组是一维参数", len(_groups[1]["params"]) == 4,
             f"应有 4 个一维参数（2 bias + γ + β），实际 {len(_groups[1]['params'])}")
_opt = torch.optim.AdamW(_groups, lr=1e-3)   # 能成功构造优化器
print("build_param_groups OK：矩阵权重做 decay，bias/LayerNorm 不做")


# ============================================================
section("TODO-4：训练五步曲 train_one_step")
# ============================================================
# 执行一次完整训练 step，返回这一步的 loss（float）：
#   pred = model(x)
#   loss = loss_fn(pred, y)
#   opt.zero_grad()
#   loss.backward()
#   opt.step()
#   return loss.item()
#
# 注意顺序：zero_grad 要在 backward 之前（清掉上一步残留的梯度）。


def train_one_step(model, x, y, opt, loss_fn):
    # TODO-4: 执行一次训练 step，返回 loss（float）
    return None


torch.manual_seed(1)
_lin = nn.Linear(4, 1)
_opt2 = torch.optim.SGD(_lin.parameters(), lr=0.1)
_x = torch.randn(16, 4)
_y = torch.randn(16, 1)
_l0 = train_one_step(_lin, _x, _y, _opt2, F.mse_loss)
_l1 = train_one_step(_lin, _x, _y, _opt2, F.mse_loss)
require_not_none("TODO-4 train_one_step", _l0)
require_true("TODO-4 返回的是 float", isinstance(_l0, float), "应返回 loss.item()")
require_true("TODO-4 loss 在下降", _l1 < _l0, f"两步 loss 应下降：{_l0:.4f} → {_l1:.4f}")
print(f"train_one_step OK：loss {_l0:.4f} → {_l1:.4f}")


# ============================================================
section("TODO-5：完整训练循环 train_loop")
# ============================================================
# 用 train_one_step + warmup_cosine_lr + LambdaLR 跑完整训练，返回最终 loss。
#   1. opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
#   2. sched = LambdaLR(opt, lr_lambda=lambda s: warmup_cosine_lr(s, warmup, total))
#   3. 循环 total 步：每步 train_one_step 后调用 sched.step()
#   4. 返回最后一步的 loss
#
# 提示：scheduler 每个 step 都要 sched.step() 推进。


def train_loop(model, x, y, base_lr, warmup, total, loss_fn):
    # TODO-5: 跑 total 步训练，返回最终 loss（float）
    return None


torch.manual_seed(2)
_xs = torch.linspace(-3, 3, 200).unsqueeze(1)
_ys = torch.sin(_xs)
_reg = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 1))
_final = train_loop(_reg, _xs, _ys, base_lr=1e-2, warmup=50, total=400, loss_fn=F.mse_loss)
require_not_none("TODO-5 train_loop", _final)
require_true("TODO-5 拟合 sin 收敛", _final < 0.01,
             f"最终 MSE 应 < 0.01，实际 {_final:.5f}（检查 step / sched 顺序）")
print(f"train_loop OK：拟合 sin(x) 最终 MSE = {_final:.5f}")


# ============================================================
section("全部 TODO 校验通过 ✓")
# ============================================================
print("""
你已经手写完成：
  1. 交叉熵 = NLL of log_softmax（和 F.cross_entropy 对拍）
  2. warmup + cosine 学习率倍率函数（GPT 同款曲线）
  3. param groups 区分 decay / no-decay（nanoGPT 标准写法）
  4. 训练五步曲 train_one_step（forward → loss → zero_grad → backward → step）
  5. 完整训练循环（优化器 + 调度器，loss 收敛）

复盘三问：
  * 为什么 cross_entropy 直接吃 logits？模型最后一层为什么不加 softmax？
  * 为什么 bias 和 LayerNorm 参数不做 weight decay？
  * warmup 解决了什么问题？为什么 lr 不一上来就用峰值？
""")
