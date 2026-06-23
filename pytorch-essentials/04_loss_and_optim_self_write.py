"""
======================================================
PyTorch 专项 / 第 4 课（自写版）：损失函数与优化器
======================================================

用法：
1. 运行：python 04_loss_and_optim_self_write.py
2. 按 TODO-1 到 TODO-5 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错
   （没填的 TODO 会返回 None，校验提示「未实现」，这是正常的）

目标：
- 用 log_softmax 手写交叉熵，并与 F.cross_entropy 对拍（不许用 F.cross_entropy）
- 实现训练五步曲 train_step，理解 zero_grad/backward/step 的顺序
- 用 param groups 把矩阵权重和一维参数分到不同 weight_decay
- 手写 warmup + cosine 的 lr 倍率函数（GPT 同款曲线）
- 串起 AdamW + LambdaLR 跑一个完整小训练

对照：本课主课 04_loss_and_optim.py（TODO 顺序对应主课 Part 1~5），
      以及 phase1 第 2/4 课（NumPy 手写损失与优化器）。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")  # ValidationError 走 stderr，也要防乱码

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


def require_shape(name, actual, expected_shape):
    require_not_none(name, actual)
    if tuple(actual.shape) != tuple(expected_shape):
        raise ValidationError(
            f"{name} 形状不对：actual={tuple(actual.shape)}, expected={tuple(expected_shape)}"
        )


def require_close(name, actual, expected, atol=1e-5):
    require_not_none(name, actual)
    if not torch.allclose(actual, expected, atol=atol):
        raise ValidationError(
            f"{name} 数值不对\nactual=\n{actual}\nexpected=\n{expected}"
        )


# ============================================================
section("TODO-1：用 log_softmax 手写交叉熵 manual_cross_entropy")
# ============================================================
# 输入 logits: (N, C) 原始分数，target: (N,) 是每个样本的真实类别索引。
# 返回标量：对所有样本求平均的交叉熵损失。
# 不许用 F.cross_entropy —— 要亲手拼出来，理解它到底等于什么。
#
# 提示（对照 phase1 你 NumPy 手写的 cross entropy）：
#   1. log_probs = F.log_softmax(logits, dim=-1)   # (N, C)
#   2. 取出每个样本目标类的 log 概率：
#        log_probs[torch.arange(N), target]        # (N,)
#      （这就是 NLLLoss 在做的事：在 log_softmax 上按 target 挑值）
#   3. 取负、再对 N 个样本求平均 → 标量
#   关键结论：F.cross_entropy = NLLLoss(log_softmax(logits))，所以它直接吃 logits。


def manual_cross_entropy(logits, target):
    # TODO-1: 用 log_softmax + 按 target 取值 + 取负求平均，返回标量交叉熵
    return None


_logits = torch.tensor([[2.0, 0.5, 0.1],
                        [0.1, 0.2, 3.0],
                        [1.0, 2.0, 0.3]])
_target = torch.tensor([0, 2, 1])
_ce_manual = manual_cross_entropy(_logits, _target)
require_not_none("TODO-1 manual_cross_entropy", _ce_manual)
require_shape("TODO-1 应返回标量", _ce_manual, ())
require_close("TODO-1 与 F.cross_entropy 对拍", _ce_manual,
              F.cross_entropy(_logits, _target))
print(f"manual_cross_entropy OK：{_ce_manual.item():.6f}"
      f"（= F.cross_entropy {F.cross_entropy(_logits, _target).item():.6f}）")


# ============================================================
section("TODO-2：训练五步曲 train_step")
# ============================================================
# 实现一步标准训练，返回这一步的 loss 标量值（float）。
# 用 mse_loss 作为损失。五步曲顺序很重要（对照主课 Part 2）：
#   1. opt.zero_grad()      # 清空上一步残留的梯度（不清会累加）
#   2. pred = model(x)      # 前向
#   3. loss = F.mse_loss(pred, y)   # 算损失
#   4. loss.backward()      # 反向求梯度（autograd）
#   5. opt.step()           # 按优化算法更新参数
# 最后 return loss.item()（标量 float，脱离计算图）。
#
# 提示：zero_grad 放在 backward 之前即可；返回 .item() 而不是 tensor。


def train_step(model, x, y, opt):
    # TODO-2: 实现训练五步曲，返回 loss.item()
    return None


torch.manual_seed(0)
_model = nn.Linear(4, 1)
_opt = torch.optim.SGD(_model.parameters(), lr=0.1)
_x = torch.randn(16, 4)
_y = torch.randn(16, 1)
_loss0 = train_step(_model, _x, _y, _opt)
_loss1 = train_step(_model, _x, _y, _opt)
require_not_none("TODO-2 train_step", _loss0)
require_true("TODO-2 返回的是 float 标量", isinstance(_loss0, float),
             "应 return loss.item()，而不是 tensor")
require_true("TODO-2 连续两步 loss 应下降", _loss1 < _loss0,
             f"step0={_loss0:.4f} 应大于 step1={_loss1:.4f}（没下降说明五步曲顺序/缺失有误）")
print(f"train_step OK：loss {_loss0:.4f} → {_loss1:.4f}（下降）")


# ============================================================
section("TODO-3：param groups 区分 weight decay build_param_groups")
# ============================================================
# 把模型参数分两组返回 [decay_group, no_decay_group]，每组是一个 dict：
#   - decay_group   : {"params": [...ndim>=2 的权重...], "weight_decay": 0.1}
#   - no_decay_group: {"params": [...ndim<2 的参数...],  "weight_decay": 0.0}
# 即：矩阵权重（Linear.weight 等，ndim>=2）做 weight decay；
#     一维参数（bias、LayerNorm 的 γ/β，ndim<2）不做。
# 这是 nanoGPT / GPT-2 训练里区分 decay/no-decay 的标准写法（对照主课 Part 3）。
#
# 提示：
#   decay   = [p for p in model.parameters() if p.ndim >= 2]
#   no_decay= [p for p in model.parameters() if p.ndim <  2]
#   return [{"params": decay, "weight_decay": 0.1},
#           {"params": no_decay, "weight_decay": 0.0}]


def build_param_groups(model):
    # TODO-3: 返回 [decay_group(wd=0.1), no_decay_group(wd=0.0)]
    return None


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)   # weight(2d)+bias(1d)
        self.ln = nn.LayerNorm(16)    # γ(1d)+β(1d)
        self.fc2 = nn.Linear(16, 2)   # weight(2d)+bias(1d)

    def forward(self, x):
        return self.fc2(self.ln(torch.relu(self.fc1(x))))


_net = _TinyNet()
_groups = build_param_groups(_net)
require_not_none("TODO-3 build_param_groups", _groups)
require_true("TODO-3 应返回长度为 2 的 list", isinstance(_groups, list) and len(_groups) == 2,
             "返回 [decay_group, no_decay_group]")
# 矩阵权重：fc1.weight, fc2.weight → 2 个；一维：fc1.bias, fc2.bias, ln.weight, ln.bias → 4 个
require_true("TODO-3 decay 组参数数量", len(_groups[0]["params"]) == 2,
             f"应有 2 个 ndim>=2 权重，实际 {len(_groups[0]['params'])}")
require_true("TODO-3 no_decay 组参数数量", len(_groups[1]["params"]) == 4,
             f"应有 4 个 ndim<2 参数，实际 {len(_groups[1]['params'])}")
require_true("TODO-3 decay 组 wd=0.1", abs(_groups[0]["weight_decay"] - 0.1) < 1e-9,
             "第 0 组 weight_decay 应为 0.1")
require_true("TODO-3 no_decay 组 wd=0.0", abs(_groups[1]["weight_decay"] - 0.0) < 1e-9,
             "第 1 组 weight_decay 应为 0.0")
# 能被 AdamW 正常接受
_ = torch.optim.AdamW(_groups, lr=1e-3)
print(f"build_param_groups OK：decay {len(_groups[0]['params'])} 个(wd=0.1)，"
      f"no_decay {len(_groups[1]['params'])} 个(wd=0.0)")


# ============================================================
section("TODO-4：warmup + cosine 学习率倍率 lr_lambda")
# ============================================================
# 返回一个【倍率】（乘到 base_lr 上），分两段（对照主课 Part 4）：
#   - step < warmup：线性从 0 升到 1，即 step / max(1, warmup)
#   - step >= warmup：cosine 从 1 平滑衰减到 min_ratio
#       progress = (step - warmup) / max(1, total - warmup)
#       cosine   = 0.5 * (1 + cos(pi * progress))     # 1 → 0
#       倍率     = min_ratio + (1 - min_ratio) * cosine  # 1 → min_ratio
#
# 提示：用 math.cos / math.pi；注意 max(1, ...) 防止除零。


def lr_lambda(step, warmup, total, min_ratio=0.1):
    # TODO-4: warmup 线性 + cosine 衰减，返回倍率（float）
    return None


_warmup, _total = 100, 1000
_r0 = lr_lambda(0, _warmup, _total)
_rw = lr_lambda(_warmup, _warmup, _total)
_rend = lr_lambda(_total, _warmup, _total)
require_not_none("TODO-4 lr_lambda", _r0)
require_true("TODO-4 step0 倍率≈0", abs(_r0 - 0.0) < 1e-6,
             f"warmup 起点应为 0，实际 {_r0}")
require_true("TODO-4 warmup 结束倍率≈1", abs(_rw - 1.0) < 1e-6,
             f"warmup 结束应到峰值 1，实际 {_rw}")
require_true("TODO-4 末端倍率≈min_ratio(0.1)", abs(_rend - 0.1) < 1e-6,
             f"末端应衰减到 min_ratio=0.1，实际 {_rend}")
# 单调性抽查：warmup 内递增，cosine 段递减
require_true("TODO-4 warmup 内应递增", lr_lambda(50, _warmup, _total) > _r0,
             "warmup 阶段应线性上升")
require_true("TODO-4 cosine 段应递减",
             lr_lambda(_warmup + 100, _warmup, _total) > lr_lambda(_warmup + 500, _warmup, _total),
             "cosine 阶段应单调下降")
print(f"lr_lambda OK：step0={_r0:.3f}  warmup={_rw:.3f}  end={_rend:.3f}")


# ============================================================
section("TODO-5：完整小训练 fit_one")
# ============================================================
# 用 AdamW + LambdaLR 跑 steps 步训练（拟合回归），返回【最终一步】的 loss(float)。
# 损失用 mse_loss。流程（对照主课 Part 5）：
#   1. opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
#   2. sched = LambdaLR(opt, lr_lambda=lambda s: lr_lambda(s, warmup, steps))
#      （warmup 取一个小数，比如 max(1, steps // 10)）
#   3. 循环 steps 次：每次跑五步曲（见 TODO-2），然后 sched.step()
#   4. return 最后一步的 loss.item()
#
# 提示：可以直接调用上面写好的 train_step；记得每个 step 后 sched.step()。
#       别忘了把每步的 loss 记下来，最后返回最后一步的值。


def fit_one(model, xs, ys, steps):
    # TODO-5: AdamW + LambdaLR 跑完整训练，返回最终 loss(float)
    return None


torch.manual_seed(1)
_xs = torch.linspace(-3, 3, 200).unsqueeze(1)
_ys = torch.sin(_xs)
_reg = nn.Sequential(nn.Linear(1, 64), nn.Tanh(),
                     nn.Linear(64, 64), nn.Tanh(),
                     nn.Linear(64, 1))
_final = fit_one(_reg, _xs, _ys, steps=400)
require_not_none("TODO-5 fit_one", _final)
require_true("TODO-5 返回 float", isinstance(_final, float),
             "应 return loss.item()")
require_true("TODO-5 loss 应明显下降到阈值以下", _final < 0.05,
             f"训练后 MSE 应 < 0.05，实际 {_final:.5f}（检查五步曲/调度器是否生效）")
print(f"fit_one OK：拟合 sin(x) 最终 MSE = {_final:.5f}（< 0.05）")


# ============================================================
section("全部 TODO 校验通过 ✓")
# ============================================================
print("""
你已经手写完成：
  1. manual_cross_entropy：F.cross_entropy = NLLLoss(log_softmax(logits))，直接吃 logits
  2. train_step：训练五步曲 zero_grad → forward → loss → backward → step
  3. build_param_groups：矩阵权重做 weight decay，bias/LayerNorm 不做（GPT 标准写法）
  4. lr_lambda：warmup 线性升 + cosine 平滑降（LLM 标准 lr 曲线）
  5. fit_one：AdamW + LambdaLR + 五步曲串成一个能收敛的完整训练

复盘三问：
  * 为什么 F.cross_entropy 直接吃 logits、模型最后一层不要自己 softmax？
  * zero_grad 为什么必须有？放在 backward 之前还是之后？不清会怎样？
  * 为什么 bias 和 LayerNorm 参数通常不做 weight decay？warmup 又是为了解决什么？
""")
