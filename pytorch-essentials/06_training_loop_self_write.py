"""
======================================================
PyTorch 专项 / 第 6 课（自写版）：工程化训练循环 ⭐
======================================================

用法：
1. 运行：python 06_training_loop_self_write.py
2. 按 TODO-1 到 TODO-5 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错
   （没填的 TODO 会返回 None，校验提示「未实现」，这是正常的）

目标：
- 写出带 eval()/no_grad/恢复 train() 的 evaluate
- 手写梯度累积训练一步（loss/accum、累加 backward、accum 后 step+zero）
- 手写梯度裁剪 + step（返回裁剪前总范数）
- 手写 checkpoint 保存/加载（model+opt+step 四件套，验证恢复一致）
- 实现 EarlyStopper（patience 监控 val loss）

对照：本课主课 06_training_loop.py（Part1/3/3/4/5），phase3 第 1 课预习。
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")  # ValidationError 走 stderr，也要防乱码

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


# 小工具：构造一个固定的回归模型与数据，全程复用
def make_model():
    return nn.Sequential(
        nn.Linear(1, 16), nn.Tanh(),
        nn.Linear(16, 1),
    )


def make_data(n=64):
    xs = torch.linspace(-3, 3, n).unsqueeze(1)
    ys = torch.sin(xs)
    return xs, ys


# ============================================================
section("TODO-1：评估函数 evaluate(model, xs, ys)")
# ============================================================
# 在【评估模式】下算平均 MSE loss 并返回 float。要求：
#   1. 进函数先记下 model.training，切到 model.eval()
#   2. 在 torch.no_grad() 下前向算 F.mse_loss(model(xs), ys)
#   3. 算完若原来是 train 状态，恢复 model.train()，别污染后续训练
#   4. 返回 python float（.item()）
#
# 提示：
#   was_training = model.training
#   model.eval()
#   with torch.no_grad():
#       loss = F.mse_loss(model(xs), ys).item()
#   if was_training: model.train()
#   return loss


def evaluate(model, xs, ys):
    # TODO-1: eval() + no_grad 算平均 loss，返回 float，结束恢复 train()
    return None


_m = make_model()
_m.train()
_xs, _ys = make_data()
_val = evaluate(_m, _xs, _ys)
require_not_none("TODO-1 evaluate", _val)
require_true("TODO-1 返回的是 float", isinstance(_val, float), "应返回 python float")
require_true("TODO-1 数值与手算一致",
             abs(_val - F.mse_loss(_m(_xs), _ys).item()) < 1e-6,
             "应等于 MSE loss")
require_true("TODO-1 评估后恢复了 train 模式", _m.training,
             "进来时是 train，评估完应 model.train() 恢复")
print(f"evaluate OK：val loss={_val:.5f}，且评估后 model.training={_m.training}")


# ============================================================
section("TODO-2：梯度累积训练一步 train_step_accum")
# ============================================================
# 把多个 micro-batch 累积成一次更新。给定 batches = [(xb,yb), ...]，accum_steps 个。
# 要求：
#   1. opt.zero_grad()
#   2. 对每个 (xb,yb)：loss = F.mse_loss(model(xb), yb) / accum_steps；loss.backward()
#      （除以 accum_steps，且不要在中间 zero_grad，让梯度累加）
#      把每个 micro 的 loss.item() 累加进 running
#   3. 全部 backward 完后 opt.step()，再 opt.zero_grad()
#   4. 返回平均 loss（running，即各 micro loss/accum 之和）
#
# 提示：
#   opt.zero_grad(); running = 0.0
#   for xb, yb in batches:
#       l = F.mse_loss(model(xb), yb) / accum_steps
#       l.backward(); running += l.item()
#   opt.step(); opt.zero_grad()
#   return running


def train_step_accum(model, batches, opt, accum_steps):
    # TODO-2: 梯度累积一步，返回平均 loss
    return None


torch.manual_seed(1)
_m = make_model()
_opt = torch.optim.SGD(_m.parameters(), lr=0.1)
_xs, _ys = make_data()
_accum = 4
_batches = list(zip(torch.chunk(_xs, _accum), torch.chunk(_ys, _accum)))
# 参考实现：用同一初始模型对照「整批一次 backward」的等效平均 loss
torch.manual_seed(1)
_m_ref = make_model()
_opt_ref = torch.optim.SGD(_m_ref.parameters(), lr=0.1)
_opt_ref.zero_grad()
_ref_running = 0.0
for _xb, _yb in _batches:
    _l = F.mse_loss(_m_ref(_xb), _yb) / _accum
    _l.backward(); _ref_running += _l.item()
_opt_ref.step()

_avg = train_step_accum(_m, _batches, _opt, _accum)
require_not_none("TODO-2 train_step_accum", _avg)
require_true("TODO-2 平均 loss 正确",
             abs(_avg - _ref_running) < 1e-6,
             f"应等于各 micro loss/accum 之和 {_ref_running:.6f}")
# step 后参数应与参考实现一致
_p = torch.cat([p.flatten() for p in _m.parameters()])
_p_ref = torch.cat([p.flatten() for p in _m_ref.parameters()])
require_close("TODO-2 step 后参数与参考一致", _p, _p_ref)
print(f"train_step_accum OK：平均 loss={_avg:.5f}，且累积更新后参数正确")


# ============================================================
section("TODO-3：梯度裁剪 + step  clip_and_step")
# ============================================================
# 假设 loss 已经 backward（梯度已在 .grad 里）。本函数：
#   1. 用 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) 裁剪，
#      它【就地】缩放梯度并返回【裁剪前】的总范数 total_norm
#   2. opt.step() 更新参数
#   3. 返回 total_norm（转成 float）
#
# 提示：
#   total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#   opt.step()
#   return float(total_norm)


def clip_and_step(model, opt, max_norm):
    # TODO-3: 裁剪梯度范数后 step，返回裁剪前总范数(float)
    return None


torch.manual_seed(2)
_m = make_model()
_opt = torch.optim.SGD(_m.parameters(), lr=0.01)
_xs, _ys = make_data()
_loss = F.mse_loss(_m(_xs), _ys)
_opt.zero_grad()
_loss.backward()
# 先记下裁剪前的真实总范数用于对照
_expected_norm = math.sqrt(sum(p.grad.pow(2).sum().item()
                               for p in _m.parameters() if p.grad is not None))
_max_norm = 0.1
_ret_norm = clip_and_step(_m, _opt, _max_norm)
require_not_none("TODO-3 clip_and_step", _ret_norm)
require_true("TODO-3 返回 float", isinstance(_ret_norm, float), "应返回 float")
require_true("TODO-3 返回的是裁剪【前】总范数",
             abs(_ret_norm - _expected_norm) < 1e-4,
             f"应约等于裁剪前总范数 {_expected_norm:.4f}")
# 裁剪后梯度范数应被压到 max_norm 附近（若原范数 > max_norm）
_after = math.sqrt(sum(p.grad.pow(2).sum().item()
                       for p in _m.parameters() if p.grad is not None))
require_true("TODO-3 裁剪后范数被压到 max_norm 附近",
             _after <= _max_norm + 1e-4,
             f"裁剪后范数 {_after:.4f} 应 <= max_norm {_max_norm}")
print(f"clip_and_step OK：裁剪前范数={_ret_norm:.4f} → 裁剪后={_after:.4f}（<= {_max_norm}）")


# ============================================================
section("TODO-4：checkpoint 保存/加载 save_ckpt / load_ckpt")
# ============================================================
# save_ckpt(path, model, opt, step)：把 {model, opt, step} 存成 .pt
#   torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)
# load_ckpt(path, model, opt)：从盘里恢复 model/opt 的 state_dict，返回 step(int)
#   ckpt = torch.load(path)
#   model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"])
#   return ckpt["step"]


def save_ckpt(path, model, opt, step):
    # TODO-4a: 保存 {model, opt, step} 到 path
    return None


def load_ckpt(path, model, opt):
    # TODO-4b: 从 path 恢复 model/opt，返回 step(int)
    return None


torch.manual_seed(3)
_m = make_model()
_opt = torch.optim.AdamW(_m.parameters(), lr=1e-2)
_xs, _ys = make_data()
# 训练几步，让参数与优化器状态都非平凡
for _ in range(5):
    _l = F.mse_loss(_m(_xs), _ys)
    _opt.zero_grad(); _l.backward(); _opt.step()

_ckpt_path = os.path.join(SCRIPT_DIR, "_ckpt_selfwrite_demo.pt")
save_ckpt(_ckpt_path, _m, _opt, step=5)
require_true("TODO-4 checkpoint 文件已生成", os.path.exists(_ckpt_path),
             "save_ckpt 应在磁盘写出 .pt 文件")

# 新建一套对象，加载后应与原模型参数一致、step 一致
_m2 = make_model()
_opt2 = torch.optim.AdamW(_m2.parameters(), lr=1e-2)
_step = load_ckpt(_ckpt_path, _m2, _opt2)
require_not_none("TODO-4 load_ckpt 返回 step", _step)
require_true("TODO-4 step 恢复一致", _step == 5, f"应恢复为 5，实际 {_step}")
_p1 = torch.cat([p.flatten() for p in _m.parameters()])
_p2 = torch.cat([p.flatten() for p in _m2.parameters()])
require_close("TODO-4 加载后参数一致", _p2, _p1)
# 校验完删掉临时文件，保持目录干净
if os.path.exists(_ckpt_path):
    os.remove(_ckpt_path)
print(f"save_ckpt/load_ckpt OK：step={_step} 恢复，参数一致，临时文件已删除")


# ============================================================
section("TODO-5：early stopping  EarlyStopper(patience)")
# ============================================================
# 实现 step(val_loss) -> bool（是否应当停止）。逻辑：
#   * 维护历史最低 best（初始 +inf）和连续未改善次数 bad（初始 0）
#   * 若 val_loss < best：刷新 best，bad 清零，返回 False
#   * 否则：bad += 1；若 bad >= patience 返回 True，否则 False
#
# 提示（在 __init__ 里 self.best=float("inf"); self.bad=0; self.patience=patience）：
#   def step(self, val_loss):
#       if val_loss < self.best:
#           self.best = val_loss; self.bad = 0; return False
#       self.bad += 1
#       return self.bad >= self.patience


class EarlyStopper:
    def __init__(self, patience=3):
        self.patience = patience
        self.best = float("inf")
        self.bad = 0

    def step(self, val_loss):
        # TODO-5: 更新 best/bad，返回是否应停止(bool)
        return None


_stopper = EarlyStopper(patience=3)
# 先降后升的曲线：0.9,0.7,0.5(best) 之后连续 3 次不改善应触发
_curve = [0.9, 0.7, 0.5, 0.6, 0.6, 0.6]
_results = [_stopper.step(v) for v in _curve]
require_not_none("TODO-5 step 返回", _results[0])
require_true("TODO-5 改善时不停",
             _results[0] is False and _results[1] is False and _results[2] is False,
             "val loss 在下降阶段不应触发停止")
require_true("TODO-5 未到 patience 不停",
             _results[3] is False and _results[4] is False,
             "连续未改善 1、2 次（< patience=3）不应停")
require_true("TODO-5 连续 patience 次未改善触发停止",
             _results[5] is True,
             "连续未改善达到 patience=3 应返回 True")
require_true("TODO-5 best 记录正确", abs(_stopper.best - 0.5) < 1e-9,
             "best 应停在历史最低 0.5")
print(f"EarlyStopper OK：best={_stopper.best:.2f}，在第 6 次评估触发停止")


# ============================================================
section("全部 TODO 校验通过 ✓")
# ============================================================
print("""
你已经手写完成：
  1. evaluate：eval() + no_grad 评估，并恢复 train()，不污染后续训练
  2. train_step_accum：梯度累积一步（loss/accum、累加 backward、accum 后 step+zero）
  3. clip_and_step：梯度裁剪 clip_grad_norm_ 后 step，返回裁剪前总范数
  4. save_ckpt/load_ckpt：保存/加载 model+opt+step，断点续训的核心
  5. EarlyStopper：patience 监控 val loss，连续不改善则提前停止

复盘三问：
  * 为什么评估要 model.eval() + torch.no_grad()？两者各管什么（Dropout/BN vs 计算图）？
  * 梯度累积里如果忘了把 loss 除以 accum_steps 会怎样？等效 lr 变成多少倍？
  * checkpoint 只存 model.state_dict() 够吗？不存 optimizer 续训会出什么问题？
""")
