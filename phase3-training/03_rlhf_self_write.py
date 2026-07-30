"""
======================================================
第 3 课（自写版）：RLHF —— 只手写 DPO loss
======================================================

⚠️ 定位：RLHF 这课是「📖 精读」，PPO / 奖励模型不从零手搓（性价比太低）。
   真正值得亲手推一遍的只有 DPO loss —— 它用一个损失函数替代了整个
   「训练奖励模型 + PPO 强化学习」流程。本自写版就聚焦这一个 TODO。

用法：
1. 运行：python3 03_rlhf_self_write.py
2. 补全 TODO-1（dpo_loss），靠 require_xxx 即时纠错
3. 配合主课 03_rlhf.py 的 Part 5 一起看

DPO 损失：
  L = -log σ( β * [ (logπ(y_w) - logπ_ref(y_w)) - (logπ(y_l) - logπ_ref(y_l)) ] )

  y_w = 更好的回答(chosen)，y_l = 更差的回答(rejected)
  π = 当前策略，π_ref = 参考模型(SFT 后冻结)，β = 温度
  直觉：让「好回答相对参考模型的对数概率」比「坏回答」高得越多，loss 越小。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")

import math
import torch
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
    if isinstance(actual, torch.Tensor):
        actual = actual.item()
    if abs(actual - expected) > atol:
        raise ValidationError(f"{name} 数值不对：actual={actual}, expected={expected}")


# ============================================================
section("TODO-1：手写 DPO loss")
# ============================================================
# 输入是 4 组「序列对数概率之和」（已经 sum 过 token 维，形状都是 (batch,)）：
#   policy_chosen   = logπ(y_w)        当前策略给 chosen 的 logprob
#   policy_rejected = logπ(y_l)        当前策略给 rejected 的 logprob
#   ref_chosen      = logπ_ref(y_w)    参考模型给 chosen 的 logprob（no_grad）
#   ref_rejected    = logπ_ref(y_l)    参考模型给 rejected 的 logprob（no_grad）
#
# TODO-1：
#   chosen_reward   = beta * (policy_chosen   - ref_chosen)
#   rejected_reward = beta * (policy_rejected - ref_rejected)
#   loss = -logσ(chosen_reward - rejected_reward) 的均值   # 用 F.logsigmoid
#   acc  = (chosen_reward > rejected_reward) 的比例         # .float().mean()
#   返回 (loss, acc)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    # TODO-1
    chosen_reward = beta * (policy_chosen - ref_chosen) 
    rejected_reward = beta * (policy_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    acc = (chosen_reward > rejected_reward).float().mean()
    return loss, acc


# ---- 校验 1：策略 == 参考（训练刚开始）→ 两边 reward 都是 0 → loss = -log σ(0) = log 2 ----
_z = torch.zeros(8)
_loss0, _acc0 = dpo_loss(_z.clone(), _z.clone(), _z.clone(), _z.clone(), beta=0.1)
require_close("TODO-1 策略=参考时 loss=log2", _loss0, math.log(2), atol=1e-4)
print(f"TODO-1 校验1 OK：策略=参考 → loss={_loss0.item():.4f} ≈ log2={math.log(2):.4f}")

# ---- 校验 2：chosen 明显更受偏好 → loss 应小于 log2，且偏好准确率 = 1 ----
_pc = torch.full((8,), 1.0)   # 策略提高了 chosen 的 logprob
_pr = torch.full((8,), -1.0)  # 策略压低了 rejected 的 logprob
_loss1, _acc1 = dpo_loss(_pc, _pr, _z.clone(), _z.clone(), beta=0.1)
require_true("TODO-1 偏好正确时 loss 更小", _loss1.item() < _loss0.item(),
             "chosen 占优时 loss 应低于 log2")
require_close("TODO-1 偏好准确率=1", _acc1, 1.0, atol=1e-6)
print(f"TODO-1 校验2 OK：chosen 占优 → loss={_loss1.item():.4f} < log2，acc=100%")

# ---- 校验 3：方向反了（rejected 更受偏好）→ loss 应大于 log2，准确率 = 0 ----
_loss2, _acc2 = dpo_loss(_pr, _pc, _z.clone(), _z.clone(), beta=0.1)
require_true("TODO-1 偏好相反时 loss 更大", _loss2.item() > _loss0.item(),
             "搞反偏好时 loss 应高于 log2")
require_close("TODO-1 偏好准确率=0", _acc2, 0.0, atol=1e-6)
print(f"TODO-1 校验3 OK：方向反 → loss={_loss2.item():.4f} > log2，acc=0%")

# ---- 校验 4：β 越大，同样的 margin 推得越狠（loss 偏离 log2 更多）----
_loss_lo, _ = dpo_loss(_pc, _pr, _z.clone(), _z.clone(), beta=0.1)
_loss_hi, _ = dpo_loss(_pc, _pr, _z.clone(), _z.clone(), beta=1.0)
require_true("TODO-1 β 越大 loss 越小", _loss_hi.item() < _loss_lo.item(),
             "正确偏好下，β 放大 margin，loss 应更小")
print(f"TODO-1 校验4 OK：β=0.1 loss={_loss_lo.item():.4f} → β=1.0 loss={_loss_hi.item():.4f}")
print()

print("=" * 60)
print("全部通过！你已手推 DPO loss —— 一个损失函数替代了 RM + PPO。")
print("RLHF 其余部分（SFT / 奖励模型 / PPO）按主课 03_rlhf.py 读懂即可，不必手搓。")
print("=" * 60)
