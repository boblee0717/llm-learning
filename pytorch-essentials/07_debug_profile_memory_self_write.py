"""
======================================================
PyTorch 专项 / 第 7 课（自写版）：调试 / 显存 / 算力
======================================================

用法：
1. 运行：python 07_debug_profile_memory_self_write.py
2. 按 TODO-1 到 TODO-5 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错
   （没填的 TODO 会返回 None，校验提示「未实现」，这是正常的）

目标：
- count_params：统计模型的总参数量与可训练参数量
- estimate_memory_gb：用「每参数 16 字节」估训练显存（GB）
- estimate_flops：用经验公式 C≈6ND 估训练总浮点运算量
- has_nan_or_inf：检测张量是否含 nan 或 inf（loss 排查急救包）
- safe_log：给输入加 eps 再 log，避免 log(0)=-inf

对照：本课主课 07_debug_profile_memory.py，以及 phase1 第 4 课（显存账 / 算力）。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")  # ValidationError 走 stderr，也要防乱码

import torch
import torch.nn as nn

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
section("TODO-1：参数量统计 count_params")
# ============================================================
# 输入一个 nn.Module，返回 (total, trainable)：
#   total     = 所有参数的元素个数之和
#   trainable = 其中 requires_grad=True 的参数元素个数之和
# 这是「这个模型多大」「冻结了多少」的第一手数据（LoRA / 迁移学习常看）。
#
# 提示：
#   p.numel() 给出一个参数张量的元素个数。
#   model.parameters() 遍历所有参数；用 p.requires_grad 筛可训练的。
#   total = sum(p.numel() for p in model.parameters())


def count_params(model):
    # TODO-1: 返回 (total, trainable)
    return None


_m = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
# 手算期望值：Linear(10,20)=10*20+20=220；Linear(20,5)=20*5+5=105；合计 325
_expect_total = 10 * 20 + 20 + 20 * 5 + 5
_res = count_params(_m)
require_not_none("TODO-1 count_params", _res)
require_true("TODO-1 应返回二元组", isinstance(_res, tuple) and len(_res) == 2,
             "返回 (total, trainable)")
require_true("TODO-1 total 正确", _res[0] == _expect_total,
             f"total 应为 {_expect_total}，得到 {_res[0]}")
require_true("TODO-1 全部可训练时 trainable==total", _res[1] == _expect_total,
             "默认参数都 requires_grad=True，trainable 应等于 total")
# 冻结第一层后 trainable 应下降
for _p in _m[0].parameters():
    _p.requires_grad = False
_res2 = count_params(_m)
require_true("TODO-1 冻结后 total 不变", _res2[0] == _expect_total, "total 不该变")
require_true("TODO-1 冻结后 trainable 下降", _res2[1] == _expect_total - 220,
             f"冻结 Linear(10,20) 的 220 个参数后 trainable 应为 {_expect_total - 220}")
print("count_params OK：total=%d, 冻结后 trainable=%d" % (_res2[0], _res2[1]))


# ============================================================
section("TODO-2：训练显存估算 estimate_memory_gb")
# ============================================================
# 输入参数量 num_params 与 bytes_per_param（默认 16），返回训练态显存估算（GB）。
# 口径：用十进制 1e9（和论文/厂商标称的 GB 一致），即 num_params*bytes/1e9。
# 为什么默认 16：fp32 权重4 + 梯度4 + Adam 一阶动量4 + 二阶动量4 = 16 字节/参数。
#（这部分不含前向激活；纯推理可传 bytes_per_param=4。）
#
# 提示：return num_params * bytes_per_param / 1e9


def estimate_memory_gb(num_params, bytes_per_param=16):
    # TODO-2: 返回训练显存估算（GB），用 /1e9 口径
    return None


_gb = estimate_memory_gb(int(7e9))           # 7B 训练态
require_not_none("TODO-2 estimate_memory_gb", _gb)
require_true("TODO-2 7B 训练显存≈112GB", abs(_gb - 112.0) < 1e-6,
             f"7e9*16/1e9 应为 112.0，得到 {_gb}")
_gb_infer = estimate_memory_gb(int(7e9), bytes_per_param=4)   # 纯推理
require_true("TODO-2 推理口径(4字节)≈28GB", abs(_gb_infer - 28.0) < 1e-6,
             f"7e9*4/1e9 应为 28.0，得到 {_gb_infer}")
print("estimate_memory_gb OK：7B 训练≈%.1fGB，纯推理≈%.1fGB" % (_gb, _gb_infer))


# ============================================================
section("TODO-3：训练算力估算 estimate_flops")
# ============================================================
# 经验公式 C ≈ 6 * N * D：N=参数量，D=训练 token 总数。
# 直觉：每 token 前向 2N + 反向 4N = 6N FLOPs，乘 D 个 token。
# 这是规划训练算力预算（要多少卡多少天）的核心公式。
#
# 提示：return 6.0 * num_params * num_tokens


def estimate_flops(num_params, num_tokens):
    # TODO-3: 返回 6*N*D
    return None


_c = estimate_flops(int(1.24e8), int(3e11))   # GPT-2 small / 300B token
require_not_none("TODO-3 estimate_flops", _c)
_expect_c = 6.0 * 1.24e8 * 3e11
require_true("TODO-3 6ND 数值正确", abs(_c - _expect_c) / _expect_c < 1e-9,
             f"应为 {_expect_c:.3e}，得到 {_c:.3e}")
print("estimate_flops OK：GPT-2(124M)/300B tok ≈ %.2e FLOPs" % _c)


# ============================================================
section("TODO-4：nan / inf 检测 has_nan_or_inf")
# ============================================================
# 输入一个张量，只要含有 nan 或 inf（±inf 都算）就返回 True，否则 False。
# 这是训练中 loss 炸掉时的第一道排查工具。
#
# 提示：
#   torch.isnan(t) / torch.isinf(t) 逐元素返回布尔张量。
#   用 .any() 聚合，再 | 组合，最后 .item() 转成 Python bool。
#   return bool((torch.isnan(t) | torch.isinf(t)).any().item())


def has_nan_or_inf(t):
    # TODO-4: 含 nan 或 inf 返回 True，否则 False
    return None


_good = torch.tensor([1.0, 2.0, -3.0])
_with_nan = torch.tensor([1.0, float("nan"), 3.0])
_with_inf = torch.tensor([1.0, float("inf"), 3.0])
_with_ninf = torch.tensor([1.0, -float("inf"), 3.0])
require_true("TODO-4 正常张量应为 False", has_nan_or_inf(_good) is False,
             "无 nan/inf 时应返回 False（注意是 bool，不是 None）")
require_true("TODO-4 含 nan 应为 True", has_nan_or_inf(_with_nan) is True, "")
require_true("TODO-4 含 +inf 应为 True", has_nan_or_inf(_with_inf) is True, "")
require_true("TODO-4 含 -inf 应为 True", has_nan_or_inf(_with_ninf) is True, "")
print("has_nan_or_inf OK：正常=False，含 nan/±inf 均=True")


# ============================================================
section("TODO-5：数值稳定的对数 safe_log")
# ============================================================
# 输入非负张量 t（可能含 0），返回 log(t + eps)，eps 默认 1e-8。
# 目的：避免 log(0) = -inf（softmax 概率为 0 时取 log 是经典 nan/inf 来源）。
# 校验：结果不能含 inf；且对远离 0 的值，safe_log 要≈普通 log。
#
# 提示：return torch.log(t + eps)


def safe_log(t, eps=1e-8):
    # TODO-5: 返回 log(t + eps)，避免 log(0)=-inf
    return None


_p = torch.tensor([0.0, 0.5, 1.0])
_sl = safe_log(_p)
require_not_none("TODO-5 safe_log", _sl)
require_shape("TODO-5 形状不变", _sl, (3,))
require_true("TODO-5 不能产生 inf", not torch.isinf(_sl).any().item(),
             "加了 eps 后 log(0+eps) 应是有限值，不能再有 -inf")
require_true("TODO-5 不能产生 nan", not torch.isnan(_sl).any().item(), "")
# 对 0.5 / 1.0 这种远离 0 的值，safe_log ≈ 普通 log
require_close("TODO-5 远离0处≈普通log", _sl[1:], torch.log(_p[1:]), atol=1e-6)
print("safe_log OK：log(0+eps)=%.3f（有限值，不再是 -inf）" % _sl[0].item())


# ============================================================
section("全部 TODO 校验通过 ✓")
# ============================================================
print("""
你已经手写完成：
  1. count_params：统计总参数量与可训练参数量（冻结后 trainable 下降）
  2. estimate_memory_gb：每参数 16 字节估训练显存（7B≈112GB）
  3. estimate_flops：C≈6ND 估训练总算力
  4. has_nan_or_inf：nan/inf 检测，loss 排查急救包
  5. safe_log：加 eps 避免 log(0)=-inf，数值稳定

复盘三问：
  * 训练态「每参数 16 字节」是哪 4 个 4 字节加起来的？纯推理为什么是 4 字节？
  * C≈6ND 里的 6 怎么来的（前向 / 反向各占多少）？
  * 除了 log(0)，还有哪些操作容易制造 nan/inf？怎么用 detect_anomaly 定位？
""")
