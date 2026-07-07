"""
======================================================
第 2 课（自写版）：模型量化
======================================================

用法：
1. 运行：python3 02_quantization_self_write.py
2. 按 TODO-1 到 TODO-6 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错

目标（对照主课 02_quantization.py）：
- 亲手推对称量化：scale = max|x|/qmax，round(x/scale) 再还原
- 亲手推非对称量化：多一个 zero_point，把 [min,max] 铺满 [0, qmax]
- 逐通道量化：每行独立 scale，对「不同通道范围差异大」的权重精度更高
- 量化感知训练 (QAT) 的 STE：前向量化、反向梯度直通
- 手算不同位数的存储字节数

关键直觉：浮点权重映射到整数只占 1/4~1/8 空间，而模型对这点精度损失非常鲁棒。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")

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
section("TODO-1 / TODO-2：对称量化与反量化")
# ============================================================
# 对称量化把浮点映射到 [-qmax, qmax]，0 精确映射到 0。
#   qmax  = 2**(num_bits-1) - 1          # 8 位 → 127
#   scale = max(|x|) / qmax
#   q     = clamp(round(x / scale), -qmax, qmax)   # 存成 int8
#   还原： x ≈ q * scale
#
# TODO-1：实现 symmetric_quantize，返回 (q_int8, scale 标量张量)
# TODO-2：实现 symmetric_dequantize，返回 q.float() * scale


def symmetric_quantize(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -qmax
    # TODO-1: 算 scale、量化、clamp、转 int8，返回 (q, scale)
    scale = x.abs().max()/qmax
    q = torch.clamp(torch.round(x / scale), qmin, qmax).to(torch.int8)
    return q, scale


def symmetric_dequantize(q, scale):
    # TODO-2: 反量化
    dq = q.float() * scale
    return dq


_x = torch.tensor([[-1.0, 0.0, 0.5, 2.0], [0.25, -0.75, 1.5, -2.0]])
_q, _s = symmetric_quantize(_x)
require_not_none("TODO-1 scale", _s)
require_shape("TODO-1 q 形状", _q, (2, 4))
require_true("TODO-1 q 是 int8", _q.dtype == torch.int8, "应 .to(torch.int8)")
require_true("TODO-1 scale = max|x|/127", abs(_s.item() - 2.0 / 127) < 1e-6,
             "max|x|=2.0，qmax=127")
require_true("TODO-1 范围内", int(_q.abs().max()) <= 127, "clamp 到 [-127,127]")
_dq = symmetric_dequantize(_q, _s)
require_shape("TODO-2 反量化形状", _dq, (2, 4))
require_true("TODO-2 0 精确映射到 0", _dq[0, 1].abs().item() < 1e-9, "对称量化下 0→0")
require_true("TODO-2 还原误差 <= scale", (_x - _dq).abs().max().item() <= _s.item() + 1e-6,
             "舍入误差不应超过一个 scale")
print(f"TODO-1/2 OK：对称量化往返，scale={_s.item():.5f}，"
      f"最大误差={(_x - _dq).abs().max().item():.5f}")
print()


# ============================================================
section("TODO-3：非对称量化（适合 ReLU 这种全非负数据）")
# ============================================================
# 非对称量化把 [min, max] 铺满 [0, qmax]，多一个 zero_point 记录「0 落在哪」。
#   qmax       = 2**num_bits - 1               # 8 位 → 255
#   scale      = (max - min) / qmax
#   zero_point = round(-min / scale)，clamp 到 [0,qmax]，转 int32
#   q          = clamp(round(x/scale) + zero_point, 0, qmax)，转 uint8
#   还原： x ≈ (q - zero_point) * scale
#
# TODO-3：实现 asymmetric_quantize，返回 (q_uint8, scale, zero_point_int32)


def asymmetric_quantize(x, num_bits=8):
    qmax = 2 ** num_bits - 1
    x_min, x_max = x.min(), x.max()
    # TODO-3: 算 scale、zero_point，量化，返回 (q, scale, zero_point)
    scale = (x_max - x_min) / qmax
    zero_point = torch.round(-x_min/scale)
    q = torch.clamp(torch.round(x/scale) + zero_point, 0, qmax).to(torch.uint8)
    return q, scale, zero_point


def asymmetric_dequantize(q, scale, zero_point):
    return (q.float() - zero_point.float()) * scale


_relu = F.relu(torch.randn(4, 4))  # 全非负
_qa, _sa, _zp = asymmetric_quantize(_relu)
require_not_none("TODO-3 zero_point", _zp)
require_true("TODO-3 q 是 uint8", _qa.dtype == torch.uint8, "应 .to(torch.uint8)")
_dqa = asymmetric_dequantize(_qa, _sa, _zp)
require_close("TODO-3 还原接近原值", _dqa, _relu, atol=_sa.item() + 1e-6)
# 对全非负数据，非对称应不差于对称（铺满整个 [0,255] 利用率更高）
_qs, _ss = symmetric_quantize(_relu)
_err_sym = (_relu - symmetric_dequantize(_qs, _ss)).abs().mean()
_err_asym = (_relu - _dqa).abs().mean()
require_true("TODO-3 非对称不差于对称", _err_asym <= _err_sym + 1e-7,
             f"asym={_err_asym:.5f} 应 <= sym={_err_sym:.5f}")
print(f"TODO-3 OK：非对称误差 {_err_asym:.5f} <= 对称误差 {_err_sym:.5f}（非负数据）")
print()


# ============================================================
section("TODO-4：逐通道量化（每行独立 scale）")
# ============================================================
# 逐张量量化全矩阵共用一个 scale；若不同行数值范围差异大，小值行会被淹没。
# 逐通道量化对每一行独立算 scale → 精度更高（权重量化常用）。
#
# TODO-4：实现 per_channel_quantize（对称、按行）
#   qmax    = 2**(num_bits-1) - 1
#   abs_max = 每行的 max(|w|)，形状 (rows, 1)      # 提示 keepdim=True
#   scales  = abs_max / qmax                        # (rows, 1)
#   q       = clamp(round(w / scales), -qmax, qmax) 转 int8
#   返回 (q, scales.squeeze())  # scales 压成 (rows,)


def per_channel_quantize(weight, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    # TODO-4
    abs_max = weight.abs().max(dim=1, keepdim=True)[0]
    scales = abs_max / qmax
    q = torch.clamp(torch.round(weight/scales), -qmax, qmax).to(torch.int8)
    return q, scales.squeeze()


_w = torch.randn(4, 8)
_w[0] *= 10.0    # 第 0 行范围大
_w[3] *= 0.01    # 第 3 行范围小
_qc, _sc = per_channel_quantize(_w)
require_shape("TODO-4 q 形状", _qc, (4, 8))
require_shape("TODO-4 scales 形状", _sc, (4,))
_dq_chan = _qc.float() * _sc.unsqueeze(1)
_err_chan = (_w - _dq_chan).abs().mean()
_qt, _st = symmetric_quantize(_w)
_err_tensor = (_w - symmetric_dequantize(_qt, _st)).abs().mean()
require_true("TODO-4 逐通道优于逐张量", _err_chan < _err_tensor,
             f"逐通道 {_err_chan:.5f} 应明显小于逐张量 {_err_tensor:.5f}")
print(f"TODO-4 OK：逐通道误差 {_err_chan:.5f} < 逐张量 {_err_tensor:.5f}"
      f"（精度提升 {_err_tensor / _err_chan:.1f}x）")
print()


# ============================================================
section("TODO-5：QAT 的 STE —— 前向量化、反向梯度直通")
# ============================================================
# 量化感知训练里用「伪量化」：前向把权重量化再反量化（注入量化噪声），
# 但 round 不可导，反向用 Straight-Through Estimator (STE)：梯度原样穿过。
#
# forward 已给好。TODO-5：实现 backward——
#   返回 (grad_output, None)
#   第一个对应输入 x（梯度直通），第二个对应 num_bits（无梯度，返回 None）。


class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8):
        qmax = 2 ** (num_bits - 1) - 1
        scale = x.abs().max() / qmax
        return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale

    @staticmethod
    def backward(ctx, grad_output):
        # TODO-5: STE，返回 (grad_output, None)
        return grad_output, None


_xq = torch.randn(5, requires_grad=True)
_y = FakeQuantize.apply(_xq)
_y.sum().backward()
require_not_none("TODO-5 梯度", _xq.grad)
require_close("TODO-5 梯度直通（sum 的梯度应全为 1，STE 原样穿过 round）",
              _xq.grad, torch.ones(5))
print("TODO-5 OK：STE 让梯度绕过不可导的 round，原样回传")
print()


# ============================================================
section("TODO-6：手算不同位数的存储字节数")
# ============================================================
# TODO-6：返回 numel 个元素、每个 num_bits 位时占用的字节数（int）。
#   提示：numel * num_bits / 8，用 int(...) 取整


def quantized_bytes(numel, num_bits):
    # TODO-6
    return int(numel * num_bits / 8)


require_true("TODO-6 FP32 = numel*4", quantized_bytes(1_000_000, 32) == 4_000_000)
require_true("TODO-6 INT8 = numel*1", quantized_bytes(1_000_000, 8) == 1_000_000)
require_true("TODO-6 INT4 = numel/2", quantized_bytes(1_000_000, 4) == 500_000)
_n70b = 70_000_000_000
print("TODO-6 OK：LLaMA-70B 存储估算")
for bits, name in [(32, "FP32"), (16, "FP16"), (8, "INT8"), (4, "INT4")]:
    gb = quantized_bytes(_n70b, bits) / 1024**3
    print(f"  {name:>4}: {gb:6.1f} GB")
print()

print("=" * 60)
print("全部通过！你已亲手实现量化的核心：对称/非对称/逐通道 + STE。")
print("=" * 60)
