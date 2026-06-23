"""
======================================================
PyTorch 专项 / 第 3 课（自写版）：nn.Module 机制
======================================================

用法：
1. 运行：python 03_nn_module_self_write.py
2. 按 TODO-1 到 TODO-6 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错

目标：
- 用 nn.Parameter 从零实现线性层 forward
- 用 nn.Parameter 从零实现 LayerNorm（带可学习 γ/β），和 nn.LayerNorm 对拍
- 用 register_buffer 登记因果掩码（理解 buffer ≠ parameter）
- 用 state_dict / load_state_dict 复制权重
- 统计可训练参数量、冻结参数（LoRA / 微调基础）

对照：本课主课 03_nn_module.py，phase2 第 3/5 课（LayerNorm / 多头注意力）。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")

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
section("TODO-1：从零实现线性层 MyLinear.forward")
# ============================================================
# __init__ 已用 nn.Parameter 建好 weight (out, in) 和 bias (out,)。
# 你来写 forward：y = x @ Wᵀ + b（和 nn.Linear 一致）。
#
# 提示：x (B, in)，weight (out, in)，需要 weight.t() 把它转成 (in, out)。
#   return x @ self.weight.t() + self.bias


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # TODO-1: 实现 y = x @ Wᵀ + b
        return None


_lin = MyLinear(4, 3)
_x = torch.randn(2, 4)
_out = _lin(_x)
require_shape("TODO-1 输出形状", _out, (2, 3))
# 和 PyTorch 官方 F.linear 对拍
_ref = torch.nn.functional.linear(_x, _lin.weight, _lin.bias)
require_close("TODO-1 数值", _out, _ref)
# weight/bias 应被登记为可训练参数
require_true("TODO-1 参数被登记",
             {n for n, _ in _lin.named_parameters()} == {"weight", "bias"},
             "weight/bias 应是 nn.Parameter 并出现在 named_parameters 里")
print("MyLinear OK：从零实现的线性层和 F.linear 数值一致")


# ============================================================
section("TODO-2：从零实现 LayerNorm（带可学习 γ/β）")
# ============================================================
# __init__ 已建好 gamma（初始化为全 1）和 beta（全 0），都是 nn.Parameter。
# forward 对最后一维做归一化：
#   mean = x.mean(-1, keepdim=True)
#   var  = x.var(-1, keepdim=True, unbiased=False)   # 注意 unbiased=False（和 nn.LayerNorm 一致）
#   return (x - mean) / sqrt(var + eps) * gamma + beta


class ScratchLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # TODO-2: 实现 LayerNorm 前向
        return None


_ln = ScratchLayerNorm(8)
_xln = torch.randn(4, 8)
_ln_out = _ln(_xln)
require_shape("TODO-2 输出形状", _ln_out, (4, 8))
# 归一化后每行均值≈0、标准差≈1（γ=1,β=0 时）
require_close("TODO-2 归一化后均值≈0", _ln_out.mean(-1), torch.zeros(4), atol=1e-4)
# 和官方 nn.LayerNorm 对拍
_ref_ln = torch.nn.functional.layer_norm(_xln, (8,), _ln.gamma, _ln.beta, 1e-5)
require_close("TODO-2 与 nn.LayerNorm 数值一致", _ln_out, _ref_ln)
print("ScratchLayerNorm OK：和官方 nn.LayerNorm 数值一致")


# ============================================================
section("TODO-3：用 register_buffer 登记因果掩码")
# ============================================================
# 在 __init__ 里把因果掩码登记成 buffer（不是 Parameter！）。
# 因为掩码是常量、不需要训练，但要随模型保存、随 .to(device) 搬走。
#
# 提示：先建掩码 mask = torch.triu(torch.ones(context_len, context_len), diagonal=1)
#   再 self.register_buffer("mask", mask)


class MaskedModule(nn.Module):
    def __init__(self, context_len, n_embd):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_embd)   # 这是参数
        # TODO-3: 把因果掩码登记为名为 "mask" 的 buffer
        #   提示：mask = torch.triu(torch.ones(context_len, context_len), diagonal=1)
        #         然后 self.register_buffer("mask", mask)
        pass  # ← 实现后删掉这行

    def forward(self, x):
        return self.proj(x)


_mm = MaskedModule(context_len=8, n_embd=16)
_buf_names = {n for n, _ in _mm.named_buffers()}
_par_names = {n for n, _ in _mm.named_parameters()}
require_true("TODO-3 mask 是 buffer", "mask" in _buf_names,
             "应该用 register_buffer 登记 mask")
require_true("TODO-3 mask 不是 parameter", "mask" not in _par_names,
             "mask 是常量，不能是 Parameter（否则优化器会去更新它）")
require_true("TODO-3 mask 在 state_dict 里", "mask" in _mm.state_dict(),
             "buffer 应出现在 state_dict 中（随模型保存）")
require_shape("TODO-3 mask 形状", _mm.mask, (8, 8))
print("MaskedModule OK：mask 是 buffer（在 state_dict、不在 parameters）")


# ============================================================
section("TODO-4：用 state_dict 复制权重 copy_weights")
# ============================================================
# 把 src 模型的全部权重复制到 dst 模型（两者结构相同）。
# 这正是加载 checkpoint 的核心：dst.load_state_dict(src.state_dict())。
# 返回 dst。
#
# 提示：dst.load_state_dict(src.state_dict()); return dst


def copy_weights(src, dst):
    # TODO-4: 把 src 的权重复制进 dst，返回 dst
    return None


_src = MyLinear(4, 3)
_dst = MyLinear(4, 3)
require_true("前置检查：复制前两模型权重不同",
             not torch.equal(_src.weight, _dst.weight), "随机初始化应不同")
_dst = copy_weights(_src, _dst)
require_not_none("TODO-4 copy_weights", _dst)
require_close("TODO-4 weight 已复制", _dst.weight, _src.weight)
require_close("TODO-4 bias 已复制", _dst.bias, _src.bias)
print("copy_weights OK：state_dict 复制权重成功（这就是加载 checkpoint 的原理）")


# ============================================================
section("TODO-5：统计可训练参数量 count_trainable")
# ============================================================
# 返回 model 中所有 requires_grad=True 的参数的元素总数。
#
# 提示：sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_trainable(model):
    # TODO-5: 返回可训练参数总数（int）
    return None


_net = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 2))
_cnt = count_trainable(_net)
require_not_none("TODO-5 count_trainable", _cnt)
# 4*16+16 + 16*2+2 = 80 + 34 = 114
require_true("TODO-5 参数量正确", _cnt == 114,
             f"应为 114（80 + 34），你算出 {_cnt}")
print(f"count_trainable OK：{_cnt} 个可训练参数")


# ============================================================
section("TODO-6：冻结模块参数 freeze_all")
# ============================================================
# 把 module 里所有参数的 requires_grad 设为 False（优化器将不再更新它们），
# 返回被冻结的参数个数（int）。这是 LoRA / 微调里「冻结预训练权重」的基础操作。
#
# 提示：遍历 module.parameters()，把每个 p.requires_grad = False，
#   并累加 p.numel()。


def freeze_all(module):
    # TODO-6: 冻结所有参数，返回被冻结的参数总数
    return None


_freeze_net = nn.Linear(4, 3)   # 4*3 + 3 = 15
_frozen = freeze_all(_freeze_net)
require_not_none("TODO-6 freeze_all", _frozen)
require_true("TODO-6 冻结数正确", _frozen == 15, f"应为 15，你算出 {_frozen}")
require_true("TODO-6 确实都被冻结了",
             all(not p.requires_grad for p in _freeze_net.parameters()),
             "所有参数 requires_grad 应为 False")
require_true("TODO-6 冻结后可训练参数为 0", count_trainable(_freeze_net) == 0)
print(f"freeze_all OK：冻结了 {_frozen} 个参数，可训练参数归零")


# ============================================================
section("全部 TODO 校验通过 ✓")
# ============================================================
print("""
你已经手写完成：
  1. 用 nn.Parameter 从零实现线性层 forward
  2. 用 nn.Parameter 从零实现 LayerNorm（带可学习 γ/β），和官方对拍
  3. 用 register_buffer 登记因果掩码（buffer ≠ parameter）
  4. 用 state_dict / load_state_dict 复制权重（= 加载 checkpoint 的原理）
  5. 统计可训练参数量
  6. 冻结参数（LoRA / 微调基础）

复盘三问：
  * nn.Parameter 和普通 tensor 放进 Module 有什么不同？
  * 因果掩码为什么用 buffer 而不是 Parameter？两者在 state_dict / parameters 里的差别？
  * 推理前为什么要 model.eval()？它影响了哪些层？
""")
