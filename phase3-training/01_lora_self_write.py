"""
======================================================
第 1 课（自写版）：LoRA 微调
======================================================

用法：
1. 运行：python3 01_lora_self_write.py
2. 按 TODO-1 到 TODO-5 顺序补全实现
3. 每补完一个 TODO 就运行一次，依靠 require_xxx 校验即时纠错

目标（对照主课 01_lora.py）：
- 从零实现 LoRALinear：低秩旁路 W·x + (B·A)·x·(α/r)
- 理解「B 初始化为 0 → 训练开始时 BA=0 → LoRA 是恒等旁路」这个关键设计
- 实现 apply_lora 的冻结逻辑：只训练 LoRA、冻结其余一切
- 实现 merge_lora：把低秩增量合并回原始权重（注意 (out,in) 的转置对齐）
- 手算 LoRA 参数量与压缩比

关键直觉：微调时权重变化量 ΔW 往往是低秩的，可以用两个小矩阵 B·A 近似。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")

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
section("TODO-1 / TODO-2：从零实现 LoRALinear")
# ============================================================
# 给一个已有的 nn.Linear 加 LoRA 旁路。
#
# TODO-1（在 __init__ 里）：
#   a) 冻结 original 的 weight（如果有 bias 也冻结）：requires_grad = False
#   b) self.scaling = alpha / rank
#   c) self.lora_A = nn.Parameter，形状 (in_features, rank)，随机初始化后 * 0.01
#      self.lora_B = nn.Parameter，形状 (rank, out_features)，全 0
#   关键：B 初始化为 0，于是训练开始时 B·A = 0 —— LoRA 此刻是「恒等旁路」，
#         不会破坏预训练权重，训练过程中 B 才慢慢长出来。
#
# TODO-2（在 forward 里）：
#   返回 原始输出 + (x @ lora_A @ lora_B) * scaling


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # TODO-1: 冻结 original、设 scaling、建 lora_A / lora_B
        self.scaling = None
        self.lora_A = None
        self.lora_B = None

    def forward(self, x):
        # TODO-2: return 原始输出 + (x @ A @ B) * scaling
        return None


# ---- 校验 TODO-1 ----
_orig = nn.Linear(64, 48)
_ll = LoRALinear(_orig, rank=4, alpha=8.0)
require_not_none("TODO-1 scaling", _ll.scaling)
require_true("TODO-1 scaling 值", abs(_ll.scaling - 8.0 / 4) < 1e-9, "scaling 应为 alpha/rank")
require_shape("TODO-1 lora_A 形状", _ll.lora_A, (64, 4))
require_shape("TODO-1 lora_B 形状", _ll.lora_B, (4, 48))
require_close("TODO-1 lora_B 初始化为 0", _ll.lora_B.data, torch.zeros(4, 48))
require_true("TODO-1 原始权重已冻结", _ll.original.weight.requires_grad is False,
             "original.weight.requires_grad 应为 False")
require_true("TODO-1 lora_A 可训练", _ll.lora_A.requires_grad is True, "lora_A 应可训练")
print("TODO-1 OK：LoRALinear 初始化正确（B=0、原权重冻结、scaling=α/r）")

# ---- 校验 TODO-2 ----
_x = torch.randn(2, 10, 64)
_out = _ll(_x)
require_shape("TODO-2 输出形状", _out, (2, 10, 48))
# B=0 时 LoRA 是恒等旁路：输出必须等于原始 Linear 的输出
require_close("TODO-2 B=0 时等于原始输出", _out, _orig(_x), atol=1e-5)
# 把 B 改成非零，输出就应该变了
with torch.no_grad():
    _ll.lora_B.copy_(torch.randn(4, 48))
require_true("TODO-2 B≠0 时输出改变",
             not torch.allclose(_ll(_x), _orig(_x), atol=1e-4),
             "B 非零后 LoRA 旁路应改变输出")
print("TODO-2 OK：forward 旁路正确（B=0 恒等、B≠0 生效）")
print()


# ============================================================
section("TODO-3：apply_lora —— 只训练 LoRA，冻结其余一切")
# ============================================================
# 替换逻辑已给好：把模型里名字匹配 target_modules 的 nn.Linear 换成 LoRALinear，
# 并收集所有 lora_A / lora_B 到 lora_params。
#
# TODO-3：补冻结逻辑——
#   先把 model 的所有参数 requires_grad = False，
#   再把 lora_params 里每个参数 requires_grad = True。


class MiniBlock(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.ffn = nn.Linear(d, d)


class MiniNet(nn.Module):
    def __init__(self, d=64, n_layers=3):
        super().__init__()
        self.blocks = nn.ModuleList([MiniBlock(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, d)


def apply_lora(model, rank=4, alpha=1.0, target_modules=("q_proj", "v_proj")):
    lora_params = []
    for module in model.modules():
        for attr in target_modules:
            if hasattr(module, attr):
                lin = getattr(module, attr)
                if isinstance(lin, nn.Linear):
                    ll = LoRALinear(lin, rank=rank, alpha=alpha)
                    setattr(module, attr, ll)
                    lora_params.extend([ll.lora_A, ll.lora_B])

    # TODO-3: 冻结全部参数，再解冻 lora_params
    return lora_params


_net = MiniNet(d=64, n_layers=3)
_lora_params = apply_lora(_net, rank=4, alpha=8.0, target_modules=("q_proj", "v_proj"))
_trainable = sum(p.numel() for p in _net.parameters() if p.requires_grad)
_lora_total = sum(p.numel() for p in _lora_params)
require_true("TODO-3 只有 LoRA 可训练", _trainable == _lora_total,
             f"可训练参数={_trainable} 应等于 LoRA 参数={_lora_total}")
require_true("TODO-3 替换了 3 层 q + 3 层 v 共 6 个 LoRA", len(_lora_params) == 12,
             "每个 LoRALinear 贡献 A、B 两个参数，6 层 → 12 个")
print(f"TODO-3 OK：可训练 {_trainable:,} / 总 {sum(p.numel() for p in _net.parameters()):,} "
      f"（{_trainable / sum(p.numel() for p in _net.parameters()) * 100:.2f}%）")
print()


# ============================================================
section("TODO-4：merge_lora —— 把低秩增量合并回原始权重")
# ============================================================
# 训练完后，可以把 LoRA 合并回原权重，推理时零额外开销。
#
# TODO-4（在 with torch.no_grad() 里）：
#   a) delta_w = (lora_A @ lora_B) * scaling        # 形状 (in, out)
#   b) nn.Linear 权重形状是 (out, in)，所以加的是 delta_w.T：
#        module.original.weight.data += delta_w.T
#   c) 合并后把旁路关掉，避免 forward 里又算一遍：lora_B.data.zero_()


def merge_lora(model):
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                # TODO-4: 合并 delta_w.T 进 original.weight，并把 lora_B 清零
                pass
            merged += 1
    return merged


_net2 = MiniNet(d=64, n_layers=2)
_lp = apply_lora(_net2, rank=4, alpha=8.0)
# 制造非零 LoRA 权重（模拟训练后的状态），否则合并没有可观察效果
with torch.no_grad():
    for p in _lp:
        p.copy_(torch.randn_like(p) * 0.1)

_xin = torch.randn(1, 8, 64)
with torch.no_grad():
    _before = _net2.blocks[0].q_proj(_xin)  # 合并前某个 LoRA 层的输出
_n_merged = merge_lora(_net2)
with torch.no_grad():
    _after = _net2.blocks[0].q_proj(_xin)   # 合并后同一层的输出
require_true("TODO-4 合并了 4 个 LoRA 层", _n_merged == 4, "2 层 ×(q+v)=4")
require_close("TODO-4 合并前后输出一致", _after, _before, atol=1e-5)
print("TODO-4 OK：merge 后权重等价、旁路已关，输出逐元素一致（推理零开销）")
print()


# ============================================================
section("TODO-5：手算 LoRA 参数量与压缩比")
# ============================================================
# TODO-5：返回给一个 (in_features × out_features) 的 Linear 加 rank=r 的 LoRA 后，
#         新增的可训练参数量。
#   提示：lora_A 是 (in, r)，lora_B 是 (r, out) → in*r + r*out


def lora_param_count(in_features, out_features, rank):
    # TODO-5: return in_features*rank + rank*out_features
    return None


_n = lora_param_count(4096, 4096, 8)
require_not_none("TODO-5", _n)
require_true("TODO-5 d=4096,r=8 → 65536", _n == 2 * 4096 * 8, "应为 in*r + r*out")
_full = 4096 * 4096
print(f"TODO-5 OK：d=4096,r=8 全参 {_full:,} → LoRA {_n:,}，压缩比 {_full / _n:.0f}x")
print()


# ============================================================
section("收尾：用你写的 LoRA 真跑一遍微调")
# ============================================================
# 全部 TODO 通过后，下面用你实现的 LoRA 在一个迷你任务上微调，
# 观察「只训练 ~少量参数」也能把 loss 降下去。

torch.manual_seed(1)
vocab, seqlen = 16, 12


class TinyLM(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.block = MiniBlock(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.emb(x)
        h = h + self.block.v_proj(F.gelu(self.block.q_proj(h)))
        return self.head(h)


def batch():
    p = torch.randint(0, vocab, (32, seqlen))
    y = (p + 1) % vocab
    return p, y


lm = TinyLM()
lora_params = apply_lora(lm, rank=4, alpha=8.0, target_modules=("q_proj", "v_proj"))
opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=5e-3)
first = last = None
for step in range(60):
    x, y = batch()
    loss = F.cross_entropy(lm(x).view(-1, vocab), y.view(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()
    if step == 0:
        first = loss.item()
    last = loss.item()
print(f"LoRA 微调：loss {first:.3f} → {last:.3f}（只训练了少量 LoRA 参数）")

print()
print("=" * 60)
print("全部通过！你已亲手实现 LoRA 的核心：低秩旁路 / 冻结 / 合并。")
print("=" * 60)
