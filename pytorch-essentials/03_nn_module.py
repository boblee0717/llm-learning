"""
第 3 课：nn.Module 机制
=======================
phase2 第 5 课你已经用 nn.Module 搭了 GPT，但很多机制是「照着写」的。
这节课拆开 nn.Module 这个黑盒，搞清楚：参数是怎么自动被登记的、
buffer 和 parameter 有什么区别、state_dict 是什么、train/eval 模式
为什么会改变 forward 的行为。这些是读懂、保存、调试任何 PyTorch 模型的基础。

核心问题：
- nn.Parameter 和普通 tensor 有什么区别？为什么放进 Module 就自动「被训练」？
- buffer（register_buffer）和 parameter 有什么不同？因果掩码为什么是 buffer？
- state_dict 里到底存了什么？怎么保存和加载模型权重？
- model.train() / model.eval() 改变了什么？为什么 Dropout/BatchNorm 要区分？

与大模型的关系：
- 加载 HuggingFace 权重、冻结部分参数做 LoRA、保存 checkpoint，全靠这套机制。
- 看懂「哪些是可训练参数、哪些是随模型走的常量」是微调和量化的前提。

前置：phase2 第 5 课（用过 nn.Module / nn.Linear / nn.Embedding），本专项第 2 课（autograd）
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码

import torch
import torch.nn as nn


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
section("Part 1: nn.Parameter —— 会被自动登记为「可训练参数」")
# ============================================================
# nn.Parameter 是一个特殊的 tensor：它 requires_grad=True，而且一旦被赋值给
# nn.Module 的属性，就会自动出现在 model.parameters() 里（优化器据此找到要更新谁）。

class MyLinear(nn.Module):
    """从零实现 nn.Linear：y = x @ W^T + b"""
    def __init__(self, in_features, out_features):
        super().__init__()
        # 用 nn.Parameter 包起来 → 自动被登记、自动 requires_grad
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        # 对比：普通 tensor 属性不会被登记为参数
        self.not_a_param = torch.ones(3)

    def forward(self, x):
        return x @ self.weight.t() + self.bias


lin = MyLinear(4, 3)
print("可训练参数（model.parameters() 能看到的）：")
for name, p in lin.named_parameters():
    print(f"  {name:8s} shape={tuple(p.shape)}  requires_grad={p.requires_grad}")
print("not_a_param 是普通 tensor，不在 parameters 里（优化器不会动它）")
print("前向输出形状:", lin(torch.randn(2, 4)).shape)


# ============================================================
section("Part 2: buffer —— 随模型走、但不训练的常量")
# ============================================================
# 有些张量需要跟着模型一起保存、一起搬到 GPU，但【不是要学习的参数】：
# 比如因果掩码、BatchNorm 的 running_mean。用 register_buffer 登记它们。
# 区别：buffer 出现在 state_dict 里（会被保存/加载、会 .to(device)），
#       但不出现在 parameters 里（优化器不更新它）。

class AttnWithMask(nn.Module):
    def __init__(self, context_len, n_embd):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd)         # 这是参数
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1)
        self.register_buffer("mask", mask)               # 这是 buffer（常量）

    def forward(self, x):
        return x  # 这里只演示登记，不算注意力


attn = AttnWithMask(context_len=8, n_embd=16)
param_names = [n for n, _ in attn.named_parameters()]
buffer_names = [n for n, _ in attn.named_buffers()]
print("parameters（会被优化器更新）:", param_names)
print("buffers   （不更新，但随模型走）:", buffer_names)
print("mask 在 state_dict 里吗？", "mask" in attn.state_dict())
print("→ 因果掩码是常量，所以用 buffer 而不是 Parameter")


# ============================================================
section("Part 3: state_dict —— 模型权重的「序列化字典」")
# ============================================================
# state_dict 是一个 {名字: 张量} 的有序字典，包含所有 parameter 和 buffer。
# 保存/加载模型本质就是存取这个字典。

print("MyLinear 的 state_dict 内容：")
for k, v in lin.state_dict().items():
    print(f"  {k:8s} → shape {tuple(v.shape)}")

# 保存 → 加载到一个新模型，验证权重一致（实际项目用 torch.save/torch.load 写文件）
saved = {k: v.clone() for k, v in lin.state_dict().items()}
lin2 = MyLinear(4, 3)                       # 新模型，权重是随机的
print("\n加载前两模型 weight 是否相同:",
      torch.equal(lin.weight, lin2.weight))
lin2.load_state_dict(saved)                 # 把权重灌进去
print("加载后两模型 weight 是否相同:",
      torch.equal(lin.weight, lin2.weight))
print("→ checkpoint 的本质就是 torch.save(model.state_dict()) + load_state_dict")


# ============================================================
section("Part 4: train() / eval() —— 切换模块行为")
# ============================================================
# 有些层在「训练」和「推理」时行为不同：Dropout 训练时随机丢弃、推理时不丢；
# BatchNorm 训练时用 batch 统计、推理时用累计统计。
# model.train() / model.eval() 就是切换这个全局开关（self.training 标志）。

class WithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.drop(x)


m = WithDropout()
x = torch.ones(10)

m.train()                                   # 训练模式：Dropout 生效
out_train = m(x)
print("train() 模式下输出（约一半被置 0，其余放大 2 倍）:")
print("  ", out_train.tolist())

m.eval()                                    # 评估模式：Dropout 关闭
out_eval = m(x)
print("eval() 模式下输出（原样通过，全 1）:")
print("  ", out_eval.tolist())
print("→ 推理 / 验证前一定要 model.eval()，否则 Dropout 还在乱丢，结果不稳定")


# ============================================================
section("Part 5: 嵌套模块与参数统计")
# ============================================================
# Module 可以嵌套 Module（用 nn.Sequential / nn.ModuleList 装子模块），
# parameters() 会递归收集所有子模块的参数。这就是 GPT 堆 N 层 Block 的方式。

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


mlp = MLP(4, 16, 2)
total = sum(p.numel() for p in mlp.parameters())
trainable = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
print(f"MLP 总参数量: {total:,}（可训练 {trainable:,}）")
print("拆解：")
for name, p in mlp.named_parameters():
    print(f"  {name:14s} {tuple(p.shape)} = {p.numel()}")

# 冻结参数：把 requires_grad 设为 False（LoRA / 微调的基础操作）
mlp.net[0].weight.requires_grad = False
frozen = sum(p.numel() for p in mlp.parameters() if not p.requires_grad)
print(f"\n冻结第一层 weight 后，被冻结参数: {frozen}（优化器将不再更新它）")


# ============================================================
section("小结")
# ============================================================
print("""
本课关键结论：
  1. nn.Parameter = 自动登记、自动 requires_grad 的 tensor；放进 Module 就进 parameters()。
  2. buffer（register_buffer）随模型保存/搬 device，但不训练 —— 因果掩码、running_mean 用它。
  3. state_dict = {名字: 张量}（含 parameter + buffer）；save/load checkpoint 就是存取它。
  4. train()/eval() 切换 self.training；Dropout/BatchNorm 行为随之改变，推理前务必 eval()。
  5. Module 可嵌套，parameters() 递归收集；requires_grad=False 即可冻结参数（LoRA/微调基础）。

下一课：损失函数与优化器 —— 把 phase1 手写的优化器换成 torch.optim，并学 lr 调度。
""")
