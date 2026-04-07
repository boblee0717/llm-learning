"""
======================================================
第 4 课：RLHF —— 让模型变得有用且安全
======================================================

核心问题：GPT 预训练后能写诗、能翻译，但也会胡说八道、输出有害内容。
         怎么让它变得"有用、诚实、无害"？
答案：ChatGPT = GPT + SFT + RLHF

三个阶段：
  1. SFT (Supervised Fine-Tuning)
     → 用人工标注的对话数据，教模型"怎么对话"
  2. 奖励模型 (Reward Model)
     → 训练一个模型来判断"哪个回答更好"
  3. PPO (Proximal Policy Optimization)
     → 用奖励模型的信号，优化生成策略

最新趋势：DPO (Direct Preference Optimization)
  → 不需要单独训练奖励模型，直接从偏好数据优化

学习目标：
1. 理解"对齐"(Alignment) 的概念
2. 从零实现一个奖励模型
3. 理解 PPO 的核心思想
4. 从零实现 DPO

运行方式：python3 04_rlhf.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Part 1: 为什么需要对齐 (Alignment)
# ============================================================

print("=" * 60)
print("Part 1: 为什么需要对齐")
print("=" * 60)
print("""
预训练模型的问题：
  预训练目标是"预测下一个 token"，模型学会了语言能力，
  但它不知道什么是"好回答"。

  用户: "如何做蛋糕？"
  预训练模型可能回答:
    ✗ "如何做蛋糕是一个常见的问题。如何做蛋糕..."（重复废话）
    ✗ 继续问更多问题（模仿网上的问答帖子格式）
    ✗ 给出不安全的建议

  对齐后的模型回答:
    ✓ "这是做蛋糕的步骤：1. 准备材料... 2. 混合..."（有帮助）

对齐的三个标准 (来自 InstructGPT 论文):
  1. Helpful (有用) — 回答用户的问题
  2. Honest (诚实) — 不编造信息
  3. Harmless (无害) — 不输出有害内容
""")


# ============================================================
# Part 2: SFT (监督微调)
# ============================================================

print("=" * 60)
print("Part 2: SFT 监督微调")
print("=" * 60)
print("""
SFT 是对齐的第一步，用法很简单：
  1. 收集高质量的 (指令, 回答) 数据对
  2. 用标准的语言模型训练方式微调

数据格式示例：
  {"instruction": "翻译成英文：今天天气真好",
   "response": "The weather is really nice today."}

关键点：
  - 数据质量 > 数据数量
  - InstructGPT 只用了 ~13K 条 SFT 数据
  - 但每条都是人工精心编写的高质量回答
""")


class TinyLM(nn.Module):
    """一个极简语言模型，用于演示 RLHF 流程"""

    def __init__(self, vocab_size=100, d_model=32, seq_len=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos)

        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        attn_out, _ = self.attn(
            self.ln1(h), self.ln1(h), self.ln1(h), attn_mask=mask
        )
        h = h + attn_out
        h = h + self.ffn(self.ln2(h))
        return self.head(h)

    def get_log_probs(self, x, y):
        """计算给定 (input, target) 的 log 概率"""
        logits = self.forward(x)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)
        return selected


vocab_size = 100
seq_len = 16

model_sft = TinyLM(vocab_size=vocab_size, seq_len=seq_len)

n_sft_samples = 200
sft_data = []
for _ in range(n_sft_samples):
    x = torch.randint(0, vocab_size, (seq_len,))
    y = (x + 1) % vocab_size  # SFT 目标：输出 = 输入 + 1
    sft_data.append((x, y))

sft_loader = torch.utils.data.DataLoader(sft_data, batch_size=32, shuffle=True)

optimizer = torch.optim.AdamW(model_sft.parameters(), lr=1e-3)
print("SFT 训练:")
for epoch in range(5):
    total_loss = 0
    for x, y in sft_loader:
        logits = model_sft(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    if epoch in [0, 2, 4]:
        print(f"  Epoch {epoch+1}: loss={total_loss/len(sft_loader):.4f}")
print()


# ============================================================
# Part 3: 奖励模型 (Reward Model)
# ============================================================

print("=" * 60)
print("Part 3: 奖励模型")
print("=" * 60)
print("""
奖励模型学习人类偏好：
  给定同一个问题的两个回答 (chosen, rejected)
  奖励模型要学会给 chosen 打更高分

训练数据格式：
  {"prompt": "什么是光合作用？",
   "chosen": "光合作用是植物利用光能将CO2和H2O转化为有机物的过程...",
   "rejected": "光合作用是一种化学反应。就这样。"}

损失函数（Bradley-Terry Model）：
  loss = -log(σ(r_chosen - r_rejected))
  → 让 chosen 的奖励分数尽量高于 rejected
""")


class RewardModel(nn.Module):
    """基于语言模型的奖励模型：最后一层换成标量输出"""

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        d_model = base_model.head.in_features
        self.reward_head = nn.Linear(d_model, 1)
        self.base.head = nn.Identity()

    def forward(self, x):
        h = self.base(x)
        rewards = self.reward_head(h)
        return rewards.mean(dim=1).squeeze(-1)  # 取序列平均作为总奖励


base_for_rm = TinyLM(vocab_size=vocab_size, seq_len=seq_len)
reward_model = RewardModel(base_for_rm)

n_pairs = 300
preference_data = []
for _ in range(n_pairs):
    prompt = torch.randint(0, vocab_size, (seq_len,))
    chosen = (prompt + 1) % vocab_size
    rejected = torch.randint(0, vocab_size, (seq_len,))
    preference_data.append((chosen, rejected))

pref_loader = torch.utils.data.DataLoader(
    preference_data, batch_size=32, shuffle=True
)

optimizer_rm = torch.optim.AdamW(reward_model.parameters(), lr=1e-3)

print("奖励模型训练:")
for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0
    for chosen, rejected in pref_loader:
        r_chosen = reward_model(chosen)
        r_rejected = reward_model(rejected)

        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        loss.backward()
        optimizer_rm.step()
        optimizer_rm.zero_grad()

        total_loss += loss.item()
        correct += (r_chosen > r_rejected).sum().item()
        total += chosen.size(0)

    if epoch in [0, 4, 9]:
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}: loss={total_loss/len(pref_loader):.4f}, "
              f"准确率={acc:.1f}%")

print("\n→ 奖励模型学会了：有规律的输出(chosen) > 随机输出(rejected)")
print()


# ============================================================
# Part 4: PPO 强化学习（简化版）
# ============================================================

print("=" * 60)
print("Part 4: PPO 强化学习（简化版）")
print("=" * 60)
print("""
PPO 的核心思想：
  1. 用当前策略 (policy) 生成回答
  2. 用奖励模型给回答打分
  3. 好回答 → 增大其生成概率
     差回答 → 减小其生成概率
  4. 但不能变化太大（"Proximal" 的含义）

PPO 损失函数：
  ratio = π(a|s) / π_old(a|s)    ← 新旧策略的概率比
  L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
  → clip 确保策略不会更新太多

还需要 KL 惩罚：
  防止 RLHF 后模型变得"太奇怪"（偏离预训练分布太远）
""")


def compute_ppo_loss(log_probs, old_log_probs, advantages, clip_eps=0.2):
    """简化版 PPO 损失"""
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def kl_penalty(log_probs_new, log_probs_ref, beta=0.1):
    """KL 散度惩罚：防止偏离参考模型太远"""
    kl = (log_probs_ref.exp() * (log_probs_ref - log_probs_new)).sum(-1).mean()
    return beta * kl


policy = TinyLM(vocab_size=vocab_size, seq_len=seq_len)
ref_policy = TinyLM(vocab_size=vocab_size, seq_len=seq_len)
ref_policy.load_state_dict(policy.state_dict())
for p in ref_policy.parameters():
    p.requires_grad = False

reward_model_simple = RewardModel(TinyLM(vocab_size=vocab_size, seq_len=seq_len))

optimizer_ppo = torch.optim.AdamW(policy.parameters(), lr=1e-4)

print("PPO 训练循环（简化演示）:")
for step in range(5):
    prompts = torch.randint(0, vocab_size, (16, seq_len))
    targets = torch.randint(0, vocab_size, (16, seq_len))

    with torch.no_grad():
        old_log_probs = policy.get_log_probs(prompts, targets).detach()
        rewards = reward_model_simple(targets).detach()

    advantages = rewards - rewards.mean()
    advantages = advantages.unsqueeze(1).expand_as(old_log_probs)

    new_log_probs = policy.get_log_probs(prompts, targets)
    ppo_loss = compute_ppo_loss(new_log_probs, old_log_probs, advantages)

    with torch.no_grad():
        ref_log_probs = ref_policy.get_log_probs(prompts, targets)
    kl_loss = kl_penalty(new_log_probs, ref_log_probs)

    total_loss = ppo_loss + kl_loss

    total_loss.backward()
    optimizer_ppo.step()
    optimizer_ppo.zero_grad()

    print(f"  Step {step+1}: ppo_loss={ppo_loss.item():.4f}, "
          f"kl_loss={kl_loss.item():.4f}, "
          f"avg_reward={rewards.mean().item():.4f}")

print()


# ============================================================
# Part 5: DPO —— 更简洁的对齐方法
# ============================================================

print("=" * 60)
print("Part 5: DPO (Direct Preference Optimization)")
print("=" * 60)
print("""
DPO 的洞察：其实不需要单独训练奖励模型！

PPO 的流程：
  偏好数据 → 训练奖励模型 → 用 RL 优化策略 (复杂！)

DPO 的流程：
  偏好数据 → 直接优化策略 (简单！)

DPO 损失函数：
  L = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

  其中：
    y_w = 更好的回答 (winner)
    y_l = 更差的回答 (loser)
    π = 当前策略
    π_ref = 参考策略（SFT 后的模型）
    β = 温度参数

直觉：让好回答相对于参考模型的概率提升，坏回答的概率下降
""")


def dpo_loss(
    policy_model, ref_model,
    chosen_x, chosen_y,
    rejected_x, rejected_y,
    beta=0.1,
):
    """DPO 损失函数"""
    chosen_logps = policy_model.get_log_probs(chosen_x, chosen_y).sum(dim=1)
    rejected_logps = policy_model.get_log_probs(rejected_x, rejected_y).sum(dim=1)

    with torch.no_grad():
        ref_chosen_logps = ref_model.get_log_probs(chosen_x, chosen_y).sum(dim=1)
        ref_rejected_logps = ref_model.get_log_probs(rejected_x, rejected_y).sum(dim=1)

    chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss, (chosen_rewards > rejected_rewards).float().mean()


policy_dpo = TinyLM(vocab_size=vocab_size, seq_len=seq_len)
ref_dpo = TinyLM(vocab_size=vocab_size, seq_len=seq_len)
ref_dpo.load_state_dict(policy_dpo.state_dict())
for p in ref_dpo.parameters():
    p.requires_grad = False

optimizer_dpo = torch.optim.AdamW(policy_dpo.parameters(), lr=1e-4)

print("DPO 训练:")
for step in range(10):
    batch_size = 32
    prompts = torch.randint(0, vocab_size, (batch_size, seq_len))
    chosen_y = (prompts + 1) % vocab_size
    rejected_y = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss, acc = dpo_loss(
        policy_dpo, ref_dpo,
        prompts, chosen_y,
        prompts, rejected_y,
        beta=0.1,
    )

    loss.backward()
    optimizer_dpo.step()
    optimizer_dpo.zero_grad()

    if step in [0, 4, 9]:
        print(f"  Step {step+1}: loss={loss.item():.4f}, "
              f"偏好准确率={acc.item()*100:.1f}%")

print("\n→ DPO 直接从偏好数据学习，不需要奖励模型，实现更简洁")
print()


# ============================================================
# Part 6: 对齐技术的全貌
# ============================================================

print("=" * 60)
print("Part 6: 对齐技术全貌")
print("=" * 60)
print("""
ChatGPT 的训练流程：

  ┌─────────────────────────────────────────────┐
  │  1. 预训练 (Pre-training)                    │
  │     大量文本 → 学会语言能力                    │
  │     数据: 互联网文本 (TB 级)                   │
  │     目标: 预测下一个 token                     │
  ├─────────────────────────────────────────────┤
  │  2. SFT (Supervised Fine-Tuning)             │
  │     高质量对话数据 → 学会对话格式               │
  │     数据: ~13K 条人工标注的 (指令, 回答)        │
  │     目标: 模仿标注者的回答                     │
  ├─────────────────────────────────────────────┤
  │  3. RLHF / DPO                              │
  │     人类偏好数据 → 学会"什么是好回答"           │
  │     数据: ~33K 条偏好对比                      │
  │     目标: 生成人类更喜欢的回答                  │
  └─────────────────────────────────────────────┘

各阶段的"大力出奇迹"程度：
  预训练:  数据多、算力大 → 能力(capability)
  SFT:     数据少但质量高 → 格式(format)
  RLHF:    数据少但精心设计 → 对齐(alignment)

PPO vs DPO:
  PPO:  更成熟，效果经过验证（OpenAI 用的）
  DPO:  更简洁，训练更稳定（开源社区更常用）
""")


# ============================================================
# 练习
# ============================================================

print("=" * 60)
print("动手练习")
print("=" * 60)
print("""
练习 1：改进奖励模型
  在 Part 3 的基础上，让奖励模型不只是取序列平均，
  而是只看最后一个 token 的输出（更接近实际做法）。
  对比两种方式的训练稳定性。

练习 2：DPO 温度实验
  在 Part 5 的 DPO 训练中，尝试不同的 β 值：0.01, 0.1, 0.5, 1.0
  观察：
  - β 太小：策略变化太大，可能不稳定
  - β 太大：策略变化太小，学不到东西

练习 3：构造对抗样例
  生成一些"看似合理但实际有害"的数据作为 rejected
  例如：rejected 的 token 模式看起来有规律，但规律是错的
  观察奖励模型是否能正确区分
""")

print("=" * 60)
print("恭喜完成第 4 课！")
print("下一课我们将学习推理优化 —— 让模型响应速度快 10 倍")
print("=" * 60)
