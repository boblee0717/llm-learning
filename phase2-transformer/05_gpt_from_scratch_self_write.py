"""
======================================================
phase2 / 第 5 课（自写版）：从零构建一个迷你 GPT
======================================================

用法：
1) 运行：python3 05_gpt_from_scratch_self_write.py
2) 按 TODO-1 到 TODO-7 顺序补全（建议照这个顺序，后面的 TODO 依赖前面的）
3) 每补完一个 TODO 就运行一次，靠底部的自动校验即时纠错
   （没填的 TODO 会 raise NotImplementedError，校验会提示「未实现」，这是正常的）

目标：
- 把前 4 课的零件（多头注意力 / 残差 / LayerNorm / FFN）用 PyTorch 拼成完整 GPT
- 理解 token + position embedding 怎么拼出模型输入
- 理解 LM Head 输出 logits、交叉熵损失怎么算
- 理解权重共享（weight tying）
- 手写自回归生成：temperature 与 top-k 采样
- 手写 next-token 训练样本的构造（x / y 错位一个）

完整结构（Pre-Norm 风格）：
    idx ──► token_embedding ─┐
                             + ──► drop ──► [TransformerBlock] × N ──► ln_f ──► lm_head ──► logits
    pos ──► position_embedding ┘

    其中每个 Block：
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

核心公式：
    logits = lm_head( ln_f( blocks( tok_emb + pos_emb ) ) )
    loss   = CrossEntropy( logits[:, t], target[:, t] )   # 逐位置预测下一个 token

提示：本文件是第 5 课主课 05_gpt_from_scratch.py 的「填空版」，
      实现卡住时可以回去对照主课，但建议先自己想清楚再看。
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


class ValidationError(Exception):
    """统一的练习校验错误。"""


def require_not_none(name, value):
    if value is None:
        raise ValidationError(f"{name} 未实现：结果是 None。")


def require_shape(name, actual_shape, expected_shape):
    if tuple(actual_shape) != tuple(expected_shape):
        raise ValidationError(
            f"{name} 形状不对：actual={tuple(actual_shape)}, expected={tuple(expected_shape)}"
        )


def require_close(name, actual, expected, atol=1e-5):
    require_not_none(name, actual)
    if not torch.allclose(actual, expected, atol=atol):
        diff = (actual - expected).abs().max().item()
        raise ValidationError(f"{name} 数值不对：最大误差 {diff:.3e}（容差 {atol:.1e}）")


def require_true(name, cond, hint=""):
    if not cond:
        raise ValidationError(f"{name} 条件不满足：{hint}")


# ============================================================
# 模型配置（不需要你改）
# ============================================================
class GPTConfig:
    """GPT 模型配置 —— 所有超参数集中管理"""

    def __init__(
        self,
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,  # 校验时设 0，避免随机性；训练时可调大
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


# ============================================================
# Part 1：带因果掩码的多头自注意力（TODO-1）
# ============================================================
section("Part 1：CausalSelfAttention（TODO-1）")
# 数据流（B=batch, T=seq_len, C=n_embd）：
#   1. qkv = c_attn(x)                    # (B, T, 3C)，一次性算出 Q/K/V
#   2. q, k, v = qkv.split(C, dim=2)      # 各 (B, T, C)
#   3. 拆成多头：(B, T, C) -> (B, n_head, T, d_k)
#      用 view(B, T, n_head, d_k).transpose(1, 2)
#   4. scores = q @ k^T / sqrt(d_k)       # (B, n_head, T, T)
#   5. 用因果掩码把「看未来」的位置填成 -inf：scores.masked_fill(self.mask[...] == 1, -inf)
#   6. weights = softmax(scores, dim=-1)，再过 attn_dropout
#   7. out = weights @ v                  # (B, n_head, T, d_k)
#   8. 合并多头：transpose(1, 2).contiguous().view(B, T, C)
#   9. out = resid_dropout(c_proj(out))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d_k = config.n_embd // config.n_head

        # Q, K, V 合并成一个线性层，效率更高
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码：上三角（不含对角线）为 1，表示「未来位置」要被屏蔽
        # 注册为 buffer：跟着模型走（.to(device)），但不是可训练参数
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1)
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        # TODO-1: 按上面 9 步实现因果多头自注意力
        #   形状自检：输入 (B, T, C)，输出 (B, T, C)
        #   关键点：
        #     - q @ k.transpose(-2, -1) 才是 (..., T, T)，别忘了只转最后两维
        #     - 掩码切片用 self.mask[:, :, :T, :T]，因为真实 T 可能小于 block_size
        #     - masked_fill 填 float('-inf')，softmax 后这些位置就变 0
        raise NotImplementedError("TODO-1 未完成：请实现 CausalSelfAttention.forward")


# ============================================================
# Part 2：Feed-Forward Network（TODO-2）
# ============================================================
section("Part 2：FeedForward（TODO-2）")
# 两层 MLP：升维 4 倍 -> GELU -> 降回原维
#   x = gelu(c_fc(x))        # (B, T, C) -> (B, T, 4C)
#   x = dropout(c_proj(x))   # (B, T, 4C) -> (B, T, C)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # TODO-2: 实现 FFN（用 F.gelu）
        raise NotImplementedError("TODO-2 未完成：请实现 FeedForward.forward")


# ============================================================
# Part 3：Transformer Block（Pre-Norm，TODO-3）
# ============================================================
section("Part 3：TransformerBlock（TODO-3）")
# Pre-Norm（GPT-2/3、LLaMA 用的版本）：LayerNorm 放在子层「之前」
#   x = x + self.attn(self.ln_1(x))
#   x = x + self.ffn(self.ln_2(x))
# 注意「残差相加」：子层输出加回输入 x，而不是直接替换。


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # TODO-3: 实现 Pre-Norm 的两条残差
        raise NotImplementedError("TODO-3 未完成：请实现 TransformerBlock.forward")


# ============================================================
# Part 4 & 5：完整的 GPT 模型（TODO-4 权重共享、TODO-5 forward、TODO-6 generate）
# ============================================================
section("Part 4 & 5：GPT 模型（TODO-4 / TODO-5 / TODO-6）")


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # TODO-4: 实现权重共享（weight tying）
        #   token_embedding 和 lm_head 的权重形状都是 (vocab_size, n_embd)，
        #   本质是同一张「词表 ↔ 向量」表，可以让它们共用一份参数（省参数、效果更好）。
        #   要点：是让两者的 .weight 指向【同一个】Parameter 对象（共享内存），
        #         而不是把数值拷贝过去（拷贝后还是两份独立参数，训练会各走各的）。
        #   想清楚把谁赋给谁后，写一行赋值；再删掉下面这行 pass。
        pass  # TODO-4：占位，实现后删掉

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token 索引
        targets: (B, T) 目标 token 索引（训练时提供；推理时为 None）
        返回:    (logits, loss)，loss 在 targets 为 None 时是 None
        """
        # TODO-5: 实现完整前向
        #   1. B, T = idx.shape；断言 T <= block_size
        #   2. tok_emb = token_embedding(idx)              # (B, T, n_embd)
        #   3. pos_emb = position_embedding(arange(T))      # (T, n_embd)，靠广播加到每个 batch
        #      注意 arange 要和 idx 在同一个 device：torch.arange(T, device=idx.device)
        #   4. x = drop(tok_emb + pos_emb)
        #   5. 依次过每个 block
        #   6. x = ln_f(x)；logits = lm_head(x)            # (B, T, vocab_size)
        #   7. 若 targets 不为 None，用 F.cross_entropy 算 loss：
        #        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        #   8. return logits, loss
        raise NotImplementedError("TODO-5 未完成：请实现 GPT.forward")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成：每步预测下一个 token，拼回去，再预测下一个。

        temperature: 控制随机性（<1 更确定，>1 更随机）
        top_k:       只在概率最高的 k 个 token 里采样（None 表示全词表采样）
        """
        # TODO-6: 实现自回归采样生成
        #   循环 max_new_tokens 次，每次：
        #     1. 把上下文截断到最后 block_size 个 token：idx_crop = idx[:, -block_size:]
        #     2. logits, _ = self(idx_crop)
        #     3. 只取最后一个位置的 logits 并除以 temperature：
        #          logits = logits[:, -1, :] / temperature
        #     4. 若 top_k 不为 None：取 top_k 个最大值，把小于第 k 大的位置设为 -inf
        #          v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        #          logits[logits < v[:, [-1]]] = float('-inf')
        #     5. probs = softmax(logits, dim=-1)
        #     6. next_token = torch.multinomial(probs, num_samples=1)
        #     7. idx = torch.cat([idx, next_token], dim=1)
        #   循环结束后 return idx
        raise NotImplementedError("TODO-6 未完成：请实现 GPT.generate")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Part 6：训练样本构造（TODO-7）
# ============================================================
section("Part 6：get_batch（TODO-7）")
# next-token 预测的样本：从一长串 token 里随机切窗口
#   x = data[i   : i+block]        # 输入
#   y = data[i+1 : i+block+1]      # 目标 = 输入右移一位
# 这样 x 的第 t 个位置，对应要预测的就是 y 的第 t 个位置（即原文的下一个 token）。


def get_batch(data, block_size, batch_size):
    """
    data:       一维 LongTensor（整段语料的 token 序列）
    返回:       x, y，形状都是 (batch_size, block_size)，y 是 x 右移一位
    """
    # TODO-7: 实现随机批次采样
    #   1. 随机起点：ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    #   2. x = 把每个起点 i 的 data[i:i+block_size] stack 起来
    #   3. y = 把每个起点 i 的 data[i+1:i+block_size+1] stack 起来
    #   4. return x, y
    raise NotImplementedError("TODO-7 未完成：请实现 get_batch")


# ============================================================
# 参考实现（仅供自动校验使用，请不要照抄到上面）
# ============================================================
def _ref_attn_forward(attn, x):
    B, T, C = x.shape
    qkv = attn.c_attn(x)
    q, k, v = qkv.split(C, dim=2)
    q = q.view(B, T, attn.n_head, attn.d_k).transpose(1, 2)
    k = k.view(B, T, attn.n_head, attn.d_k).transpose(1, 2)
    v = v.view(B, T, attn.n_head, attn.d_k).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.d_k)
    scores = scores.masked_fill(attn.mask[:, :, :T, :T] == 1, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    weights = attn.attn_dropout(weights)
    out = weights @ v
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return attn.resid_dropout(attn.c_proj(out))


def _ref_ffn_forward(ffn, x):
    return ffn.dropout(ffn.c_proj(F.gelu(ffn.c_fc(x))))


def _ref_block_forward(block, x):
    x = x + _ref_attn_forward(block.attn, block.ln_1(x))
    x = x + _ref_ffn_forward(block.ffn, block.ln_2(x))
    return x


def _ref_gpt_forward(model, idx, targets=None):
    B, T = idx.shape
    tok = model.token_embedding(idx)
    pos = model.position_embedding(torch.arange(T, device=idx.device))
    x = model.drop(tok + pos)
    for block in model.blocks:
        x = _ref_block_forward(block, x)
    x = model.ln_f(x)
    logits = model.lm_head(x)
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss


@torch.no_grad()
def _ref_generate_greedy(model, idx, max_new_tokens):
    # top_k=1 时 softmax 只剩一个非零位置，multinomial 必选它 → 等价于贪心 argmax。
    # 用它做确定性对照，能在不依赖随机种子的情况下验证 generate 的核心逻辑。
    for _ in range(max_new_tokens):
        idx_crop = idx[:, -model.config.block_size:]
        logits, _ = _ref_gpt_forward(model, idx_crop)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


# ============================================================
# 自动校验
# ============================================================
def run_check(label, fn):
    try:
        fn()
        print(f"  ✓ {label}")
        return True
    except NotImplementedError as err:
        print(f"  … {label}：未实现 —— {err}")
        return False
    except ValidationError as err:
        print(f"  ✗ {label}：{err}")
        return False
    except Exception as err:  # noqa: BLE001
        print(f"  ✗ {label}：运行出错 —— {type(err).__name__}: {err}")
        return False


def check_todo1():
    torch.manual_seed(0)
    cfg = GPTConfig()
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, cfg.block_size, cfg.n_embd)
    out = attn(x)
    require_not_none("TODO-1", out)
    require_shape("TODO-1", out.shape, (2, cfg.block_size, cfg.n_embd))
    require_close("TODO-1 数值", out, _ref_attn_forward(attn, x))
    # 因果性：改动最后一个位置的输入，不应影响前面位置的输出
    x2 = x.clone()
    x2[:, -1, :] += 5.0
    out2 = attn(x2)
    require_close("TODO-1 因果掩码", out[:, :-1], out2[:, :-1])


def check_todo2():
    torch.manual_seed(1)
    cfg = GPTConfig()
    ffn = FeedForward(cfg).eval()
    x = torch.randn(2, cfg.block_size, cfg.n_embd)
    out = ffn(x)
    require_not_none("TODO-2", out)
    require_shape("TODO-2", out.shape, (2, cfg.block_size, cfg.n_embd))
    require_close("TODO-2 数值", out, _ref_ffn_forward(ffn, x))


def check_todo3():
    torch.manual_seed(2)
    cfg = GPTConfig()
    block = TransformerBlock(cfg).eval()
    x = torch.randn(2, cfg.block_size, cfg.n_embd)
    out = block(x)
    require_not_none("TODO-3", out)
    require_shape("TODO-3", out.shape, (2, cfg.block_size, cfg.n_embd))
    require_close("TODO-3 数值", out, _ref_block_forward(block, x))
    # Pre-Norm 是残差结构：输出应明显不同于纯子层输出，但量级可控
    require_true("TODO-3 残差存在", not torch.allclose(out, x), "输出和输入完全相同，可能漏了残差或子层")


def check_todo4():
    cfg = GPTConfig()
    model = GPT(cfg)
    shared = model.token_embedding.weight is model.lm_head.weight
    if not shared:
        # 留白态（还没写共享）/ 只拷了数值（两份独立对象）都归到「未实现」一列，
        # 提示统一指向「要共享同一个对象」。
        if torch.equal(model.token_embedding.weight, model.lm_head.weight):
            raise NotImplementedError(
                "TODO-4 未完成：两者数值相同但不是同一个对象——别拷数值，要共享同一个 Parameter"
            )
        raise NotImplementedError("TODO-4 未完成：请让 token_embedding 与 lm_head 共享权重")
    # 真共享后，改一个应同时改另一个（确认是同一块内存）
    with torch.no_grad():
        model.lm_head.weight[0, 0] += 1.0
    require_true(
        "TODO-4 权重共享",
        model.token_embedding.weight[0, 0].item() == model.lm_head.weight[0, 0].item(),
        "改 lm_head.weight 应同步反映到 token_embedding.weight（同一块内存）",
    )


def check_todo5():
    torch.manual_seed(3)
    cfg = GPTConfig()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(idx, targets)
    require_not_none("TODO-5 logits", logits)
    require_shape("TODO-5 logits", logits.shape, (2, cfg.block_size, cfg.vocab_size))
    ref_logits, ref_loss = _ref_gpt_forward(model, idx, targets)
    require_close("TODO-5 logits 数值", logits, ref_logits, atol=1e-4)
    require_not_none("TODO-5 loss", loss)
    require_true("TODO-5 loss 是标量", loss.dim() == 0, f"loss 应是标量，实际维度 {loss.dim()}")
    require_true("TODO-5 loss 有限", torch.isfinite(loss).item(), "loss 出现 NaN/Inf")
    require_close("TODO-5 loss 数值", loss, ref_loss, atol=1e-4)
    # targets=None 时不应返回 loss
    _, none_loss = model(idx)
    require_true("TODO-5 推理无 loss", none_loss is None, "targets=None 时 loss 应为 None")


def check_todo6():
    torch.manual_seed(4)
    cfg = GPTConfig()
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=4)
    require_not_none("TODO-6", out)
    require_shape("TODO-6 输出长度", out.shape, (1, 3 + 5))
    require_true(
        "TODO-6 保留前缀",
        torch.equal(out[:, :3], prompt),
        "生成结果应以原始 prompt 开头",
    )
    require_true(
        "TODO-6 token 合法",
        bool(((out >= 0) & (out < cfg.vocab_size)).all().item()),
        "生成的 token 索引应落在 [0, vocab_size) 内",
    )
    # 确定性对照：top_k=1 必然贪心，应与参考的 argmax 逐步生成完全一致。
    # 这能抓出「忘了取最后一个位置 / 忘了拼接 / top_k 实现错」等逻辑错误。
    greedy = model.generate(prompt.clone(), max_new_tokens=5, temperature=1.0, top_k=1)
    ref_greedy = _ref_generate_greedy(model, prompt.clone(), max_new_tokens=5)
    require_true(
        "TODO-6 贪心逻辑",
        torch.equal(greedy, ref_greedy),
        "top_k=1 时应等价于贪心（每步取概率最高 token），结果与参考实现不一致",
    )
    # 上下文裁剪：prompt 比 block_size 还长时，必须截断到最后 block_size 个 token，
    # 否则 forward 里的 T <= block_size 断言会失败。
    long_prompt = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 4))
    out_long = model.generate(long_prompt, max_new_tokens=3, temperature=1.0, top_k=1)
    require_shape("TODO-6 超长 prompt 裁剪", out_long.shape, (1, cfg.block_size + 4 + 3))


def check_todo7():
    torch.manual_seed(5)
    block_size, batch_size = 8, 4
    data = torch.arange(100, dtype=torch.long)  # 0,1,2,... 便于看错位
    x, y = get_batch(data, block_size, batch_size)
    require_not_none("TODO-7 x", x)
    require_not_none("TODO-7 y", y)
    require_shape("TODO-7 x", x.shape, (batch_size, block_size))
    require_shape("TODO-7 y", y.shape, (batch_size, block_size))
    # 核心不变量：y 是 x 右移一位 -> y[:, :-1] == x[:, 1:]
    require_true(
        "TODO-7 next-token 错位",
        torch.equal(y[:, :-1], x[:, 1:]),
        "y 应该是 x 整体右移一位（每个位置的目标 = 下一个 token）",
    )


section("开始自动校验（按 TODO 顺序，未实现的会标 …）")
results = [
    run_check("TODO-1 CausalSelfAttention.forward", check_todo1),
    run_check("TODO-2 FeedForward.forward", check_todo2),
    run_check("TODO-3 TransformerBlock.forward", check_todo3),
    run_check("TODO-4 权重共享 weight tying", check_todo4),
    run_check("TODO-5 GPT.forward", check_todo5),
    run_check("TODO-6 GPT.generate", check_todo6),
    run_check("TODO-7 get_batch", check_todo7),
]

passed = sum(results)
print(f"\n通过 {passed}/{len(results)} 个 TODO")


# ============================================================
# 全部通过后：跑一个迷你训练 + 生成（无需补全，作为奖励演示）
# ============================================================
if all(results):
    section("彩蛋：全部通过！来训练一个字符级迷你 GPT")

    text = (
        "to be or not to be that is the question "
        "whether tis nobler in the mind to suffer "
    ) * 20

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    cfg = GPTConfig(
        vocab_size=len(chars),
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.1,
    )
    model = GPT(cfg)
    print(f"词表大小 {len(chars)}，模型参数量 {model.count_parameters():,}")

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    model.train()
    for step in range(300):
        xb, yb = get_batch(data, cfg.block_size, batch_size=16)
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == 299:
            print(f"  step {step:>3d}  loss {loss.item():.4f}")

    model.eval()
    prompt = "to be"
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=60, temperature=0.8, top_k=5)
    generated = "".join(itos[i] for i in out[0].tolist())
    print(f"\n提示词 '{prompt}' 续写：\n  {generated!r}")
    print("\n（数据极少、模型极小，能复读出训练片段就算成功，体会 next-token 预测即可）")
else:
    section("还有 TODO 没完成")
    print("把上面标 … 或 ✗ 的 TODO 补全后再运行，全部通过会触发训练 + 生成彩蛋。")
