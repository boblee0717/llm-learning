"""
======================================================
PyTorch 专项 / 第 8 课（自写版）：毕业项目 —— 亲手拼装训练流 🎓
======================================================

用法：
1. 运行：python 08_capstone_train_self_write.py
2. 按 TODO-1 到 TODO-5 顺序补全（这是「整合」练习，后面的 TODO 会用到前面的）
3. 每补完一个就运行一次，靠 require_xxx 即时纠错（没填的返回 None，提示「未实现」是正常的）

目标（把前 7 课零件亲手串成一条训练流）：
- build_param_groups：矩阵权重做 weight decay、一维参数不做（第 4 课）
- lm_loss：把 (B,T,V) 摊平喂 cross_entropy（语言建模损失）
- evaluate：eval + no_grad 算平均 val loss，并恢复 train 模式（第 6 课）
- accumulate_and_step：梯度累积 + 裁剪 + step（第 6 课）
- fit：把上面零件组装成完整训练循环，返回最优 val loss（毕业验收）

对照：本课主课 08_capstone_train.py。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows / PowerShell 下中文输出防乱码
sys.stderr.reconfigure(encoding="utf-8")  # ValidationError 走 stderr，也要防乱码

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(1337)


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
    if not torch.allclose(actual, expected, atol=atol):
        raise ValidationError(
            f"{name} 数值不对\nactual=\n{actual}\nexpected=\n{expected}"
        )


# ---- 以下零件已为你备好（沿用主课），你只需填 5 个 TODO 把它们组装起来 ----
VOCAB_SIZE = 16
BLOCK_SIZE = 8
N_EMBD = 32


class MiniLM(nn.Module):
    """迷你 token 语言模型：Embedding（输入第一层）+ MLP + LM head。"""

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd), nn.Tanh(),
            nn.Linear(n_embd, n_embd), nn.Tanh(),
        )
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        return self.head(self.mlp(self.embed(idx)))   # (B,T)→(B,T,C)→(B,T,vocab)


class SeqDataset(Dataset):
    def __init__(self, data, block_size):
        self.data, self.block_size = data, block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        return (self.data[i : i + self.block_size],
                self.data[i + 1 : i + 1 + self.block_size])


_data = torch.arange(0, 1024) % VOCAB_SIZE
_full = SeqDataset(_data, BLOCK_SIZE)
_gen = torch.Generator().manual_seed(42)
_nval = int(len(_full) * 0.2)
_train_ds, _val_ds = random_split(_full, [len(_full) - _nval, _nval], generator=_gen)
_train_loader = DataLoader(_train_ds, batch_size=16, shuffle=True, drop_last=True, num_workers=0)
_val_loader = DataLoader(_val_ds, batch_size=16, shuffle=False, num_workers=0)
_small_loader = DataLoader(_train_ds, batch_size=16, shuffle=False, drop_last=True, num_workers=0)


# ============================================================
section("TODO-1：build_param_groups —— 区分 decay / no-decay（第 4 课）")
# ============================================================
# 返回 [decay 组, no_decay 组] 两个 dict：
#   - ndim>=2 的参数（矩阵权重，如 Linear.weight、Embedding.weight）→ 做 weight decay
#   - ndim<2 的参数（bias 等一维参数）→ 不做 weight decay
# 每个 dict 形如 {"params": [...], "weight_decay": 数值}。
#
# 提示：遍历 model.parameters()，按 p.ndim 分两组；decay 组 wd=weight_decay，另一组 wd=0.0。


def build_param_groups(model, weight_decay=0.1):
    # TODO-1: 返回 [{"params":..., "weight_decay":weight_decay}, {"params":..., "weight_decay":0.0}]
    return None


_pg_model = MiniLM(VOCAB_SIZE, N_EMBD)
_pg = build_param_groups(_pg_model, weight_decay=0.1)
require_not_none("TODO-1 build_param_groups", _pg)
require_true("TODO-1 返回两组", len(_pg) == 2, "应返回 [decay, no_decay] 两个 dict")
require_true("TODO-1 decay 组都是矩阵权重",
             all(p.ndim >= 2 for p in _pg[0]["params"]), "decay 组应只含 ndim>=2 的参数")
require_true("TODO-1 no_decay 组都是一维",
             all(p.ndim < 2 for p in _pg[1]["params"]), "no_decay 组应只含 ndim<2 的参数")
require_true("TODO-1 wd 设置正确",
             _pg[0]["weight_decay"] == 0.1 and _pg[1]["weight_decay"] == 0.0,
             "decay 组 wd=0.1、no_decay 组 wd=0.0")
print("build_param_groups OK：decay", len(_pg[0]["params"]), "组 / no_decay", len(_pg[1]["params"]), "组")


# ============================================================
section("TODO-2：lm_loss —— 语言建模损失（摊平后喂 cross_entropy）")
# ============================================================
# 输入 logits: (B, T, V)，y: (B, T)（long）。输出标量 loss。
# cross_entropy 只吃 (N, V) 和 (N,)，所以要先把前两维摊平：
#   logits → (B*T, V)，y → (B*T,)，再 F.cross_entropy。
#
# 提示：B, T, V = logits.shape；用 logits.view(B*T, V) 和 y.reshape(B*T)。


def lm_loss(logits, y):
    # TODO-2: 摊平后返回 F.cross_entropy(...)
    return None


_lg = torch.randn(2, 4, VOCAB_SIZE)
_yy = torch.randint(0, VOCAB_SIZE, (2, 4))
_loss2 = lm_loss(_lg, _yy)
require_not_none("TODO-2 lm_loss", _loss2)
require_close("TODO-2 与 cross_entropy 一致", _loss2,
              F.cross_entropy(_lg.view(8, VOCAB_SIZE), _yy.reshape(8)))
print("lm_loss OK：标量 loss =", round(_loss2.item(), 4))


# ============================================================
section("TODO-3：evaluate —— eval + no_grad 算平均 val loss（第 6 课）")
# ============================================================
# 遍历 loader 算每个 batch 的 lm_loss，返回平均值（python float）。要点：
#   1. 进来先记下 model.training，切 model.eval()；算完恢复原状态（别污染训练）。
#   2. 整段放在 torch.no_grad() 里（评估不需要梯度）。
#   3. 返回的是 float（用 .item()），不是 tensor。
#
# 提示：losses = []；with torch.no_grad(): for xb,yb in loader: losses.append(lm_loss(model(xb),yb).item())


def evaluate(model, loader):
    # TODO-3: 返回平均 val loss（float），评估后恢复 train 状态
    return None


_ev_model = MiniLM(VOCAB_SIZE, N_EMBD)
_ev_model.train()
_ev = evaluate(_ev_model, _small_loader)
require_not_none("TODO-3 evaluate", _ev)
require_true("TODO-3 返回 float", isinstance(_ev, float), "应返回 python float（.item()）")
require_true("TODO-3 评估后恢复 train 模式", _ev_model.training,
             "evaluate 结束应把模型恢复到进入前的 train 状态")
print("evaluate OK：平均 val loss =", round(_ev, 4))


# ============================================================
section("TODO-4：accumulate_and_step —— 梯度累积 + 裁剪 + step（第 6 课）")
# ============================================================
# 输入 micro_batches：一个 [(xb,yb), ...] 列表，长度 = accum_steps。要做：
#   1. opt.zero_grad()
#   2. 对每个 micro-batch：loss = lm_loss(model(xb), yb) / accum_steps；loss.backward()（梯度累加）
#   3. 裁剪：torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#   4. opt.step()
#   5. 返回累加的平均 loss（float）
# 关键坑：micro loss 要【除以 accum_steps】，否则等效放大 lr。
#
# 提示：running=0.0; 循环里 running += loss.item()；最后 return running。


def accumulate_and_step(model, micro_batches, opt, accum_steps, max_norm=1.0):
    # TODO-4: 梯度累积 + 裁剪 + step，返回平均 loss（float）
    return None


_as_model = MiniLM(VOCAB_SIZE, N_EMBD)
_as_opt = torch.optim.AdamW(_as_model.parameters(), lr=1e-2)
_mb = [next(iter(_small_loader)) for _ in range(2)]
_l1 = accumulate_and_step(_as_model, _mb, _as_opt, accum_steps=2)
require_not_none("TODO-4 accumulate_and_step", _l1)
require_true("TODO-4 返回 float", isinstance(_l1, float), "应返回平均 loss 的 float")
for _ in range(25):
    _last = accumulate_and_step(_as_model, _mb, _as_opt, accum_steps=2)
require_true("TODO-4 多步后 loss 下降", _last < _l1,
             "若真的 opt.step 了，反复训练同一批 loss 应明显下降")
print("accumulate_and_step OK：首步 loss", round(_l1, 4), "→ 25 步后", round(_last, 4))


# ============================================================
section("TODO-5：fit —— 把零件组装成完整训练循环（毕业验收）")
# ============================================================
# 用你上面写的 build_param_groups / accumulate_and_step / evaluate 拼出训练循环：
#   1. opt = AdamW(build_param_groups(model), lr=3e-3)
#   2. 循环 total_steps 步：每步从 train_loader 取 accum_steps 个 micro-batch（迭代器用尽就重起），
#      调 accumulate_and_step 训练一步。
#   3. 每隔若干步用 evaluate(model, val_loader) 评估，记录最优（最小）val loss。
#   4. 返回最优 val loss（float）。
#
# 提示：train_iter = iter(train_loader)；用 try/except StopIteration 在耗尽时 train_iter=iter(...) 重起。
#       best = float("inf")；每次评估 best = min(best, val)。


def fit(model, train_loader, val_loader, total_steps, accum_steps=2):
    # TODO-5: 完整训练循环，返回最优 val loss（float）
    return None


torch.manual_seed(0)
_fit_model = MiniLM(VOCAB_SIZE, N_EMBD)
_best = fit(_fit_model, _train_loader, _val_loader, total_steps=300, accum_steps=2)
require_not_none("TODO-5 fit", _best)
require_true("TODO-5 返回 float", isinstance(_best, float), "应返回最优 val loss 的 float")
require_true("TODO-5 模型确实学会了 next-token", _best < 0.5,
             f"最优 val loss 应远低于随机基线 ln(16)≈2.77，实际={_best}")
print("fit OK：最优 val loss =", round(_best, 4), "（从 ~2.77 降下来，模型学会了）")


# ============================================================
section("全部 TODO 校验通过 ✓ —— 你毕业了 🎓")
# ============================================================
print("""
你已经亲手把前 7 课的零件组装成一条完整训练流：
  1. build_param_groups：矩阵权重做 weight decay、一维参数不做
  2. lm_loss：(B,T,V) 摊平喂 cross_entropy
  3. evaluate：eval + no_grad 算 val loss 并恢复 train 模式
  4. accumulate_and_step：梯度累积（micro loss 除以 accum）+ 裁剪 + step
  5. fit：组装成完整循环，模型 val loss 从随机基线一路降下来

复盘三问：
  * 一个真实训练脚本的代码顺序是什么？（体检 → 数据 → 模型 → 优化器 → 循环 → 存盘 → 验收）
  * 为什么梯度累积时 micro loss 要除以 accum_steps？
  * nn.Embedding 把什么形状变成了什么形状？它和普通 Linear 查表有什么关系？

→ 恭喜打通 PyTorch 专项 8 课，正式进入 phase3（训练 / LoRA / 量化 / RLHF / 推理）！
""")
