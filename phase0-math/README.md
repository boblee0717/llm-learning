# 第 0 阶段：矩阵运算补强（Matrix Warm-up）

> 面向"代码看得懂、形状一推就晕"的复习者：把后面 Transformer / GPT 里反复出现的矩阵套路一次性练熟。

这一阶段是**为 phase1 / phase2 服务的回炉**，不追求完整的线性代数体系，只覆盖 LLM 里真正用到的那部分。

## 为什么需要这一阶段

在 phase2 看 attention 的时候你会反复遇到：

- `Q @ K.T / sqrt(d_k)` —— **矩阵乘法 + 形状推理**
- `softmax(scores, axis=-1)` —— **`axis` / `keepdims` 的直觉**
- `(B, T, H, d_h) → (B, H, T, d_h)` —— **reshape / transpose / permute**
- `dL/dW = X.T @ dL/dY` —— **矩阵求导**

这些套路在 phase1 第 1、2、3 课里出现过，但当时是"边学神经网络边凑出来的"，没有系统性练过。这一阶段就是把它们**抽出来单独练**。

## 课程结构

按顺序学习，每课 30–45 分钟。每课都配 **主课脚本 + 自写练习 + 重置脚本**，与 phase1 完全一致的节奏。

| 课 | 主课文件 | 自写练习 | 核心内容 | 后面会在哪用上 |
|---|---|---|---|---|
| 第 1 课 | `01_vectors_and_axes.py` | `01_vectors_and_axes_self_write.py` | 向量、点积、范数、`axis` / `keepdims`、广播规则 | softmax 行归一、cosine 相似度、LayerNorm |
| 第 2 课 | `02_matmul_and_shapes.py` | `02_matmul_and_shapes_self_write.py` | 矩阵乘法两种解读、`@` / `matmul` / `einsum`、batched matmul | $QK^T$、`attn @ V`、`(B, T, d)` 的批量乘 |
| 第 3 课 | `03_reshape_transpose_split.py` | `03_reshape_transpose_split_self_write.py` | 转置 / reshape / permute / split / concat / outer / Hadamard / **`np.triu`·`np.tril` 三角矩阵构造** | 多头切分、`W_qkv` 拆分、残差拼接、**causal mask** |
| 第 4 课 | `04_matrix_calculus.py` | `04_matrix_calculus_self_write.py` | 线性层求导、链式法则、softmax 雅可比、数值梯度校验 | 反向传播、训练每一步都在做 |

## 快速开始

```bash
# 1. 进入目录
cd phase0-math

# 2. 跑主课
python3 01_vectors_and_axes.py
python3 02_matmul_and_shapes.py
python3 03_reshape_transpose_split.py
python3 04_matrix_calculus.py

# 3. 做自写练习（推荐方式）
python3 01_vectors_and_axes_self_write.py
# ↑ 按 TODO 从前往后填，每填一个就跑一次，看 assert 是否通过

# 4. 想再练一遍？一键重置
python3 reset_exercises_01.py
python3 reset_exercises_02.py
python3 reset_exercises_03.py
python3 reset_exercises_04.py
```

只依赖 `numpy`，不需要 PyTorch。

## 学习方式（强烈建议照着走）

| 步骤 | 做什么 | 为什么 |
|---|---|---|
| 1 | 读主课注释 + 跑一遍，看输出形状 | 先把"形状直觉"建起来 |
| 2 | 关掉主课，打开 self_write，**用脑子推形状** | 不推形状的练习等于没练 |
| 3 | 填一个 TODO 就跑一次，让 assert 帮你纠错 | 即时反馈是这一阶段的关键 |
| 4 | 全部通过后做 5 分钟复盘 | 写下"本课最容易错的 1 个形状陷阱" |
| 5 | 隔 2–3 天用 reset 脚本重置后再做一遍 | 矩阵运算靠肌肉记忆，必须二刷 |

## 阶段完成标准（自检清单）

完成第 0 阶段后，闭着眼睛你应该能回答：

- [ ] `(2, 3) @ (3, 5)` 的结果是什么形状？为什么？
- [ ] `(B, T, d_in) @ (d_in, d_out)` 的结果是什么形状？广播在哪一维？
- [ ] `np.sum(x, axis=1)` 和 `np.sum(x, axis=1, keepdims=True)` 在 softmax 里为什么必须用后者？
- [ ] `(B, T, H, d_h)` 怎么变成 `(B, H, T, d_h)`？为什么 attention 要这样 reshape？
- [ ] $Y = XW$，已知 $\partial L / \partial Y$，怎么求 $\partial L / \partial W$ 和 $\partial L / \partial X$？（写出公式）
- [ ] softmax 的雅可比为什么是 $\text{diag}(s) - s s^T$？
- [ ] `Q @ K.T` 里 K 为什么要转置？转置的是哪两个轴？
- [ ] 数值梯度校验的核心思想是什么？为什么 $\epsilon$ 不是越小越好？
- [ ] `np.triu(np.ones((n,n)), k=1)` 和 `np.tril(np.ones((n,n)))` 分别长什么样？哪个对应 GPT 的 causal mask？

能答上来 6/8，就直接去 phase2 复习 attention，矩阵这一关已经过了。

## 与其他阶段的关系

```
phase0-math  ──┐
               ├──→  phase1-foundations（NumPy / 梯度 / 神经网络）
               │
               └──→  phase2-transformer（QKV / Multi-Head / GPT）
```

phase0 不是一个独立的"前置课"，它更像一本**随时可以翻回去查的工具书**。当你在 phase2 看到一个形状变换看不懂时，回来跑一下对应那节的主课脚本即可。

## 推荐外部配套

按需查，不必都看：

- 3Blue1Brown：[Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) —— 把"矩阵就是线性变换"这件事讲透
- 李宏毅：[自注意力机制（下）](https://www.youtube.com/watch?v=gmsMY5kc-zw) —— 重点看里面的矩阵形式推导
- The Matrix Cookbook（PDF）—— 矩阵求导公式手册，遇到不会的查它就行
