# 第一阶段：深度学习基础

> 面向后端开发者的大模型入门 —— 用代码理解原理

这一阶段的目标是：先用最小可运行例子吃透“模型是怎么学会的”，再进入 Transformer 细节。

## 快速开始（建议先跑通）

```bash
# 进入目录
cd phase1-foundations

# 建议使用 venv（避免系统 Python 的 pip 限制）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
python3 -m pip install --upgrade pip
python3 -m pip install numpy matplotlib

# 先跑主课代码
python3 01_numpy_basics.py
python3 02_gradient_descent.py
python3 03_neural_network.py
```

## 课程结构

按顺序学习，每课约 30-60 分钟：

| 课程 | 主课文件 | 自写练习 | 核心内容 | 与大模型的关系 |
|------|------|------|------|------|
| 第 1 课 | `01_numpy_basics.py` | `01_numpy_basics_self_write.py` | 张量运算、矩阵乘法、Softmax | Transformer 底层运算 |
| 第 2 课 | `02_gradient_descent.py` | `02_gradient_descent_self_write.py` | 损失函数、梯度、参数更新 | 模型训练核心机制 |
| 第 3 课 | `03_neural_network.py` | `03_neural_network_self_write.py` | 前向/反向传播、激活函数 | 深度学习完整流程 |

## 推荐学习路径（可执行版）

每课建议按下面节奏走一遍：

1. 先运行主课脚本，确认看懂输出
2. 再做自写练习脚本，按 TODO 从前往后填
3. 每完成一个 TODO 就运行一次，利用校验信息修正
4. 完成后做 5 分钟复盘：写下“本课 3 个关键结论”

## 练习重置脚本

如果你想重复练习，可一键重置 TODO 状态：

```bash
cd phase1-foundations

# 第1课重置
python3 reset_exercises_01.py

# 第2课重置
python3 reset_exercises_02.py

# 第3课重置
python3 reset_exercises_03.py
```

说明：

- 重置后会恢复 TODO 空白实现
- 讲解、打印和校验模块会保留
- 适合“重新做一遍”或“教学演示前清空状态”

## 补充概念速查：axis 与 keepdims

这两个参数在 Softmax/Attention 里非常常见。

- `axis=1`：按行聚合（每行单独处理）
- `axis=0`：按列聚合（每列单独处理）
- `keepdims=False`：聚合后删除该轴
- `keepdims=True`：保留该轴且长度变为 1（方便广播）

以 `score.shape=(2,3)` 为例：

- `np.max(score, axis=1).shape == (2,)`
- `np.max(score, axis=1, keepdims=True).shape == (2,1)`

稳定版行级 softmax：

```python
exp_scores = np.exp(score - np.max(score, axis=1, keepdims=True))
attn = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
```

### 多轴示例（`axis=(1,2)`）

如果 `x.shape=(2,3,4)`：

- `np.sum(x, axis=0).shape == (3,4)`
- `np.sum(x, axis=1).shape == (2,4)`
- `np.sum(x, axis=2).shape == (2,3)`
- `np.sum(x, axis=(1,2)).shape == (2,)`
- `np.sum(x, axis=(1,2), keepdims=True).shape == (2,1,1)`

直觉：`axis=(1,2)` 就是“同时压缩后两维，只保留第 0 维”。

### 常见坑：不加 keepdims 的广播失败

如果 `score.shape=(2,3)`：

- `row_max = np.max(score, axis=1)` -> `(2,)`
- `score - row_max` 可能报广播错误

更稳妥：

```python
row_max = np.max(score, axis=1, keepdims=True)  # (2,1)
stable = score - row_max                         # (2,3) - (2,1) 可广播
```

## 补充概念速查：数值溢出与 clip

在 Sigmoid 等函数中常见 `np.clip(x, -500, 500)` 这样的写法，目的是防止 `exp` 溢出。

**问题**：当 `x` 是很大的负数（如 `-1000`），Sigmoid 计算中需要 `e^{-x} = e^{1000}`，超出 `float64` 的最大值（约 `1.8e+308`），结果变成 `inf`，进而传播成 `NaN`。

**解决**：用 `np.clip` 把输入限制在安全范围内（±500），`e^{500}` 仍在 float64 范围内。

**为什么不影响结果**：Sigmoid 在 `|x| > 20` 时输出已非常接近 0 或 1，clip 到 ±500 不会改变实际行为。

```python
np.exp(1000)    # → inf（溢出）
np.exp(-1000)   # → 0.0（正常）
np.exp(500)     # → 1.4e+217（安全范围内）

# 所以 sigmoid 里这样写：
1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

同理，Softmax 中用 `score - max(score)` 也是类似思路：先减去最大值再取 exp，防止 `e^{大数}` 溢出。

## 补充概念速查：权重初始化（Xavier / He）

### 为什么不能随便初始化？

假设一个 100 层网络，每层做 `z = W @ x`：

- 每层输出方差放大（如 ×1.1）→ `1.1^100 ≈ 13781` → **梯度爆炸**
- 每层输出方差缩小（如 ×0.9）→ `0.9^100 ≈ 0.00003` → **梯度消失**

目标：让每一层输出的方差 ≈ 输入的方差，信号就能稳定传播。

### 推导

对于 `z = W @ x`（W 的列数为 fan_in）：

```
Var(z) = fan_in × Var(W) × Var(x)
```

要 `Var(z) = Var(x)` → `Var(W) = 1/fan_in` → `W ~ randn × sqrt(1/fan_in)`

ReLU 激活函数会把一半负值变成 0（方差减半），需要补偿 ×2 → `W ~ randn × sqrt(2/fan_in)`

### 常见变体

| 方法 | 公式 | 适用激活函数 |
|------|------|------------|
| Xavier | `randn × sqrt(1/fan_in)` | Sigmoid, Tanh |
| He (Kaiming) | `randn × sqrt(2/fan_in)` | ReLU |

第 3 课代码中用的是 **He 初始化**（因为隐藏层用了 ReLU）。

## 阶段完成标准（自检）

完成第一阶段后，你应该能自己解释：

- 为什么大模型训练需要大量 GPU（本质是大规模矩阵运算）
- 学习率为什么会导致“收敛慢 / 震荡 / 发散”
- 前向传播和反向传播到底分别在做什么
- 小模型手写训练与 GPT 训练在原理上哪里相同

## 推荐配套资源

### 建议顺序

1. 完成本目录 3 节代码课
2. 补看李宏毅 `Deep Learning / Gradient Descent / Backpropagation`
3. 用 3Blue1Brown 强化数学直觉
4. 再看 `Self-attention`，为第二阶段做准备

### 李宏毅机器学习（主线推荐）

- [Brief Introduction of Deep Learning](https://youtu.be/Dr-WRlEFefw) - 建立神经网络整体直觉
- [Gradient Descent](https://youtu.be/yKKNr-QKz2Q) - 对应第 2 课，理解损失函数与参数更新
- [Backpropagation](https://youtu.be/ibJpTrp5mcE) - 对应第 3 课，理解链式法则
- [自注意力机制 (Self-attention) (上)](https://www.youtube.com/watch?v=hYdO9CscNes) - 进入 Transformer 前必看
- [自注意力机制 (Self-attention) (下)](https://www.youtube.com/watch?v=gmsMY5kc-zw) - 重点看矩阵形式、Multi-Head、位置编码
- [Self-attention 讲义 `self_v7.pdf`](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/self_v7.pdf) - 配合视频效果更好
- [机器学习 2023 课程主页](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php) - 后续重点看 `HW 4 | Self-attention` 与 `HW 5 | Transformer`
- [机器学习 2021 中文版播放列表](https://www.youtube.com/playlist?list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J) - 适合系统补课

### 3Blue1Brown（直觉增强）

- [Essence of Calculus（微积分的本质）](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) - 12 集完整系列
- `Essence of Calculus` 前 5 集 - 导数几何意义与链式法则
- `Essence of Calculus` 第 11 集 - Taylor Series
- [3Blue1Brown - 神经网络](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - 神经网络可视化
- [3Blue1Brown - Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) - 注意力机制直觉

### 进阶补充

- [Andrej Karpathy - micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) - 从零实现自动微分，建议第 3 课后看

## 下一步

完成第一阶段后，进入第二阶段：**Transformer 架构** —— 大模型的核心。
