# 第一阶段：深度学习基础

> 面向后端开发者的大模型入门 —— 用代码理解原理

## 环境准备

```bash
# 安装依赖（只需要 NumPy 和 Matplotlib）
sudo apt install python3-pip
pip install numpy matplotlib

# 或者用 conda
conda install numpy matplotlib
```

## 课程结构

按顺序学习，每课约 30-60 分钟：

| 课程 | 文件 | 核心内容 | 与大模型的关系 |
|------|------|----------|---------------|
| 第 1 课 | `01_numpy_basics.py` | 张量运算、矩阵乘法、Softmax | Transformer 的底层运算 |
| 第 2 课 | `02_gradient_descent.py` | 损失函数、梯度、参数更新 | 模型训练的核心机制 |
| 第 3 课 | `03_neural_network.py` | 前向/反向传播、激活函数 | 深度学习的完整流程 |

## 学习方式

1. **先运行一遍**：`python3 01_numpy_basics.py`，看看输出
2. **逐段阅读代码**：理解每一步在做什么
3. **动手修改**：改参数、加代码，看看效果变化
4. **完成练习**：每课末尾有练习题

## 完成后你将理解

- 为什么大模型训练需要那么多 GPU（大量矩阵运算）
- 为什么要调学习率（太大震荡、太小收敛慢）
- 为什么深层网络比浅层网络更强大（非线性叠加）
- 前向传播和反向传播到底在做什么
- 大模型训练和我们手写的小网络本质上完全一样，只是规模不同

## 推荐配套资源

### 数学基础（3Blue1Brown 微积分系列）

- [Essence of Calculus（微积分的本质）](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) - 完整系列（12集），用动画直观理解导数、积分、极限
  - 重点看前 5 集：导数的几何含义、链式法则（反向传播的数学基础）
  - 第 11 集 Taylor Series 对理解函数近似很有帮助

### 深度学习可视化

- [3Blue1Brown - 神经网络](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - 最直观的数学可视化
- [3Blue1Brown - Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) - 用可视化理解注意力机制（可提前看，建立直觉）

### 其他

- [Andrej Karpathy - micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) - 用 Python 从零实现自动微分
- [李宏毅机器学习](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php) - 最好的中文 ML 课程

## 下一步

完成第一阶段后，进入第二阶段：**Transformer 架构** —— 大模型的核心。
