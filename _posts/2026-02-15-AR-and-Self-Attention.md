---
title: 图像优化线性AR算法到Self-Attention机制
author: yuchong
date: 2026-02-15 00:34:00 +0800
categories: [AI]
tags: [AI]
math: true
---

# 线性AR算法

在几年前，我做过一个深度图算法优化的项目。**问题背景** ：我们有采集+开源（～10W）depth+RGB pair对数据集，但是由于depth获取的物理原理（结构光，TOF等）原因，depth一般质量很差，孔洞，噪声，边缘模糊等。我们如何优化depth呢？其可以写为一个最优化问题：

$$
\min \quad E(x) = \sum_{i=1}^{N} \sum_{j \in \mathcal{N}(i)} w_{ij}(x_i - x_j)^2 + \alpha \sum_{i \in \Omega} (x_i - \hat{x}_i)^2
$$

其中：
- $\alpha$  是惩罚因子；
- $athcal{N}(i)$  是以 $x_i$ 中心点的临域为中心点的 $n$ 邻域；
- $\hat{x}_i$  表示坐标点 $i$ 处depth 优化前真实值；
- $x_i$  表示优化后的目标depth；
- 已知观测集合： $\Omega \subset \{1, \ldots, N\}$ .

该最优化的含义是：要求输出depth忠实于GT值，并且要求优化后的depth在其邻域里是足够平滑的。

上述最优化方程求导，我们有：

**第一项：**

$$
E_s = \sum_i \sum_{j \in \mathcal{N}(i)} w_{ij}(x_i - x_j)^2
$$

(1) $i=k$ :

$$
\frac{\partial}{\partial x_k} w_{kj}(x_k - x_j)^2 = 2w_{kj}(x_k - x_j)
$$

(2) $j=k$ :

$$
\frac{\partial}{\partial x_k} w_{ik}(x_i - x_k)^2 = -2w_{ik}(x_i - x_k) = 2w_{ik}(x_k - x_i)
$$

合并后：

$$
\frac{\partial E_s}{\partial x_k} = 2 \sum_j w_{kj}(x_k - x_j) + 2 \sum_i w_{ik}(x_k - x_i)
$$

一般情况下：

$$
w_{ij} = w_{ji}
$$

所以有：

$$
\frac{\partial E_s}{\partial x_k} = 2 \sum_j (w_{kj} + w_{jk})(x_k - x_j)
$$

**第二项：**

$$
E_d = \alpha \sum_{i \in \Omega} (x_i - \hat{x}_i)^2
$$

若 $ k \in \Omega $：

$$
\frac{\partial E_d}{\partial x_k} = 2\alpha (x_k - \hat{x}_k)
$$

若 $ k \notin \Omega $：

$$
\frac{\partial E_d}{\partial x_k} = 0
$$

综合以上两步，总偏导：

$$
\frac{\partial E}{\partial x_k} = 2 \sum_j (w_{kj} + w_{jk})(x_k - x_j) + 2\alpha \mathbf{1}_{k \in \Omega}(x_k - \hat{x}_k)
$$

根据极值条件有：

$$
\sum_j (w_{kj} + w_{jk})(x_k - x_j) + \alpha \mathbf{1}_{k \in \Omega}(x_k - \hat{x}_k) = 0
$$

整理得到：

$$
\left( \sum_j (w_{kj} + w_{jk}) + \alpha \mathbf{1}_{k \in \Omega} \right) x_k - \sum_j (w_{kj} + w_{jk}) x_j = \alpha \mathbf{1}_{k \in \Omega} \hat{x}_k
$$

写成线性方程组形式，对所有 $ k = 1, \ldots, N $，得到：

$$
\sum_{j=1}^{N} A_{kj} x_j = b_k \tag{1}
$$

其中：

$$
A_{kk} = \sum_j (w_{kj} + w_{jk}) + \alpha \mathbf{1}_{k \in \Omega}
$$

$$
A_{kj} = -(w_{kj} + w_{jk}), \quad j \neq k
$$

$$
b_k = \alpha \mathbf{1}_{k \in \Omega} \hat{x}_k
$$

$ A $ 其实可以看作关联矩阵（或图拉普拉斯矩阵的变体）。

# Self-Attention机制

注意力机制公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V
$$

注意力矩阵：

$$
A = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right)
$$

最终输出可以改写为：

$$
\text{Output} = AV
$$

对序列中第 $ i $ 个位置：

$$
\text{output}_i = \sum_{j=1}^{n} \alpha_{ij} V_j \tag{2}
$$

以前，我们为了构建一个序列中元素依赖的模型，主要采取RNN, RNN的模式是：每个时间步依赖前一个隐藏状态, 无法并行化。而Self-Attention将依赖关系进行矩阵化，其核心思想是把“多个依赖操作”用整体的组合表示，这样就可以用矩阵乘法一次计算所有输出。串行依赖在逻辑上是“输出依赖输入”，而矩阵化把依赖变成权重组合，计算这些组合的操作彼此独立 → 并行。 

Transformer本身的自注意力计算是完全并行的，但它天然丢失了序列顺序信息，所以必须通过位置编码（Positional Encoding）来弥补顺序依赖。如果只看数学形式，（1）=（2），区别是（2）所表示的为非线性的，（1）是一个线性化的模型，（1）还缺少一个位置编码来弥补顺序依赖。
