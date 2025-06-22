# Hands-on-Transformer
本人的Transformer练手项目。

# Attention理解

## Self-Attention

### 注意力分数的含义
以单样本为例，假设 $Q$ 为：
$$
\begin{bmatrix}
q_1 \\
q_2 \\
q_3
\end{bmatrix}
$$

$K^T$ 为：
$$
\begin{bmatrix}
k_1 & k_2 & k_3
\end{bmatrix}
$$

则注意力分数 $\alpha$ 则为：
$$
\alpha = QK^T = \begin{bmatrix}
q1 \cdot k1 & q1 \cdot k2 & q1 \cdot k3 \\
q2 \cdot k1 & q2 \cdot k2 & q2 \cdot k3 \\
q3 \cdot k1 & q3 \cdot k2 & q3 \cdot k3
\end{bmatrix}
$$

矩阵中每个元素的含义：
+ 第1行（ $q_1$ 与所有 $k$ 的点积）：表示 $word_1$ 对 $word_1$、$word_2$、$word_3$ 的注意力原始分数
+ 第2行（ $q_2$ 与所有 $k$ 的点积）：表示 $word2$ 对 $word_1$ 、$word_2$、$word_3$ 的注意力原始分数
+ 以此类推

### 注意力分数与V矩阵的相乘
**注意力分数矩阵：**
$$
\text{Attention Weights} = \begin{bmatrix}
\alpha_{11} & \alpha_{12} & \alpha_{13} \\
\alpha_{21} & \alpha_{22} & \alpha_{23} \\
\alpha_{31} & \alpha_{32} & \alpha_{33}
\end{bmatrix}
$$
其中 $\alpha_{ij}$ 表示 $word_i$ 对 $word_j$ 的注意力权重（每行和为 1）。

**$V$ 矩阵：**
$$
\begin{bmatrix}
v_1 \\
v_2 \\
v_3
\end{bmatrix}
$$

**矩阵相乘过程：**
$Output$ 矩阵的每一行是注意力权重的对应行与 $V$ 的列的加权和：
$$
\text{Output} = \begin{bmatrix}
\alpha_{11} \cdot v_1 + \alpha_{12} \cdot v_2 + \alpha_{13} \cdot v_3 \\
\alpha_{21} \cdot v_1 + \alpha_{22} \cdot v_2 + \alpha_{23} \cdot v_3 \\
\alpha_{31} \cdot v_1 + \alpha_{32} \cdot v_2 + \alpha_{33} \cdot v_3
\end{bmatrix}
$$

+ 第一行是 $word_1$ 的上下文向量：融合了 $word_1$ 对 $word_1$、$word_2$、$word_3$ 的注意力权重，加权后的 $V$ 信息
+ 第二行是 $word_2$ 的上下文向量：同理，融合了 $word_2$ 对所有词的注意力
+ 最终，每个位置的输出都包含了序列中所有词的信息（通过注意力权重加权），这是 Transformer 能捕捉长距离依赖的关键

# Tips

## 搭建逻辑
+ 整体到局部
+ SublayerConnection的实现
    + 匿名函数实现残差连接
+ Transformer的作用机理
    + Encoder和Decoder的输入输出是什么
    + Generator的输入输出是什么（自回归特性）
    + src_mask和tgt_mask的作用分别是什么
+ 超参数的意义
    +  src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout
+ qkv的具体含义

## 张量的shape变化
+ src，tgt及embedding后的shape
+ 注意力机制内部的shape
    + qkv在：进入多头注意力机制前的shape、self_attn和src_attn的shape
    + p_attn的shape
    + 注意力机制输出的是什么及其shape
+ mask的shape

## pytorch工程细节
+ view()的作用机理
    + 在MultiHeadedAttention中的使用
    + tensor.contiguous()的意义
+ 多处unsqueeze()的出现和广播机制
    + 在MultiHeadedAttention中的使用（针对mask）
    + 在PositionalEncoding中的使用（针对position）
+ zip()的作用机理
    + 在MultiHeadedAttention中的使用
+ register_buffer的作用机理
    + 在PositionalEncoding中的使用
+ tensor.type_as的作用机理
