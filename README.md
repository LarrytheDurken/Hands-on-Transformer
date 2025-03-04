# Hands-on-Transformer
本人的Transformer练手项目。

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
