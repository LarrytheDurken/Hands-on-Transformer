import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import math
import copy


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # 用于将输入转换为向量
        self.tgt_embed = tgt_embed  # 用于将输出转换为向量
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 1️⃣ src和tgt的shape: (batch_size, seq_len)，这两个一般是词ID（整数序列，代表词表中的索引），seq_len是序列的最大长度
        # 2️⃣ src和tgt经过embedding后变为词向量的形式，故shape变为(batch_size, seq_len, d_model)
        # 3️⃣ 最后经过全连接层(Generator)后shape一般为(batch_size, seq_len, vocab), 因此要对最后一层（-1）进行softmax

        # 由于一个批次中的序列长度往往不一样，因此需要对每个序列进行填充，所以要消除原序列被填充部分的影响
        # 除了消除填充部分影响，同时屏蔽目标序列中的后续位置，确保模型在生成过程中遵循自回归原则
        return self.decode(self.encoder(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):

    def __init__(self, d_model, vocab):
        # 将词向量映射为具体词汇的概率
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    # 要对最后的输出进行一个归一化
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2*(x-mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    # 由于Encoder层中每个子层内部都包含层归一化和残差连接，因此该类将这两个定义为范式，通过改变sublayer实现不同的功能
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer): # sublayer是一个function
        return x + self.dropout(sublayer(self.norm(x)))  # 传统Transformer是后归一化，这里进行了改进


class EncoderLayer(nn.Module):
    # Encoder层主要包含两个子层
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size  # 这个size是为了便于创建Encoder类中的norm

    def forward(self, x, mask):
        # 注意这个匿名函数：这要追溯到SublayerConnection中的残差连接中
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # x已经在第一个sublayer中经过了mask，故不用再来一次
        return self.sublayers[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    # attention只是一种矩阵运算，不需要更新参数

    # qkv的shape一般为(batch_size, h, seq_len, d_k)
    # 即使shape存在区别，但最后两维也一定为(seq_len, d_k)，因为注意力机制的目的就是检测序列之间的关系
    d_k = query.size(-1)
    # scores的shape一般(batch_size, h, seq_len, seq_len)，表示每个token对其他token的注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)  # softmax后的注意力分布
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):  # query，key，value哪来的？？？？？？？？？？？？？？？？？？？？？？？？？
        if mask is not None:
            mask = mask.unsqueeze(1)  # 为什么？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        nbatches = query.size(0)

        # qkv的shape: (batch_size, seq_len, d_model)
        query, key, value = [
            # 这里不能写成view(nbatches, self.h, -1, self.d_k)：
            # 1.view的作用机理：按照内存存储顺序（行优先）对张量重新塑形
            # 2.如果如此写，会导致不同sequence的token混叠
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))  # 注意zip的用法
        ]
        # qkv的shape: (batch_size, h, seq_len, d_model)

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()  # 保证内存空间连续
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    # 作为Encoder和Decoder子层的FFN
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 数值缩放以确保稳定性：
        # 1.扩大嵌入向量的数值范围，使其与后续位置编码的幅值匹配（位置编码的每个维度幅值约为1）
        # 2.后续attention计算中除以sqrt(d_model)，故前后保持数值稳定
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # 为了下边可以利用广播机制，扩展一个维度  
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)  # 广播
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加批次的维度
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(
        src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    # 注意这里没有mask的事
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model