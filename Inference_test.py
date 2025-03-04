import torch
from Transformer import make_model, subsequent_mask

def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()

    # 单序列
    # src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # src_mask = torch.ones(1, 1, 10)

    # 多序列
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]])
    src_mask = torch.tensor([
                                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                                [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
                            ])

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(2, 1).type_as(src)  # type_as：使该张量的数据类型和设备设置与括号内的张量一致

    # 单序列
    '''for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )'''
    
    # 多序列
    for i in range(9):
        out = test_model.decode(  # out的shape: (batch_size, seq_len(tgt), d_model)
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])  # prob的shape: (batch_size, vocab_size)，下一个词的概率分布
        _, next_words = torch.max(prob, dim=1)  # 获取batch中所有预测结果
        # 将下一个词拼在tgt后边
        ys = torch.cat(
            [ys, next_words.unsqueeze(1)],  # 保持维度一致
            dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


run_tests()

