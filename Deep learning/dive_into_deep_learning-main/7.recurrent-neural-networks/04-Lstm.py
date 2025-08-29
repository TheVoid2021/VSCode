import torch
from torch import nn
from d2l import torch as d2l

# 设置批量大小和时间步数
batch_size, num_steps = 32, 35
# 加载时间机器数据集
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

"""
定义和初始化模型参数。 如前所述，超参数num_hiddens定义隐藏单元的数量。 
我们按照标准差0.01的高斯分布初始化权重，并将偏置项设为0。
"""
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    # 定义一个正态分布初始化函数
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    # 定义一个返回三个参数的函数
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 初始化LSTM的各个门和记忆元的参数
    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 在[初始化函数]中， 长短期记忆网络的隐状态需要返回一个额外的记忆元， 单元的值为0，形状为（批量大小，隐藏单元数）。
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

"""
定义实际的模型。 
1. 输入门、 遗忘门、 输出门和候选记忆元的计算。
2. 记忆元的计算。
3. 隐藏状态的计算。
4. 输出层的计算。
5. 将输出拼接成一个张量， 并返回最终的隐藏状态和输出。
提供三个门和一个额外的记忆元。 请注意，只有隐状态才会传递到输出层， 
而记忆元不直接参与输出计算。
"""
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        # 计算输入门
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        # 计算遗忘门
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        # 计算输出门
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        # 计算候选记忆元
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        # 更新记忆元
        C = F * C + I * C_tilda
        # 更新隐藏状态
        H = O * torch.tanh(C)
        # 计算输出
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

# 设置模型参数
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
# 创建并训练模型
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# 显示训练结果
d2l.plt.show()