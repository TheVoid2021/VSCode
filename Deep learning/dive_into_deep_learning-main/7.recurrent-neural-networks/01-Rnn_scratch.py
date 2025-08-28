import math  
import torch  
from torch import nn  
from torch.nn import functional as F  
from d2l import torch as d2l  

# 设置批量大大小和时间步长
batch_size, num_steps = 32, 35
# 加载时间机器数据集
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# ------------------------------
"""
使用one-hot编码将标签转换为向量---独热编码
为了将词表示成向量输入到神经网络，一个简单的办法是使用one-hot向量。
假设词典中不同字符的数量为N（即词典大小`vocab_size`），
每个字符已经同一个从0到N-1的连续整数值索引一一对应。
如果一个字符的索引是整数i, 那么我们创建一个全0的长为N的向量，
并将其位置为i的元素设成1。该向量就是对原字符的one-hot向量。
"""
F.one_hot(torch.tensor([0, 2]), len(vocab))

# 创建一个2x5的矩阵，并进行转置和one-hot编码
X = torch.arange(10).reshape((2, 5))
F.one_hot(X.T, 28).shape

"""
初始化循环神经网络模型的模型参数
隐藏单元个数 num_hiddens是一个超参数。
"""
def get_params(vocab_size, num_hiddens, device):

    """获取模型参数"""
    num_inputs = num_outputs = vocab_size
  # 输入和输出的维度都是词表大小
    def normal(shape):
        """生成正态分布的参数"""
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))  # 输入到隐藏层的权重
    W_hh = normal((num_hiddens, num_hiddens))  # 隐藏层到隐藏层的权重
    b_h = torch.zeros(num_hiddens, device=device)  # 隐藏层的偏置
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))  # 隐藏层到输出层的权重
    b_q = torch.zeros(num_outputs, device=device)  # 输出层的偏置
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)  # 启用梯度计算
    return params

"""
根据循环神经网络的计算表达式实现该模型。
首先定义init_rnn_state函数来返回初始化的隐藏状态。
它返回由一个形状为(批量大小, 隐藏单元个数)的值为0的NDArray组成的元组。
使用元组是为了更便于处理隐藏状态含有多个NDArray的情况。
隐变量在循环中是自我不断变化的， 因此也称作隐状态。
"""
def init_rnn_state(batch_size, num_hiddens, device):
    """初始化循环神经网络的状态"""
    return (torch.zeros((batch_size, num_hiddens), device=device), )

"""
rnn函数定义了在一个时间步里如何计算隐藏状态和输出。
这里的激活函数使用了tanh函数。
当元素在实数域上均匀分布时，tanh函数值的均值为0。
"""
def rnn(inputs, state, params):
    """循环神经网络的前向传播"""
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)  # 隐藏层状态更新
        Y = torch.mm(H, W_hq) + b_q  # 输出层计算
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# 创建一个类来包装上面这些函数， 并存储从零开始实现的循环神经网络模型的参数。
class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)  # 将输入转换为one-hot编码
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

# 输出形状是（时间步数批量大小，词表大小）， 而隐状态形状保持不变，即（批量大小，隐藏单元数）。
num_hiddens = 512  # 设置隐藏层大小
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape

"""
定义预测函数来生成prefix（含有数个字符的字符串）后面num_preds个字符。
这个函数将逐步生成一个字符， 其中基于之前的所生成的字符， 来预测下一个字符。
在循环遍历prefix中的开始字符时， 我们不断地将隐状态传递到下一个时间步，但是不生成任何输出。 
这被称为预热（warm-up）期， 因为在此期间模型会自我更新（例如，更新隐状态）， 但不会进行预测。 
预热期结束后，隐状态的值通常比刚开始的初始值更适合预测， 从而预测字符并输出它们。
隐变量在循环中是自我不断变化的， 因此也称作隐状态。
"""
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())

"""
裁剪梯度 解决梯度爆炸问题
"""
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

"""
在训练模型之前，让我们[定义一个函数在一个迭代周期内训练模型]。 
跟之前章节的模型训练函数相比，这里的模型训练函数有以下几点不同：

1. 使用困惑度评价模型。
2. 在迭代模型参数前裁剪梯度。
3. 对时序数据采用不同采样方法将导致隐藏状态初始化的不同。相关讨论可参考（语言模型数据集（周杰伦专辑歌词））。

另外，考虑到后面将介绍的其他循环神经网络，为了更通用，这里的函数实现更长一些。
"""
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

"""
训练模型
"""
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)

d2l.plt.show()