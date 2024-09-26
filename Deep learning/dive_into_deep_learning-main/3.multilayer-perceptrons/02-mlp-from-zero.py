import torch
from torch import nn
from d2l import torch as d2l
from IPython import display
import matplotlib.pyplot as plt

# 从零开始实现多层感知机
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256
# 权重初始化 随机  全部设置0或1 试试有什么区别
# 输入层到隐藏层
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True)*0.01) # *0.01?  784*256
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True)) # 256
# 隐藏层到输出层
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True)*0.01) # 256*10
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True)) # 10

# 为什么随机  可以设置为0 1 试试
# W1 = nn.Parameter(torch.ones(num_inputs, num_hiddens, requires_grad=True)) # *0.01?  行数 列数 梯度
# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# W2 = nn.Parameter(torch.ones(num_hiddens, num_outputs, requires_grad=True))
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# 定义实现relu函数
# torch.zeros_like(X) 函数  返回的变量类型及格式都与X一致，但是元素值是0 
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)  # torch.max(a, X)

# 定义网络模型
def net(X):
    X = X.reshape((-1, num_inputs)) # -1 表示根据其他维度计算该维度大小，这里就是batch_size  num_inputs->784
    H = relu(X @ W1 + b1) # 矩阵乘法 @符号简写  【256*784】@【784*256】+【256】
    # O = H@W2 + b2
    return (H @ W2 + b2) # 矩阵乘法 @符号简写  【256*256】@【256*10】+【10】

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none') # reduction='none' 不进行求和（默认求和）  返回每个样本的损失
# loss = nn.CrossEntropyLoss(reduction='sum')

# 分类精度--判断预测的类别是否是正确的
def accuracy(y_hat, y):
    # 计算预测正确的数量
    # 如果预测值是二维矩阵，而且每一行列数大于1
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 每一行元素预测值最大的下标存到y_hat中 argmax返回的是索引下标--获取概率最高类别的索引
        y_hat = y_hat.argmax(axis=1)
    # print(y_hat.dtype, y_hat.type(torch.float).dtype)
    # y_hat.type(y.dtype) 设置元素数值类型和y保持一致
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())  # 预测类型成功的个数
# print(accuracy(y_hat, y)/len(y))  # 预测成功率

def evaluate_accuracy(net, data_iter):
    # 计算在指定数据集上模型的精度
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
        # net.eval() 作用1. 网络结构如果有Batch Normalization 和 Dropout 层的，做模型评测必须加上这个命令以防止报错
        # 作用2： 关闭梯度计算
    # Accumulator 累加器 不断迭代数据X y 不断累加评测结果
    # Accumulator 初始化传入数值2 表示有数组有两个变量，第一个位置存正确测试数，第二个位置存样本总数，根据批次遍历依次累加
    metric = Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter: # 迭代一次 一次批量
            # metric.add(该批次的预测类别正确的样本数，批次的总样本数)
            # y.numel() pytorch中用于返回张量的元素总个数
            metric.add(accuracy(net(X), y), y.numel())

    # 返回值=分类正确样本数/样本总数=精度
    return metric[0] / metric[1]

class Accumulator:
    # 在n个变量上累加
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        # print(self.data, args)
        self.data = [a + float(b) for a,b in zip(self.data, args)]
        # print(self.data, args)
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    # 在动画中绘制数据
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g--', 'r:'),
                 nrows=1, nclos=1, figsize=(3.5, 2.5)):

        # 增量的绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, nclos, figsize=figsize)
        if nrows * nclos == 1:
            self.axes = [self.axes, ]
        # 使用lamda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a,b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        # 通过以下两行代码实现了在PyCharm中显示动图
        plt.draw()
        plt.pause(interval=0.001)
        display.clear_output(wait=True)

def trian_epoch_ch3(net, train_iter, loss, updater):
    # 训练模型一个迭代周期（定义见第三章）
    # 如果模型是用nn模组定义的
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式 告诉pytorch要计算梯度
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)  # 三个参数需要累加的迭代器
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y) # 计算损失函数
        # 如果优化器是pytorch内置的优化器
        # 下面两个加的结果有什么区别
        # print(float(l)*len(y), accuracy(y_hat,y), y.size().numel(),
        #       float(l.sum()), accuracy(y_hat, y), y.numel())
        if isinstance(updater, torch.optim.Optimizer):
            # 使用pytorch内置的优化器和损失函数
            updater.zero_grad() # 1.梯度先置零
            l.mean().backward() # 2.计算梯度
            updater.step()      # 3.调用step参数进行一次更新
            # metric.add(float(l)*len(y), accuracy(y_hat,y), y.size().numel())
            # 报错 ValueError: only one element tensors can be converted to Python scalars
        else:
            # 使用定制的优化器和损失函数
            # 自己实现的l是一个向量
            l.sum().backward()
            updater(X.shape[0])
            # metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    # 损失累加/总样本数  训练正确的/总样本数
    return metric[0] / metric[2], metric[1] / metric[2]

# 开启训练
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    # 训练模型
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3,0.9],
                        legend=['train-loss', 'train-acc', 'test-acc'])
    for epoch in range(num_epochs):
        train_metrics = trian_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics+(test_acc,))
    train_loss, train_acc = train_metrics
    print(train_loss)
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# 多层感知机的训练过程跟之前的softmax回归的训练过程完全相同 多了一个隐藏层，训练损失下降了，但是训练精度没有提升
num_epochs, lr = 10, 0.03
updater = torch.optim.SGD(params, lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()


# 预测标签
def predict_ch3(net, test_iter, n):  #@save
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    
predict_ch3(net, test_iter, 6)
d2l.plt.show()

# Traceback (most recent call last):
#   File "e:\desktop\vsCode\Python\dive_into_deep_learning-main\ch03\02-mlp-from-zero.py", line 176, in <module>
#     train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
#   File "e:\desktop\vsCode\Python\dive_into_deep_learning-main\ch03\02-mlp-from-zero.py", line 136, in train_ch3
#     assert train_loss < 0.5, train_loss
#            ^^^^^^^^^^^^^^^^
# AssertionError: 0.6467081826527913

# 这个错误是由于训练损失 train_loss 大于 0.5 导致的断言失败。
# 具体来说，train_loss 的值为 0.6467，不满足 assert train_loss < 0.5 的条件。

# 解决方案：将学习率提高到0.03