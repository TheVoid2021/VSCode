import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


# 模拟创建数据集
max_degree = 20  # 多项式的最大阶数   特征的维度20
n_train, n_test = 100, 100   # 训练集和测试数据集的大小
true_w = np.zeros(max_degree)  # 分配大量的空间 
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])   # 真实权重只给前4个特征赋值，后面16个特征仍旧是0，作为噪音

# np.random.normal：表示使用 NumPy 中的随机模块 random 中的正态分布函数 normal
features = np.random.normal(size=(n_train+n_test, 1)) # 生成服从正态分布的随机数数组 均值0标准差1的随机数
np.random.shuffle(features) # 用于对数组或列表进行随机重排（打乱顺序）。具体来说，它会将数组或列表中的元素按随机顺序重新排列，从而达到打乱数据集或样本顺序的效果。
# np.power() 是 NumPy 库中的一个函数，用于计算数组中元素的指数幂。具体来说，np.power(x, y) 将数组 x 中的每个元素按照指数 y 进行幂运算。
# 这个函数的功能是计算数组元素的指数幂，即x^y。其中，x 是底数数组，y 是指数数组。
# 如果 y 是一个标量，则对x中的每个元素进行相同的指数幂运算；
# 如果 y 是一个数组，则要求 x 和 y 的形状相同，对应位置的元素进行指数幂运算。
# x,y 可以是标量、数组或者其他可转换为数组的对象
# NumPy 库中的一个函数，用于创建一个等差数列的一维数组。
# 具体来说，np.arange(start, stop, step) 会生成一个从起始值 start 开始，到停止值 stop 之前（不包括 stop），步长为 step 的等差数列
# start：数列的起始值（包含在数列中）。
# stop：数列的结束值（不包含在数列中）。
# step：数列的步长，即相邻两个数之间的差值。
# 如果只提供一个参数，则默认起始值为 0，步长为 1；如果提供两个参数，则默认步长为1。
# np.arange() 函数生成的数列是左闭右开区间，即包含起始值而不包含结束值。
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
print(features.shape, np.arange(max_degree).reshape(1, -1).shape)  # (200, 1) (1, 20)
print(poly_features.shape)      # (200, 20)
print(poly_features[1, :])
for i in range(max_degree):
  # [:, i] 表示选取矩阵的所有行，但只选取第 i 列。
  poly_features[:, i] /= math.gamma(i+1)  # gamma(n)=(n-1)!
# labels的维度：（n_train+n_test,）
# np.dot() 是 NumPy 库中用于计算两个数组的点积（内积）的函数。点积是两个向量相乘并求和的结果
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# Numpy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]
print(features[:2], poly_features[:2, :], labels[:2])

# 模型训练和测试
def train_epoch_ch3(net, train_iter, loss, updater):
    #训练模型一个迭代周期（定义见第三章）#
    # 如果模型是用nn模组定义的
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式 告诉pytorch要计算梯度
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)  # 三个参数需要累加的迭代器
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
        metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    # 损失累加/总样本数  训练正确的/总样本数
    return metric[0] / metric[2], metric[1] / metric[2]


# 评估给定数据集上模型的损失
def evaluate_loss(net, data_iter, loss):
  metric = d2l.Accumulator(2)  # 损失函数的总和，样本数量
  for X, y in data_iter:
    out = net(X)
    l = loss(out, y)
    metric.add(l.sum(), l.numel())
  return metric[0]/metric[1] # 损失累加/总样本数 = 平均损失 

def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
  loss = nn.MSELoss(reduction="none")
  input_shape = train_features.shape[-1]
  # 不设置偏置值 因为已经在多项式中实现了它
  net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))  # 就一个简单的线性回归
  batch_size = min(10, train_labels.shape[0])
  train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)), batch_size)
  test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)), batch_size)
  trainer = torch.optim.SGD(net.parameters(), lr=0.01)
  animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log', xlim=[1, num_epochs], ylim=[1e-3,1e2], legend=['trian', 'test'])

  for epoch in range(num_epochs):
    train_epoch_ch3(net, train_iter, loss, trainer)
    if epoch == 0 or (epoch+1)%20 == 0:
      animator.add(epoch+1, (evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss)))
  print('weight', net[0].weight.data.numpy())


# 正常 从多项式特征中选择前4个维度，即1,x,x^2,x^3/3!  # 真实的人工定义的权重就是4个值 
print("normal")
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
d2l.plt.show()

print("overfitting")
# 过拟合 从多项式特征中选择所有维度 -- 全部20个维度，即1,x,x^2,x^3***/20! 
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
d2l.plt.show()

print('underfitting')
# 欠拟合 从多项式特征中选择前2个维度，即1,x/3! 
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
d2l.plt.show()
