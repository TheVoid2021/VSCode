import torch
from d2l import torch as d2l

# 如何结合非线性函数来构建具有更强表达能力的多层神经网络架构
# 多层感知机在输出层和输入层之间增加一个或多个全连接隐藏层，并通过 -激活函数- 转换隐藏层的输出。
# 常用的激活函数包括ReLU函数、sigmoid函数和tanh函数。

# Relu函数
# ReLU函数通过将相应的活性值设为0，仅保留正元素并丢弃所有负元素。
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
d2l.plt.show();

# ReLU函数的导数
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
d2l.plt.show();

# sigmoid函数
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
d2l.plt.show();

# sigmoid函数的导数
x.grad.zero_() # 清除以前的梯度
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
d2l.plt.show();

# tanh函数
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
d2l.plt.show();

# tanh函数的导数
x.grad.zero_()
y.sum().backward()
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
d2l.plt.show();

