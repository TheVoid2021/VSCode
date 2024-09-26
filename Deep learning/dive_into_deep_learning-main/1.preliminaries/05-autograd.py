import torch

# 深度学习框架可以自动计算导数：我们首先将梯度附加到想要对其计算偏导数的变量上，然后记录目标值的计算，执行它的反向传播函数，并访问得到的梯度。
# 梯度永远指向值变化最大的那个方向

#一个标量函数关于向量x的梯度是向量，并且与x具有相同的形状。
print('1.自动梯度计算')
x = torch.arange(4.0) #arange:创建一个长为4的向量，0.，1.，2.，3.
x.requires_grad_(True)  # 等价于x = torch.arange(4.0, requires_grad=True)  将梯度附加到想要对其计算偏导数的变量 
print('x:', x) 
print('x.grad:', x.grad) 
y = 2 * torch.dot(x, x)  # 记录目标值的计算 （dot:计算两个张量x和x的点积）
print('y:', y)
y.backward()  # 调用反向传播函数来自动计算y关于x每个分量的梯度
print('x.grad:', x.grad)  
print('x.grad == 4*x:', x.grad == 4 * x) #快速验证这个梯度是否计算正确。


# 计算另一个函数
x.grad.zero_() # 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
y = x.sum() # 1.sum：计算张量中所有元素的和
print('y:', y)
y.backward()  # 2.调用反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度。
print('x.grad:', x.grad)


# 非标量变量的反向传播
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()  # 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
print('x:', x)
y = x * x  #一般情况下，不会对向量进行求导，所以会先求和变成标量再求导
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward() #反向传播并求和
print('x.grad:', x.grad)


print('2.Python控制流的梯度计算')
#使用自动微分的一个好处是：即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。
def f(a):  # 一个简单的函数
    b = a * 2
    print(b.norm())
    while b.norm() < 1000:  # 求L2范数：元素平方和的平方根
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.tensor(2.0)  # 初始化变量
a.requires_grad_(True)  # 1.将梯度赋给想要对其求偏导数的变量
print('a:', a)
d = f(a)  # 2.记录目标函数
print('d:', d)
d.backward()  # 3.执行目标函数的反向传播函数
print('a.grad:', a.grad)  # 4.获取梯度


print('3.分离计算')
# 在某层网络需要把参数固定的时候，会用到这个功能
# 在PyTorch中，y.detach()是一个用于从计算图中分离张量的方法。计算图是PyTorch用于自动微分的关键概念，用于构建和跟踪张量之间的操作。
# 在计算图中，张量的计算历史被记录下来，以便在反向传播时计算梯度。但有时我们希望从计算图中分离出某个张量，使其不再与原始的计算历史关联。
# 这在某些情况下是很有用的，例如当我们只关心使用该张量进行正向传播的结果，并且不需要计算梯度时。
# 当调用y.detach()时，将返回一个与y相同数据但不再与计算图关联的新张量。这意味着对返回的张量进行操作不会对原始计算图产生任何影响，也不会计算任何梯度。
# 该方法可用于将张量作为输入传递给不允许梯度传播的函数或模型。

x.grad.zero_()
y = x * x
u = y.detach()  # 把y看成一个常数赋给u u与x无关，是一个常数
print(u)
z = u * x  # z对x的导数 
z.sum().backward()
print(x.grad)
print(u == x.grad)

# u是常数不是x的函数，y还是x的函数，还是可以对y求x的导数
x.grad.zero_()
y.sum().backward()
print(x.grad)
print(x.grad == 2*x)
