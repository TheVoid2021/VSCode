import torch 
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

"""
    检查第二个全连接层的参数：
    1、这个全连接层包含两个参数，分别是该层的权重和偏置。
    2、两者都存储为单精度浮点数（float32）。
"""
print(net[2].state_dict())

# 访问目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)

"""
    参数初始化：
    1、内置初始化
    2、自定义初始化
"""
# 将所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

# model.apply(fn)会递归地将函数fn应用到父模块的每个子模块submodule，也包括model这个父模块自身
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]

# 将所有参数初始化为给定的常数，比如初始化为1
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
        
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]

# 使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42。
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
# net[0].weight[0:]

# 在多个层间共享参数： 我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。
# 我们需要给共享层一个名称，以便可以引用它的参数
"""
    参数绑定：
    1、第三个和第五个神经网络层的参数是绑定的。 它们不仅值相等，而且由相同的张量表示。
    2、如果我们改变其中一个参数，另一个参数也会改变。
    3、由于模型参数包含梯度，因此在反向传播期间第二个隐藏层 （即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。
"""
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8 ,1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不是只有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
