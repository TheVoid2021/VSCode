# 生成一个网络，其中包含一个具有256个单元和ReLU激活函数的全连接隐藏层
# 然后是一个具有10个隐藏单元且不带激活函数的全连接输出层
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20,256), nn.ReLU(), nn.Linear(256,10))

X = torch.rand(2, 20)
net(X)

"""
    自定义块：
    1、将输入数据作为其前向传播函数的参数。
    2、通过前向传播函数来生成输出（输出的形状可能与输入的形状不同）。
    3、计算其输出关于输入的梯度，可通过其反向传播函数进行访问。
    4、存储和访问前向传播计算所需的参数。
    5、根据需要初始化模型参数。
"""
"""
    多层感知机：
    1、输入是一个20维的输入。
    2、具有256个隐藏单元的隐藏层和一个10维输出层。
"""
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接层
    def __init__(self):
        # 调用MLP父类Module的构造函数来执行必要的初始化
        # 这样，在类实例化中也可以指定其他函数的参数，例如模型参数params
        super().__init__()
        self.hidden = nn.Linear(20, 256) # 隐藏层
        self.out = nn.Linear(256, 10) # 输出层
        
    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块定义
        return self.out(F.relu(self.hidden(X)))
net = MLP()
net(X)


"""
    顺序块（相当于Sequential类）：
    1、一种将块逐个追加到列表中的函数；
    2、一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。
"""
class MySequential(nn.Module):
    def __init__(self, *args):
        # __init__函数将每个模块逐个添加到有序字典_modules中
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # _module的类型是OrderedDict，在模块的参数初始化过程中， 系统知道在_modules字典中查找需要初始化参数的子块。
            self._modules[str(idx)] = module
            
    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)

"""
   实现FixedHiddenMLP模型类：
   1、实现一个隐藏层， 其权重（self.rand_weight）在实例化时被随机初始化，之后为常量。
   2、这个权重不是一个模型参数，因此它永远不会被反向传播更新。
   3、神经网络将这个固定层的输出通过一个全连接层。
   4、返回输出时做一个循环操作，输出需小于1
"""
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20 ,20)
        
    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
net = FixedHiddenMLP()
net(X)


# 混合搭配各种组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
    
    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
