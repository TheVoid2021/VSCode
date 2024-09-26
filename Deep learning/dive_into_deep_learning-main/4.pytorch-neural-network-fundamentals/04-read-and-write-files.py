import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')

# 存储一个张量列表，然后把它们读回内存。
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)

# 写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2

"""
    加载和保存模型参数：
    1、深度学习框架提供了内置函数来保存和加载整个网络；
    2、保存模型的参数而不是保存整个模型。
"""
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
    
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 为了恢复模型，我们实例化了原始多层感知机模型的一个备份
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
