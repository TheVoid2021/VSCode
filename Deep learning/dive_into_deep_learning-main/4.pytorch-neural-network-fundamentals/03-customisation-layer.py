"""
    自定义CenteredLayer类：(不带参数)
    1、要从其输入中减去均值；
    2、要构建它，我们只需继承基础层类并实现前向传播功能。
"""
import torch
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        return X - X.mean()

"""
    自定义带参数的全连接层：
    1、该层需要两个参数，一个用于表示权重，另一个用于表示偏置项；
    2、使用修正线性单元作为激活函；
    3、in_units和units，分别表示输入数和输出数。
"""
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
