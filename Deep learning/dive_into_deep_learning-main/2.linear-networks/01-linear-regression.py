import math
import time
import numpy as np
import torch
from d2l import torch as d2l

# 机器学习模型中的关键要素是训练数据、损失函数、优化算法，还有模型本身。
#• 矢量化使数学表达上更简洁，同时运行的更快。
#• 最小化目标函数和执行极大似然估计等价。
#• 线性回归模型也是一个简单的神经网络。

# 矢量化加速
# 在训练我们的模型时，我们经常希望能够同时处理整个小批量的样本。为了实现这一点，需要我们对计算进
# 行矢量化，从而利用线性代数库，而不是在Python中编写开销高昂的for循环。

#为了说明矢量化为什么如此重要，我们考虑对向量相加的两种方法：在一种方法中，我们将使用Python的for循环遍历向量；
# 在另一种方法中，我们将依赖对+的调用。
n = 10000
a = torch.ones(n)  #实例化两个全为1的10000维向量。
b = torch.ones(n)

#对工作负载进行基准测试。
#定义一个计时器：
class Timer: #@save
    #记录多次运行时间#  
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        #启动计时器#
        self.tik = time.time()
    def stop(self):
        #停止计时器并将时间记录在列表中#
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        #返回平均时间#
        return sum(self.times) / len(self.times)
    def sum(self):
        #返回时间总和#
        return sum(self.times)
    def cumsum(self):
        #返回累计时间#
        return np.array(self.times).cumsum().tolist()
    
c = torch.zeros(n)
timer = Timer()
# 1.使用for循环，每次执行一位的加法。
for i in range(n):
    c[i] = a[i] + b[i]
print(c)
print("{0:.5f} sec".format(timer.stop()))

# 2.或者，我们使用重载的+运算符来计算按元素的和。
timer.start()
d = a + b
print(d)
print("{0:.5f} sec".format(timer.stop()))

# 第二种方法比第一种方法快得多。矢量化代码通常会带来数量级的加速。另外，我们将更多的
# 数学运算放到库中，而无须自己编写那么多的计算，从而减少了出错的可能性。




# 正态分布与平方损失
# 定义一个Python函数来计算正态分布。
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp((- 0.5 / sigma ** 2) * (x - mu) ** 2)


## 可视化正态分布
x = np.arange(-7, 7, 0.01) # 生成从-7到7，步长为0.01的等差数列。
params = [(0, 1), (0, 2), (3, 1)]  # 均值和标准差对
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5),legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
d2l.plt.show()

# 改变均值会产生沿x轴的偏移，增加方差将会分散分布、降低其峰值。
