import numpy as np
from d2l import torch as d2l


# 微分和积分是微积分的两个分支，前者可以应用于深度学习中的优化问题。
#• 导数可以被解释为函数相对于其变量的瞬时变化率，它也是函数曲线的切线的斜率。
#• 梯度是一个向量，其分量是多变量函数相对于其所有变量的偏导数。
#• 链式法则可以用来微分复合函数。

def f(x):
    return 3 * x ** 2 - 4 * x # f(x) = 3x^2 - 4x


def numerical_lim(f, x, h): #
    return (f(x + h) - f(x)) / h  # 导数极限：f(x+h) - f(x) / h 


# 通过逐步减小步长h，来观察数值计算中的极限值。
h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}') #  # h=0.10000, numerical limit=4.00000
    h *= 0.1    

# 绘制一个函数及其在特定点（x=1）处的切线。
x = np.arange(0, 3, 0.1)
d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
d2l.plt.show();


x = np.arange(0.5, 3, 0.2)
d2l.plot(x, [x ** 3 - 1 / x, 4 * x - 4], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
d2l.plt.show();
