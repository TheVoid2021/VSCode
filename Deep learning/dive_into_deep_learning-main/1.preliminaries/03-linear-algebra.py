import torch

# 标量、向量、矩阵和张量是线性代数中的基本数学对象。
# 标量是一个值，可以是向量中的一个分量，也可以是矩阵中的一个元素。
# 向量是一组标量，可以看作是列向量或行向量，是矩阵中的一行或者一列，也可以是矩阵本身。
# 矩阵是向量的组合，是张量的一种特殊形式。
# 张量是矩阵的推广，可以具有任意数量的轴。在深度学习中，我们经常用张量来表示数据。

#• 向量泛化自标量，矩阵泛化自向量。
#• 标量、向量、矩阵和张量是分别具有零、一、二和任意数量的轴。
#• 一个张量可以通过sum和mean沿指定的轴降低维度。
#• 两个矩阵的按元素乘法被称为他们的Hadamard积。它与矩阵乘法不同。
#• 在深度学习中，我们经常使用范数，如L1范数、L2范数和Frobenius范数。
#• 我们可以对标量、向量、矩阵和张量执行各种操作。


print('1.标量与变量')
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y, x * y, x / y, x ** y)

x = torch.arange(4)
print('2.向量')
print('x:', x)
print('x[3]:', x[3])  # 通过张量的索引来访问任一元素
print('向量的形状:', x.shape)  # 向量（用张量表示）的形状
print('向量的长度:', len(x))  # 长度
z = torch.arange(24).reshape(2, 3, 4)  # 两个维度的三行四列（相当于把两个三行四列的矩阵重叠在一起，有两层）
print('三维张量的长度:', len(z))

print('3.矩阵')
A = torch.arange(20).reshape(5, 4)
print('A:', A)
print('A.shape:', A.shape) # 矩阵的形状
print('A.shape[-1]:', A.shape[-1]) # 获取最后一个维度的大小
print('A.T:', A.T)  # 矩阵的转置

print('4.矩阵的计算')
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print('A:', A)
print('B:', B)
print('A + B:', A + B)  # 矩阵相加
print('A * B:', A * B)  # 矩阵相乘  

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print('X:', X)
print('a + X:', a + X)  # 矩阵的每一个值加上标量
print('a * X:', a * X)  # 矩阵的每一个值乘以标量
print((a * X).shape) # 矩阵的形状不变

print('5.矩阵的sum运算')
print('A:', A)
print('A.shape:', A.shape)
print('A.sum():', A.sum()) # 汇总所有元素
print('A.sum(axis=0):', A.sum(axis=0))  # 汇总所有行以生成输出向量
print('A.sum(axis=1):', A.sum(axis=1))  # 汇总所有列以生成输出向量
print('A.sum(axis=1, keepdims=True)', A.sum(axis=1, keepdims=True))  # 计算总和保持轴数不变 利于广播机制（广播机制：如果两个数组的形状不相同，它们需要通过广播机制扩展到相同的形状）
print('A.sum(axis=[0, 1]):', A.sum(axis=[0, 1]))  # 汇总所有元素以生成输出标量
print('A.mean():', A.mean())  # 计算平均值
print('A.sum() / A.numel():', A.sum() / A.numel())   # 计算平均值的另一种方法

print('6.向量-向量相乘（点积）')
x = torch.arange(4, dtype=torch.float32)   # 4*1维  0123
y = torch.ones(4, dtype=torch.float32)  # 4*1维  1111
print('x:', x)
print('y:', y)
print('向量-向量点积:', torch.dot(x, y))  # 0*1+1*1+2*1+3*1=6  dot:点积(标量)

print('7.矩阵-向量相乘(向量积)')
print('A:', A)  # 5*4维
print('x:', x)  # 4*1维
print('torch.mv(A, x):', torch.mv(A, x)) # 5*4维 * 4*1维 = 5*1维（长为5的向量）  mv:矩阵-向量相乘

print('8.矩阵-矩阵相乘(向量积)')
print('A:', A)  # 5*4维
B = torch.ones(4, 3)  # 4*3维
print('B:', B)  
print('torch.mm(A, B):', torch.mm(A, B))  # mm:矩阵-矩阵相乘

# 目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。
# 向量的范数是表示一个向量有多大 在线性代数中，向量范数是将向量映射到标量的函数f
print('9.范数')
u = torch.tensor([3.0, -4.0])
print('向量的𝐿2范数:', torch.norm(u))  # 向量的𝐿2范数 √(3^2+(-4)^2)=5 (norm：计算向量元素平方和的平方根)  
print('向量的𝐿1范数:', torch.abs(u).sum())  # 向量的𝐿1范数 |3|+|-4|=7 (abs sum：计算向量元素的绝对值之和)
v = torch.ones((4, 9))
print('v:', v)
print('矩阵的𝐿2范数:', torch.norm(v))  # 矩阵的𝐿2范数 √(1^2+1^2+1^2+1^2+1^2+1^2+1^2+1^2+1^2)=3 （F范数：矩阵的𝐿2范数（矩阵元素平方和的平方根））

print('10.根据索引访问矩阵')
y = torch.arange(10).reshape(5, 2)
print('y:', y)
index = torch.tensor([1, 4]) 
print('y[index]:', y[index])  # 索引为1和4的行

print('11.理解pytorch中的gather()函数')
a = torch.arange(15).view(3, 5)
print('11.1二维矩阵上gather()函数')
print('a:', a)
b = torch.zeros_like(a)
b[1][2] = 1  ##给指定索引的元素赋值
b[0][0] = 1  ##给指定索引的元素赋值
print('b:', b)
c = a.gather(0, b)  # dim=0 gather()函数根据索引矩阵b，从a中提取元素 
d = a.gather(1, b)  # dim=1 gather()函数根据索引矩阵b，从a中提取元素
print('c:', c)
print('d:', d)
print('11.2三维矩阵上gather()函数')
a = torch.randint(0, 30, (2, 3, 5))
print('a:', a)
index = torch.LongTensor([[[0, 1, 2, 0, 2],
                           [0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1]],
                          [[1, 2, 2, 2, 2],
                           [0, 0, 0, 0, 0],
                           [2, 2, 2, 2, 2]]])
print(a.size() == index.size()) # True
b = torch.gather(a, 1, index)
print('b:', b)
c = torch.gather(a, 2, index)
print('c:', c)
index2 = torch.LongTensor([[[0, 1, 1, 0, 1],
                            [0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0]]])
d = torch.gather(a, 0, index2)
print('d:', d)

print('12.理解pytorch中的max()和argmax()函数')
a = torch.tensor([[1, 2, 3], [3, 2, 1]])
b = a.argmax(1)  
c = a.max(1)
d = a.max(1)[1]
print('a:', a)  # 2*3维
print('a.argmax(1):', b) #argmax(1)函数返回的是索引  第一维的最大值索引 第二维的最大值索引
print('a.max(1):', c) #max(1)函数返回的是 torch.return_types.max(values=tensor([3, 3]),indices=tensor([2, 0]))
print('a.max(1)[1]:', d) #max(1)[1]:返回的是索引 第一维的最大值索引 第二维的最大值索引

print('13.item()函数')
a = torch.Tensor([1, 2, 3])
print('a[0]:', a[0])  # 直接取索引返回的是tensor数据 张量
print('a[0].item():', a[0].item())  # 获取python number 标量
