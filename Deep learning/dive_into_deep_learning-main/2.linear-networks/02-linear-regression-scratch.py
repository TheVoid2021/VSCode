import random  #随机梯度下降和初始化权重参数时需要用到
import torch

# 线性回归的从零开始实现

# mm只能进行矩阵乘法,也就是输入的两个tensor维度只能是( n × m ) (n\times m)(n×m)和( m × p ) (m\times p)(m×p)
# bmm是两个三维张量相乘, 两个输入tensor维度是( b × n × m )和( b × m × p ), 第一维b代表batch size，输出为( b × n × p )
# matmul可以进行张量乘法, 输入可以是高维. 输入tensor维度是( b × n × m )和( b × m × p ), 第一维b代表batch size，输出为( b × n × p )

# python知识补充：
# Python3 range() 函数返回的是一个可迭代对象（类型是对象），而不是列表类型， 所以打印的时候不会打印列表。
# Python3 list() 函数是对象迭代器，可以把range()返回的可迭代对象转为一个列表，返回的变量类型为列表。
# Python3 range(start, stop[, step]) 函数返回一个数字序列，开始值start，结束值stop（不包含），步长step。
# Python3 shuffle() 方法将序列的所有元素随机排序。shuffle( 
# 人造数据集 （使用线性模型参数w = [2, −3.4]⊤、b = 4.2 和噪声项ϵ生成数据集及其标签：y = Xw + b + ϵ. 噪声项ϵ可以视为模型预测和标签时的潜在观测误差。）
def synthetic_data(w, b, nums_example):  # nums_example: 样本数
    X = torch.normal(0, 1, (nums_example, len(w))) # 生成一个nums_example行，len(w)列的符合正态分布随机数矩阵 （均值为0，方差为1）
    y = torch.matmul(X, w) + b # 矩阵乘法
    y += torch.normal(0, 0.01, y.shape)  # 加入噪声，均值为0，方差为0.01，跟y形状一样的随机数矩阵
    return X, y.reshape(-1, 1)  # y从行向量转为列向量 ，reshape(-1, 1)中-1表示根据其他维度的值自动计算列数h
# X：特征矩阵 1000*2 ，y：标签向量 1000*1

true_w = torch.tensor([2, -3.4]) # 定义真实权重 2：第一个特征的权重，-3.4：第二个特征的权重
true_b = 4.2 # 定义真实偏置 4.2：标签的偏置
# features中的每一行都包含一个二维数据样本，labels中的每一行都包含一维标签值（一个标量）。
features, labels = synthetic_data(true_w, true_b, 1000) # 生成一个1000行，2列的随机数矩阵（特征和标签）
print('第一个features:', features[0],'\n 第一个label:', labels[0])


# 读数据集 （定义一个read_data函数，该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量。每个小批量包含一组特征和标签。）
def read_data(batch_size, features, lables):
    nums_example = len(features) # 获取样本数
    indices = list(range(nums_example))  # 生成0-999的元组，然后将range()返回的可迭代对象转为一个列表
    random.shuffle(indices)  # 将样本随机打乱。（随机读取样本，没有特定的顺序）
    for i in range(0, nums_example, batch_size):  # range(start, stop, step)（从start到stop，每次取step）
        index_tensor = torch.tensor(indices[i: min(i + batch_size, nums_example)]) # 最后一次拿取不足一个批量的时候，有多少拿多少
        yield features[index_tensor], lables[index_tensor]  # yield 生成器，每次调用yield，输入一个值，返回一个值，而不是一次性生成所有值
# 通常，我们利用GPU并行运算的优势，处理合理大小的“小批量”。每个样本都可以并行地进行模型计算，且 
# 每个样本损失函数的梯度也可以被并行计算。GPU可以在处理几百个样本时，所花费的时间不比处理一个样本时多太多。

# 取十个样本查看
batch_size = 10
for X, y in read_data(batch_size, features, labels):
    print("十个X特征矩阵样本:", X, "\n十个标签向量样本y", y)    
    break;

# 初始化参数
# torch.normal是PyTorch库中的一个函数，用于生成服从正态分布（也称为高斯分布）的随机数。
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True) # 生成一个2行1列的符合正态分布随机数矩阵 （均值为0，方差为0.01），requires_grad=True 表示需要计算梯度
b = torch.zeros(1, requires_grad=True) # 生成一个1行1列的0矩阵，requires_grad=True 表示需要计算梯度 
# 设置 requires_grad=True 时，PyTorch 会跟踪该张量的所有操作，以便在反向传播时计算梯度。

# 定义线性回归模型
def net(X, w, b):
    return torch.matmul(X, w) + b 

# 定义均方损失函数
def loss(y_hat, y):
    # 这里y_hat和y的可能维度不同，所以需要将y的维度reshape为y_hat的维度
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  

# 定义优化算法
def sgd(params, batch_size, lr):  # params: 参数列表，也就是上面的list([w, b])
    with torch.no_grad():  # with torch.no_grad() 则主要是用于停止autograd模块的工作，
        for param in params:
            param -= lr * param.grad / batch_size   # /batch_size 是为了求平均梯度，而不是求总和
            param.grad.zero_()  # 梯度清零，这样下一次计算梯度的时候就不会跟上一次的梯度相关了
# torch.no_grad() 是一个上下文管理器，用于临时禁用梯度计算。更新参数不需要梯度计算！
# 以起到加速和节省显存的作用，具体行为就是停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。


# 训练模型 （在训练数据集上训练模型。在每个小批量中，通过调用net(X, w, b)计算预测值，调用loss函数计算损失，调用sgd函数更新模型参数。）
lr = 0.03  # 学习率
num_epochs = 3 # 训练次数 （迭代次数）


for epoch in range(0, num_epochs):
    for X, y in read_data(batch_size, features, labels):
        f = loss(net(X, w, b), y) # `X`和`y`的小批量损失
        # 因为`f`形状是(`batch_size`, 1)，而不是一个标量。`f`中的所有元素被加到一起，
        # 并以此计算关于[`w`, `b`]的梯度
        f.sum().backward() # 计算f的sum，并反向传播
        sgd([w, b], batch_size, lr)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels) # 计算训练集上的损失
        print("w {0} \nb {1} \n loss {2:f}".format(w, b, float(train_l.mean())))

print("w误差 ", true_w - w, "\n b误差 ", true_b - b)
