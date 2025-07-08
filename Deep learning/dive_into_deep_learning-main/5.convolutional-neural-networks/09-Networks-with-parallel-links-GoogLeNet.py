import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 路径1，单1×1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 路径2，1×1卷积层后接3×3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 路径3，1×1卷积层后接5×5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 路径4，3×3最大汇聚层后接1×1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))
        # 在通道维度上连接输出
        return torch.cat((p1, p2, p3, p4), dim=1)

# GoogLeNet一共使用9个Inception块和平均汇聚层的堆叠
# 第一个模块使用64个通道、7×7卷积层
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

"""
    第二个模块：
        第一个卷积层：64个通道、1×1卷积层
        第二个卷积层：通道数增加为3倍的3×3卷积层
"""
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

"""
    第三个模块：串联2个Inception块
        第一个Inception块通道数：64+128+32+32=256
        第二个Inception块通道数：128+192+96+64=480
                   
        其中第一路径：64-->128
            第二路径：96-->128-->192 （96/192 = 1/2）
            第三路径：16-->32-->96 （16/192 = 1/12）
            第四路径：32-->64
        
"""
# Inception(in_channels, c1, c2, c3, c4)
# c1，c2，c3，c4是每条路径的输出通道数
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

"""
    第四个模块：串联5个Inception块 （第二、第三条路径按比例减少通道数）
        第一个Inception块通道数：192+208+48+64=512
        第二个Inception块通道数：160+224+64+64=512
        第三个Inception块通道数：128+256+64+64=512
        第四个Inception块通道数：112+288+64+64=528
        第五个Inception块通道数：256+320+128+128=832
                   
        其中第一路径：192-->160-->128-->112-->256
            第二路径：96-->208-->224-->256-->144-->288-->160-->320
            第三路径：16-->48-->24-->64-->24-->64-->32-->64-->32-->128
            第四路径：64-->64-->64-->64-->128
        
"""
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

"""
    第五个模块：串联2个Inception块 ,后紧跟输出层
        第一个Inception块通道数：256+320+128+128=832
        第二个Inception块通道数：384+384+128+128=1024
                   
        其中第一路径：256-->384
            第二路径：160-->320-->192-->384
            第三路径：32-->128-->48-->128
            第四路径：128-->128
        
"""
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   # 使用全局平均汇聚层，将每个通道的高度和宽度变成1
                   nn.AdaptiveAvgPool2d((1, 1)),
                   # 将输出变成二维数组
                   nn.Flatten())

# 连接一个输出个数为标签类别数的全连接层
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

# 为了更好的训练Fashion-MNIST，将输入的高度和宽度从224降到96
# 演示各个模块的输出形状变化
# size = (batch_size, in_channels, height, weight)
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# 定义精度评估函数，将数据集复制到显存中，通过调用accuracy计算数据集的精度
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    # 判断net是否属于torch.nn.Module类
    if isinstance(net, nn.Module):
        net.eval()
        
        # 如果不在参数选定的设备，将其传输到设备中
        if not device:
            device = next(iter(net.parameters())).device
    
    # Accumulator是累加器，定义两个变量：正确预测的数量，总预测的数量。
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            # 将X, y复制到设备中
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            
            # 计算正确预测的数量，总预测的数量，并存储到metric中
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 定义GPU训练函数，
#   1、为了使用gpu，首先需要将每一小批量数据移动到指定的设备（例如GPU）上；
#   2、使用Xavier随机初始化模型参数；
#   3、使用交叉熵损失函数和小批量随机梯度下降。
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    # 定义初始化参数，对线性层和卷积层生效
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    
    # 在设备device上进行训练
    print('training on', device)
    net.to(device)
    
    # 优化器：随机梯度下降
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    # 损失函数：交叉熵损失函数
    loss = nn.CrossEntropyLoss()
    
    # Animator为绘图函数
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    
    # 调用Timer函数统计时间
    timer, num_batches = d2l.Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        
        # Accumulator(3)定义3个变量：损失值，正确预测的数量，总预测的数量
        metric = d2l.Accumulator(3)
        net.train()
        
        # enumerate() 函数用于将一个可遍历的数据对象
        for i, (X, y) in enumerate(train_iter):
            timer.start() # 进行计时
            optimizer.zero_grad() # 梯度清零
            X, y = X.to(device), y.to(device) # 将特征和标签转移到device
            y_hat = net(X)
            l = loss(y_hat, y) # 交叉熵损失
            l.backward() # 进行梯度传递返回
            optimizer.step()
            with torch.no_grad():
                # 统计损失、预测正确数和样本数
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop() # 计时结束
            train_l = metric[0] / metric[2] # 计算损失
            train_acc = metric[1] / metric[2] # 计算精度
            
            # 进行绘图
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
                
        # 测试精度
        test_acc = evaluate_accuracy_gpu(net, test_iter) 
        animator.add(epoch + 1, (None, None, test_acc))
        
    # 输出损失值、训练精度、测试精度
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f},'
          f'test acc {test_acc:.3f}')
    
    # 设备的计算能力
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec'
          f'on {str(device)}')

# 训练，将图像转为96像素×96像素
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
d2l.plt.show()

