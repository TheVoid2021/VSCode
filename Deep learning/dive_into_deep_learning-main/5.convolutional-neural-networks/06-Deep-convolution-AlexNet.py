import torch
from torch import nn
from d2l import torch as d2l

# 控制容量和预处理
net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))

# 构造一个高度和宽度都为224的单通道数据，来观察每一层输出的形状
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)


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

# 这里使用的是Fashion-MNIST数据集以节省时间
# Fashion-MNIST图像的分辨率（28×28像素）低于ImageNet图像，将它们增加到224×224
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# 训练AlexNet：使用更小的学习速率训练，这是因为网络更深更广、图像分辨率更高
lr, num_epochs = 0.01, 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
d2l.plt.show()
