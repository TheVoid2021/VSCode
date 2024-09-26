import torch
from torch import nn
from d2l import torch as d2l


# 定义vgg块，（卷积层数，输入通道，输出通道）
def vgg_block(num_convs, in_channels, out_channels):
    # 创建空网络结果，之后通过循环操作使用append函数进行添加
    layers = []
    
    # 循环操作，添加卷积层和非线性激活层
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
        
    # 最后添加最大值汇聚层
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# 原VGG网络有5个卷积块，前两个有一个卷积层，后三个块有两个卷积层
# 该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11

# (卷积层数，输出通道数)
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

# 通过for循环实现VGG-11
def vgg(conv_arch):
    # 定义空网络结构
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        # 添加vgg块
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        # 下一层输入通道数=当前层输出通道数
        in_channels = out_channels
        
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)

# 构建一个高度和宽度为224的单通道数据样本，以观察每个层输出的形状
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

# 构建了一个通道数较少的网络，足够用于训练Fashion-MNIST数据集
ratio = 4
# //为整除
small_conv_arch = [(pair[0], pair[1] // 4) for pair in conv_arch]
net = vgg(small_conv_arch)

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

# 与上一节AlexNet相同，但使用VGG-11模型
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
d2l.plt.show()