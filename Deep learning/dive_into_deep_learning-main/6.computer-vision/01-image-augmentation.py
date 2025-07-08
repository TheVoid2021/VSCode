"""
  图像增广（Image Augmentation）
  图像增广在对训练图像进行一系列的随机变化之后，生成相似但不同的训练样本，从而扩大了训练集的规模。 
  此外，应用图像增广的原因是，随机改变训练样本可以减少模型对某些属性的依赖，从而提高模型的泛化能力。
  在实践中，我们使用图像增广训练模型，以减少对训练图像的过拟合。
  在预测时，我们不使用随机图像增广，以确保输出结果是确定的，从而对测试结果进行评估。
  在下面，我们将图像增广应用到了训练数据集和测试数据集。
  如我们所见，在训练时，图像增广会随机改变图像的形状和色彩。
  这在训练中引入随机性，从而缓解模型的过拟合。
  在测试时，我们不使用随机图像增广，从而确保测试结果是可以重现的。
"""
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 使用一个图像来演示图像增广
d2l.set_figsize()
img = d2l.Image.open('E:/desktop/vsCode/Deep learning/dive_into_deep_learning-main/6.computer-vision/img/cat1.jpg')
d2l.plt.imshow(img);

# 本函数已保存在d2lzh_pytorch包中方便以后使用
# 定义一个函数，用于显示图像
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

# 定义辅助函数apply，此函数在输入图像img上多次运行图像增广方法aug并显示所有结果。
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

# 随机水平翻转图像
apply(img, torchvision.transforms.RandomHorizontalFlip())

# 随机垂直翻转图像
apply(img, torchvision.transforms.RandomVerticalFlip())

# 随机裁剪一个面积为原始面积10%到100%的区域，该区域的宽高比从0.5～2之间随机取值。 然后，区域的宽度和高度都被缩放到200像素。
shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 改变图像颜色的四个方面：亮度、对比度、饱和度和色调。
# 随机更改图像的亮度，随机值为原始图像的50%到150%之间。
apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))

# 随机更改图像的对比度，随机值为原始图像的50%到150%之间。
apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))


# 定义颜色增强变换，包括亮度、对比度、饱和度和色相的调整
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# 应用颜色增强变换到图像上
apply(img, color_aug)


# 定义一个数据增强的列表，包括随机水平翻转、颜色增强和形状增强 Compose实例
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
# 对图像应用数据增强
apply(img, augs)


# 加载CIFAR10数据集，train参数为True表示加载训练集，root参数指定数据集的根目录，download参数为True表示如果数据集不存在则自动下载
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",download=True)
# 显示前32张图片，4行8列，scale参数指定图片的缩放比例
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);


# 定义训练数据增强方法
train_augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.ToTensor()])

# 定义测试数据增强方法
test_augs = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 加载CIFAR10数据集
def load_cifar10(is_train, augs, batch_size):
    # 加载CIFAR10数据集
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    # 返回数据加载器
    return dataloader

#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """用多GPU进行小批量训练"""
    if isinstance(X, list):
        # 微调BERT中所需
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """用多GPU进行模型训练"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
    
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

# 定义一个函数，用于初始化网络权重
def init_weights(m):
    # 如果当前层是线性层或卷积层，则使用Xavier均匀分布初始化权重
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

# 对网络中的每一层应用初始化函数
net.apply(init_weights)

# 定义一个函数，用于使用数据增强进行训练
def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    # 加载训练集，使用训练集的数据增强方式
    train_iter = load_cifar10(True, train_augs, batch_size)
    # 加载测试集，使用测试集的数据增强方式
    test_iter = load_cifar10(False, test_augs, batch_size)
    # 定义损失函数，使用交叉熵损失函数，不进行归一化
    loss = nn.CrossEntropyLoss(reduction="none")
    # 定义优化器，使用Adam优化器，学习率为0.001
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    # 调用train_ch13函数进行训练
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

# train_augs是指采取了数据增广方式，test_augs是指没有采取数据增广方式
# 调用train_with_data_aug函数进行训练，这里train_augs是采取了数据增广的训练集，而test_augs没有采取数据增广方式
# train_with_data_aug(train_augs, test_augs, net)
# 如果想要测试没有数据增广的模型，可以直接使用test_augs作为训练集和测试集，因为test_augs没有进行数据增广方式
train_with_data_aug(test_augs, test_augs, net)
# 从结果上来说，模型在没有使用数据增广的训练集上的准确率更高，而在测试集上的准确率更低，这是因为数据增广使得模型在训练集上的准确率更高，
d2l.plt.show()