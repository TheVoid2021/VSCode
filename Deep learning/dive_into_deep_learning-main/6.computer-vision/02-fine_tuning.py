import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

# 下载并解压数据集
data_dir = d2l.download_extract('hotdog')

# 加载训练集和测试集
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# 显示训练集中的8张热狗图片和8张非热狗图片
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);

# 定义数据增广方式
# 使用RGB通道的均值和标准差，以标准化每个通道
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# 定义训练集的数据增广方式
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

# 定义测试集的数据增广方式
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

# 加载预训练的ResNet-18模型
pretrained_net = torchvision.models.resnet18(pretrained=True)

"""
可以更改 PyTorch 下载预训练模型和权重文件的缓存路径。PyTorch 使用 torch.hub 模块来管理这些缓存文件，
你可以通过设置 TORCH_HOME 环境变量来更改缓存路径。
在 Windows 系统中，你可以通过以下方式设置环境变量：

      打开“控制面板”。
      选择“系统和安全”。
      选择“系统”。
      在左侧菜单中选择“高级系统设置”。
      在“系统属性”窗口中，点击“环境变量”按钮。
      在“环境变量”窗口中，点击“新建”按钮，添加一个新的环境变量 TORCH_HOME，并设置其值为你想要的缓存路径，例如 C:\my_torch_cache。
      点击“确定”保存更改。
"""

# 打印预训练模型的输出层
pretrained_net.fc

# 定义微调的ResNet-18模型
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);

# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    # 加载训练集
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    # 加载测试集
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    # 获取可用的GPU设备
    devices = d2l.try_all_gpus()
    # 定义损失函数
    loss = nn.CrossEntropyLoss(reduction="none")
    # 定义优化器
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    # 训练模型
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
    
# 微调预训练模型
train_fine_tuning(finetune_net, 5e-5)

# 设置一个从头开始训练模型作为对照组 所有的权重全都随机初始化，不使用预训练模型 pretrain=False
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)

d2l.plt.show()