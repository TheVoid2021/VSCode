import torch
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

# 数据增强 (定义了训练和验证数据集的数据增强方法，包括随机水平翻转、随机灰度化、随机仿射变换和归一化。)
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),  
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomAffine(5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomAffine(5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# 加载数据集 (加载FashionMNIST数据集，并将其分为训练集和测试集，使用DataLoader进行批量加载。)
# 加载训练数据集
training_data = datasets.FashionMNIST(
    root='data',  # 数据集的根目录
    train=True,  # 是否为训练集
    download=True,  # 是否下载数据集
    transform=transforms.ToTensor()  # 数据预处理
)
# 创建训练数据加载器
training_loader = torch.utils.data.DataLoader(training_data, batch_size=16, shuffle=True)  # 每个批次16个样本，打乱顺序

# 加载测试数据集
test_data = datasets.FashionMNIST(
    root='data',  # 数据集的根目录
    train=False,  # 是否为训练集
    download=True,  # 是否下载数据集
    transform=transforms.ToTensor()  # 数据预处理
)
# 创建测试数据加载器
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)  # 每个批次16个样本，不打乱顺序

# 查看图片 (获取一个训练样本，用于后续的图片查看或调试。)
img, label = next(iter(training_loader))
# print("img:", img)
# print(f"img.size:{img.size()}")
# print("label:", label)
# print(f"lebel.size:", label.size())

# 使用GPU还是CPU
device = 'cuda' if torch.cuda.is_available() else "CPU"
# print(f"using: {device}")

# 定义卷积神经网络 (两个卷积层、两个全连接层和一个池化层。)
class CNN(nn.Module):
    # 定义一个CNN类，继承自nn.Module
    def __init__(self):
        # 初始化函数
        super(CNN, self).__init__()
        # 调用父类的初始化函数
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  
        # 定义第一个卷积层，输入通道数为1，输出通道数为32，卷积核大小为3，步长为1，填充为1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
        # 定义第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3，步长为1，填充为1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 定义池化层，池化核大小为2，步长为2，填充为0
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 定义第一个全连接层，输入维度为64 * 7 * 7，输出维度为128
        self.fc2 = nn.Linear(128, 10)
        # 定义第二个全连接层，输入维度为128，输出维度为10
        
    def forward(self, x):
        # 定义前向传播函数
        x = self.pool(F.relu(self.conv1(x)))
        # 对第一个卷积层进行激活函数和池化操作
        x = self.pool(F.relu(self.conv2(x)))
        # 对第二个卷积层进行激活函数和池化操作
        x = x.view(-1, 64 * 7 * 7)
        # 将卷积层的输出展平为一维向量
        x = F.relu(self.fc1(x))
        # 对第一个全连接层进行激活函数操作
        x = self.fc2(x)
        # 对第二个全连接层进行操作
        return x
        # 返回输出


# 实例化模型 (实例化模型并将其移动到指定的设备（GPU或CPU）。)
my_model = CNN().to(device)

#使用交叉熵损失函数和随机梯度下降（SGD）优化器。
# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化方法
optimizer = optim.SGD(my_model.parameters(), lr=0.005, momentum=0.1)

# 训练神经网络 (进行模型训练，包括前向传播、计算损失、反向传播和参数更新。同时计算测试集的损失和准确率。)
# 设置训练次数
epochs = 20
# 初始化训练误差和测试误差列表
training_losses = []
test_losses = []

# 获取开始时间
e1 = cv2.getTickCount()
# 遍历训练次数
for epoch in range(epochs):
    # 初始化总误差
    total_loss = 0.0
    # 遍历训练集
    for images, labels in training_loader:
        # 将图像和标签移动到指定设备
        images, labels = images.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = my_model(images)
        # 计算损失
        loss = criterion(output, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累加损失
        total_loss += loss.item()
    # 测试集测试
    else:
        # 初始化测试误差和准确率
        test_loss = 0
        accuracy = 0
        # 不计算梯度
        with torch.no_grad():
            # 设置为评估模式
            my_model.eval()
            # 遍历测试集
            for images, labels in test_loader:
                # 将图像和标签移动到指定设备
                images, labels = images.to(device), labels.to(device)
                # 前向传播
                output = my_model(images)
                # 累加损失
                test_loss += criterion(output, labels)
                # 计算预测结果
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                # 计算准确率
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        # 设置为训练模式
        my_model.train()
        # 将训练误差和测试误差添加到列表中
        training_losses.append(total_loss / len(training_loader))
        test_losses.append(test_loss / len(test_loader))
        
        # 打印训练误差、测试误差和准确率
        print("训练集训练次数:{}/{}:".format((epoch + 1), epochs),
              "训练误差:{:.3f}".format(total_loss / len(training_loader)),
              "测试误差:{:.3f}".format(test_loss / len(test_loader)),
              "模型分类准确率:{:.3f}".format(accuracy / len(test_loader)))

# 可视化训练误差和测试误差
training_losses = np.array(torch.tensor(training_losses, device='cpu'))
test_losses = np.array(torch.tensor(test_losses, device='cpu'))
plt.plot(training_losses, label="training_losses")
plt.plot(test_losses, label="test_losses")
plt.legend()
plt.show()

# 保存模型 (保存训练好的模型参数。)
torch.save(my_model.state_dict(), './model/FashionMnist_weight_cnn.pth')


# 加载模型并测试
model = CNN() # 实例化一个神经网络模型,需要和权重文件的网络模型一样
model.load_state_dict(torch.load("./model/FashionMnist_weight_cnn.pth"))    # 加载预训练模型
model.eval()    # 预测结果前必须要做的步骤，其作用为将模型转为evaluation模式

# 获取测试集数据
images, labels = next(iter(test_loader))

# 获取第 n 张图片
img = images[8]  # 获取第 n 张图片
img = img.reshape((1, 1, 28, 28))  # 将图片转换为模型期望的形状

# 测试
with torch.no_grad():  # 关闭梯度计算
    output = model.forward(img)  # 前向传播
    ps = torch.exp(output)  # 将输出转换为概率
    # 返回矩阵每一行最大值和下标,元组类型
    top_p, top_class = ps.topk(1, dim=1)  # topk函数：返回矩阵每一行最大的k个值和下标
    labellist = ['T恤', '裤子', '套衫', '裙子', '外套', '凉鞋', '汗衫', '运动鞋', '包包', '靴子'] 
    prediction = labellist[top_class.item()]  # 根据下标找到对应的标签 （top_class.item()返回的是下标）
    probability = float(top_p.squeeze().item()) / 100  # 将tensor转换为float类型并除以100 （因为模型输出的概率是0-1之间的数，所以需要除以100）
    print(f'神经网络猜测图片里是 {prediction}，概率为{probability}%')  # 打印预测结果
