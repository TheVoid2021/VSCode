import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import os
from PIL import Image
import numpy as np

# 预处理：将两个步骤整合在一起
transform = transforms.Compose([
    transforms.ToTensor(), # 转为Tensor，范围改为0-1
    transforms.Normalize((0.1307,), (0.3081)) # 数据归一化，即均值为0，标准差为1
])

# 训练数据集
train_data = MNIST(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

# 测试数据集
test_data = MNIST(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_data, shuffle=False, batch_size=64)

# 模型
class CNNModel(nn.Module):
    # 定义一个CNN模型类，继承自nn.Module
    def __init__(self):
        # 初始化函数
        super(CNNModel, self).__init__()
        # 调用父类的初始化函数
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 定义第一个卷积层，输入通道数为1，输出通道数为32，卷积核大小为3，步长为1，填充为1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 定义第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3，步长为1，填充为1
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 定义第一个全连接层，输入特征数为64 * 7 * 7，输出特征数为128
        self.fc2 = nn.Linear(128, 10)
        # 定义第二个全连接层，输入特征数为128，输出特征数为10

    def forward(self, x):
        # 定义前向传播函数
        x = torch.relu(self.conv1(x))
        # 对第一个卷积层进行激活函数ReLU
        x = torch.max_pool2d(x, 2)
        # 对第一个卷积层进行最大池化操作，池化窗口大小为2
        x = torch.relu(self.conv2(x))
        # 对第二个卷积层进行激活函数ReLU
        x = torch.max_pool2d(x, 2)
        # 对第二个卷积层进行最大池化操作，池化窗口大小为2
        x = x.view(-1, 64 * 7 * 7)
        # 将卷积层输出展平为一维向量
        x = torch.relu(self.fc1(x))
        # 对第一个全连接层进行激活函数ReLU
        x = self.fc2(x)
        # 对第二个全连接层进行线性变换
        return x
        # 返回输出
# CrossEntropyLoss
model = CNNModel() # 实例化一个卷积神经网络模型
criterion = nn.CrossEntropyLoss() # 交叉熵损失，相当于Softmax+Log+NllLoss
optimizer = torch.optim.SGD(model.parameters(), 0.01) # 第一个参数是初始化参数值，第二个参数是学习率

# 模型训练
def train():
    for index, data in enumerate(train_loader):
        inputs, targets = data # input为输入数据，target为标签
        optimizer.zero_grad() # 梯度清零
        outputs = model(inputs) # 模型预测
        loss = criterion(outputs, targets) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        if index % 100 == 0: # 每一百次保存一次模型，打印损失
            torch.save(model.state_dict(), "./model/model copy.pkl") # 保存模型
            torch.save(optimizer.state_dict(), "./model/optimizer copy.pkl")
            print("损失值为：%.2f" % loss.item())

# 加载模型
if os.path.exists('./model/model copy.pkl'):
    model.load_state_dict(torch.load("./model/model copy.pkl")) # 加载保存模型的参数

# 模型测试
def test():
    correct = 0 # 正确预测的个数
    total = 0 # 总数
    with torch.no_grad(): # 测试不用计算梯度
        for data in test_loader:
            inputs, targets = data
            outputs = model(inputs) # output输出10个预测取值，其中最大的即为预测的数
            _, predict = torch.max(outputs.data, dim=1) # 返回一个元组，第一个为最大概率值，第二个为最大值的下标
            total += targets.size(0) # target是形状为(batch_size,1)的矩阵，使用size(0)取出该批的大小
            correct += (predict == targets).sum().item() # predict和target均为(batch_size,1)的矩阵，sum()求出相等的个数
        print("准确率为：%.2f" % (correct / total))

# 自定义手写数字识别测试
def test_mydata():
    image = Image.open('./test/test_three.png') # 读取自定义手写图片
    image = image.resize((28, 28)) # 裁剪尺寸为28*28
    image = image.convert('L') # 转换为灰度图像
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.unsqueeze(0) # 添加一个维度，使其形状为(1, 1, 28, 28)
    output = model(image)
    _, predict = torch.max(output.data, dim=1)
    print("此手写图片值为:%d,其最大概率为:%.2f" % (predict[0], output[0, predict[0]].item()))
    plt.title('此手写图片值为：{}'.format((int(predict[0]))), fontname="SimHei")
    plt.imshow(image.squeeze(), cmap='gray')
    plt.show()

# 主函数
if __name__ == '__main__':
    # 自定义测试
    test_mydata()
    # 训练与测试
    # for i in range(5): # 训练和测试进行五轮
    #     train()
    #     test()
