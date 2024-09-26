import torch   #PyTorch 是一个用于深度学习的开源框架，主要用于构建和训练神经网络模型。
    # 张量（Tensor）：PyTorch 的核心是张量，它是一个多维数组，类似于 NumPy 的 ndarray，但提供了 GPU 加速和自动求导功能。
    # 自动求导：PyTorch 提供了自动求导机制，可以自动计算张量的梯度，这对于训练神经网络非常有用。
    # 神经网络模块：PyTorch 提供了丰富的神经网络模块，包括卷积神经网络（CNN）、循环神经网络（RNN）、长短期记忆网络（LSTM）等。
    # 优化器：PyTorch 提供了多种优化器，如随机梯度下降（SGD）、Adam、RMSprop 等，用于训练神经网络。
    # 数据加载器：PyTorch 提供了 DataLoader，用于加载和处理数据集，支持批量处理、数据增强等功能。
    # PyTorch 还提供了许多其他功能，如模型保存和加载、可视化工具、分布式训练等，方便进行深度学习研究和开发。
import cv2 #conda install -c conda-forge opencv
    # OpenCV（Open Source Computer Vision Library）是一个开源的计算机视觉和机器学习软件库。它拥有超过2500个优化算法，
    # 包括经典和最新的计算机视觉以及机器学习技术。它被广泛用于面部识别、对象识别、图像分割、动作跟踪、生成3D模型等任务。
import torch.optim as optim  #用于定义优化器
    # torch.optim 是 PyTorch 中的一个模块，包含了多种优化算法，用于调整模型的参数以最小化损失函数。
    # 常用的优化器包括 SGD（随机梯度下降）、Adam、RMSprop 等。
    # 这些优化器通过更新模型的权重来加速和改进训练过程。
    # 例如：optimizer = optim.Adam(model.parameters(), lr=0.001) 定义了一个 Adam 优化器，并设置学习率为 0.001。
import numpy as np  #Numpy 是一个用于科学计算的 Python 库，主要用于处理大型多维数组和矩阵，它提供了高性能的多维数组对象和一系列用于处理这些数组的工具。
    #多维数组对象（ndarray）：NumPy 的核心是多维数组对象（ndarray），它提供了高效的数组操作和计算能力。
    #广播机制：NumPy 支持广播（broadcasting），允许不同形状的数组进行算术运算，这在处理多维数据时非常有用。
    #数学函数库：NumPy 提供了大量的数学函数，包括线性代数、傅里叶变换、随机数生成等。
    #数据读写：NumPy 支持从文件中读取和写入数组数据，支持多种文件格式（如 CSV、文本文件、二进制文件等）。
    #线性代数、随机数生成：NumPy 提供了丰富的线性代数函数和随机数生成函数，方便进行科学计算。
import torchvision  #torchvision 是 PyTorch 的一个库，专门用于处理计算机视觉任务。它包含了一些常用的数据集、模型和工具，
    # 使得在 PyTorch 中进行图像处理和计算机视觉任务变得更加简单和高效。
    # 常用的数据集、模型和工具：torchvision 提供了一些常用的数据集（如 CIFAR、ImageNet、COCO 等），
    # 以及一些预训练的模型（如 ResNet、VGG、AlexNet 等）。
    # 这些数据集和模型可以用于训练和评估计算机视觉任务。
from torchvision import datasets,transforms  #torchvision.datasets 和 torchvision.transforms 是 torchvision 库中的两个模块，
    # 用于处理和增强图像数据集。
import torch.nn as nn   #用于定义神经网络
    # torch.nn 是 PyTorch 中的一个模块，提供了构建神经网络的基础组件。
    # 该模块包含了各种神经网络层（如全连接层、卷积层、循环层等）、激活函数（如ReLU、Sigmoid等）和损失函数（如交叉熵损失、均方误差等）。
    # 通过组合这些组件，可以方便地构建和训练复杂的神经网络模型。
    # 例如：class Neuralnetwork(nn.Module): 定义了一个神经网络类，继承自 nn.Module。
import torch.nn.functional as F  #torch.nn.functional 是 PyTorch 中的一个模块，提供了一些常用的神经网络层和激活函数。
    # 这些函数可以直接在神经网络模型中使用，方便构建和训练复杂的神经网络。
    # 例如：F.relu(x) 使用 ReLU 激活函数对输入张量 x 进行激活。
import matplotlib.pyplot as plt  #conda install -c conda-forge matplotlib
    # Matplotlib 是一个用于绘制图表和可视化数据的 Python库。它提供了丰富的绘图工具，
    # 可以用于生成各种静态、交互式和动画图表。Matplotlib 是数据科学、机器学习和科学计算领域中最流行的绘图库之一。
    # 例如：plt.plot(x, y) 绘制一条折线图，plt.show() 显示图形。

# 使用Composes将transforms组合在一起，tranforms子类进行数据预处理和数据增强  定义了一个数据增强（Data Augmentation）的流程，用于在训练和验证阶段对图像数据进行预处理。
data_transforms = {
    'train':transforms.Compose([transforms.ToTensor(),  # 图像转换成pytorch中的张量（0-255）
                                #功能：将图像转换成PyTorch张量，并将图像的像素值从[0, 255]范围缩放到[0, 1]范围。
                                #用途：将图像数据转换为PyTorch可以处理的格式，以便后续的模型训练和推理。
                                transforms.RandomHorizontalFlip(p=0.5),  # 图像依概率随机翻转
                                #功能：以50%的概率随机水平翻转图像。
                                #用途：增加数据多样性，防止模型过拟合。
                                transforms.RandomGrayscale(p=0.2),   # 图像依概率随机灰度化
                                #功能：以20%的概率随机将图像转换为灰度图像。
                                #用途：增加数据多样性，防止模型过拟合。
                                transforms.RandomAffine(5),  # 图像中心保持不变的随机仿射变换
                                #功能：对图像进行随机仿射变换，包括旋转、平移、缩放和错切等，但中心点保持不变。
                                #用途：增加数据多样性，防止模型过拟合。
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), # 归一化
                                #功能：将图像的每个通道的像素值归一化到[-1, 1]范围，将每个通道的像素值减去均值（0.5），然后除以标准差（0.5）。
                                #用途：将图像数据标准化，使得模型训练更加稳定和高效。
                                ]),
    'val':transforms.Compose([transforms.ToTensor(),  # 图像转换成pytorch中的张量 （0-255）
                                transforms.RandomHorizontalFlip(p=0.5),  # 图像依概率随机翻转
                                transforms.RandomGrayscale(p=0.2),   # 图像依概率随机灰度化
                                transforms.RandomAffine(5),  # 图像中心保持不变的随机仿射变换
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), # 归一化
                                ])
    #注意事项：
    # 在训练阶段（'train'），数据增强操作是随机进行的，以增加模型的泛化能力。
    # 在验证阶段（'val'），数据增强操作被禁用，以确保验证集的图像数据与训练集保持一致，以便准确评估模型的性能。
    # 归一化参数（均值和标准差）通常是根据训练集计算得到的，这里假设使用的是(0.5, 0.5, 0.5)和(0.5, 0.5, 0.5)。
    # 上面这段代码的目的是通过数据增强技术来提高模型的泛化能力和鲁棒性，同时确保验证阶段的图像数据与训练阶段保持一致。
}

# 张量（Tensor）是一个多维数组，是深度学习和机器学习中的基本数据结构。
# 张量可以看作是一个通用的矩阵，可以有任意数量的维度。
# 张量可以存储在CPU或GPU上，用于计算和存储数据。
# 张量是PyTorch中的基本数据结构，用于表示和操作数据。

# 加载训练集 
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,  # 若本地没有数据集，下载 （下载后会保存在root路径下）
    transform=transforms.ToTensor()
) 
# 构建训练集数据载入器 并提供给定数据集的可迭代对象。
training_loader = torch.utils.data.DataLoader(training_data,batch_size=16,shuffle=True)
# 加载测试集
test_data = datasets.FashionMNIST(    
    root='data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
# 构建测试集数据载入器 
test_loader = torch.utils.data.DataLoader(test_data,batch_size=16,shuffle=False) 

# 定义神经网络模型
class myModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)  # 输入层到隐藏层 (784 是输入图片的像素点数,256 是隐藏层的神经元数)
        self.fc2 = nn.Linear(256, 128)  # 隐藏层到隐藏层 (256 是隐藏层的神经元数,128 是隐藏层的神经元数)
        self.fc3 = nn.Linear(128, 64)  # 隐藏层到隐藏层 (128 是隐藏层的神经元数,64 是隐藏层的神经元数)
        self.fc4 = nn.Linear(64, 10)  # 隐藏层到输出层 (64 是隐藏层的神经元数,10 是输出层的神经元数)
        
    def forward(self, x):   # 前向传播
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)   #手动展平
        
        x = F.relu(self.fc1(x))  # 激活函数 (ReLU 函数)
        x = F.relu(self.fc2(x))  # 激活函数 (ReLU 函数)
        x = F.relu(self.fc3(x))  # 激活函数 (ReLU 函数)
        x = F.log_softmax(self.fc4(x), dim=1)  # 输出结果 (dim=1 表示在最后一个维度上进行归一化)
        
        return x
#通过四个全连接层进行前向传播，并在每个全连接层之后应用 ReLU 激活函数。最后，在输出层应用 log_softmax 函数，对输出进行归一化，使得输出可以被视为概率分布。
# 注意事项
# 输入形状：输入张量 x 应该是形状为 (batch_size, 1, 28, 28) 的四维张量，表示一批28x28的灰度图像。
# 激活函数：ReLU 激活函数用于引入非线性，有助于模型学习复杂的模式。
# 归一化：log_softmax 函数用于将输出转换为概率分布，适用于多分类问题。
# 损失函数：通常与 log_softmax 配合使用的损失函数是 nn.NLLLoss，因为 log_softmax 的输出是 log-probabilities。
# 展平操作: 手动使用 x.view 将输入展平。
# 层定义: 单独定义每一层。
# 激活函数: 使用 ReLU 激活函数。
# 输出: 对输出应用 log_softmax。

# 实例化模型
device = 'cuda' if torch.cuda.is_available() else "CPU"
my_model = myModule().to(device)


# 定义损失函数和优化器
# 定义损失函数
criterion = nn.CrossEntropyLoss()   # 多分类,使用交叉熵损失函数 (交叉熵损失函数：用于多分类问题)
# 交叉熵（Cross-Entropy）是一个用于衡量两个概率分布之间差异的损失函数，常用于分类问题。
# 在多分类问题中，交叉熵损失函数可以衡量模型预测的概率分布与真实分布之间的差异。它的值越小，表示模型的预测越准确。

# 定义优化方法 SGD 随机梯度下降
optimizer = optim.SGD(my_model.parameters(),lr=0.005,momentum=0.1) 
# SGD 是 Stochastic Gradient Descent（随机梯度下降）的缩写，是一种用于训练机器学习模型（尤其是神经网络）的优化算法。SGD 是梯度下降算法的一种变体，它通过随机选择样本进行梯度计算，从而加快训练速度并减少计算资源的需求。
# SGD 的基本原理
# 梯度下降：梯度下降是一种优化算法，通过迭代地调整参数，使得目标函数（通常是损失函数）的值逐渐减小。每次迭代，算法会计算目标函数在当前参数值下的梯度，并沿着梯度的反方向更新参数。
# 随机梯度下降：在传统的梯度下降中，每次迭代都会计算整个数据集的梯度。而在随机梯度下降中，每次迭代只计算一个样本的梯度。这种方法可以加快训练速度，但可能会引入噪声，导致目标函数的收敛不稳定。
# SGD 的优点
# 训练速度快：由于每次迭代只计算一个样本的梯度，SGD 的训练速度通常比传统的梯度下降要快得多，尤其是在大数据集上。
# 内存消耗小：由于每次迭代只处理一个样本，SGD 的内存消耗通常比传统的梯度下降要小得多。
# SGD 的缺点
# 收敛不稳定：由于每次迭代只计算一个样本的梯度，SGD 的收敛过程可能会比较不稳定，有时会出现震荡现象。
# 需要调整学习率：SGD 的收敛速度和稳定性很大程度上取决于学习率的选择。如果学习率过大，可能会导致训练过程发散；如果学习率过小，可能会导致训练过程收敛过慢。

# 训练神经网络
epochs = 20   # 训练次数
# 训练误差和测试误差存储在这里,最后进行可视化
training_losses = []  # 训练误差
test_losses = []  # 测试误差
# print("training start...")
# 记录训练时间
e1 = cv2.getTickCount()  # 获取当前时间

for epoch in range(epochs): 
    total_loss = 0.0  # 训练误差
    # 遍历训练集中所有数据
    for images,labels in training_loader:
        images,labels = images.to(device),labels.to(device)  # 将数据送入GPU
        optimizer.zero_grad()   # 将优化器中的所有求导结果设置为0 （在反向传播之前，需要将优化器中的所有求导结果设置为0）
        output = my_model(images)   # 神经网络的输出 (前向传播)
        loss = criterion(output,labels) # 损失函数 (计算预测值和真实值之间的误差)
        loss.backward() # 后向传播 (计算损失函数对神经网络参数的梯度)
        optimizer.step()    # 更新参数 (根据梯度更新参数)
        total_loss += loss.item()   # 损失函数求和 
    else:
        # 测试数据集
        test_loss = 0
        accuracy = 0
        # 测试的时候不需要自动求导和反向传播
        with torch.no_grad():
            # 关闭Dorpout
            my_model.eval() # 预测结果前必须要做的步骤，其作用为将模型转为evaluation模式
            # 遍历测试集
            for images,labels in test_loader:
                # 对传入的图片进行正向推断
                # 将数据也送入GPU
                images,labels = images.to(device),labels.to(device)  # 将数据送入GPU
                output = my_model(images)  # 神经网络的输出 (前向传播)
                test_loss += criterion(output,labels)  # 损失函数 (计算预测值和真实值之间的误差)
                ps = torch.exp(output)  # 计算softmax       
                top_p,top_class = ps.topk(1,dim=1)  # 取top1 (dim=1 表示在第1维上取最大值)
                equals = top_class == labels.view(*top_class.shape)  # 比较预测值和真实值
                accuracy += torch.mean(equals.type(torch.FloatTensor))  # 计算准确率
        # 恢复Dropout
        my_model.train()
        # 将训练误差和测试误差存储在列表中
        training_losses.append(total_loss/len(training_loader))  # 训练误差
        test_losses.append(test_loss/len(test_loader))  # 测试误差
        
        print("训练集训练次数:{}/{}:".format((epoch+1),epochs),
              "训练误差:{:.3f}".format(total_loss/len(training_loader)),
              "测试误差:{:.3f}".format(test_loss/len(test_loader)),
              "模型分类准确率:{:.3f}".format(accuracy/len(test_loader)))

# 可视化训练误差和测试误差      
# 将训练误差和测试误差数据从GPU转回CPU 并且将tensor->numpy (因为numpy 是cup only 的数据类型)
training_losses = np.array(torch.tensor(training_losses,device='cpu'))  # 将训练误差数据从GPU转回CPU 并且将tensor->numpy
test_losses = np.array(torch.tensor(test_losses,device='cpu'))  # 将测试误差数据从GPU转回CPU 并且将tensor->numpy
# 可视化
plt.plot(training_losses,label="training_losses")  #plot函数：绘制训练误差
plt.plot(test_losses,label="test_losses")  #plot函数：绘制测试误差
plt.legend()  #legend函数：显示图例
plt.show()  #show函数：显示图像

# 模型保存
torch.save(my_model.state_dict(),"./model/FashionMnist_weight_fcn.pth")  # 保存模型参数

# 加载模型并测试
model = myModule()
model.load_state_dict(torch.load("./model/FashionMnist_weight_fcn.pth"))
model.eval()

images, labels = next(iter(test_loader))
img = images[0].reshape((28, 28)).numpy()
img = torch.from_numpy(img).view(1, 784)

with torch.no_grad():
    output = model.forward(img)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(1, dim=1)
    labellist = ['T恤', '裤子', '套衫', '裙子', '外套', '凉鞋', '汗衫', '运动鞋', '包包', '靴子']
    prediction = labellist[top_class]
    probability = float(top_p)
    print(f'神经网络猜测图片里是 {prediction}，概率为{probability*100}%')


# 主函数
if __name__ == '__main__':
    torch.no_grad()