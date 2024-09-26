import os  #用于文件和目录操作
import argparse  #用于解析命令行参数
# argparse 是 Python 的标准库模块之一，用于解析命令行参数和选项。
# 它可以帮助编写用户友好的命令行接口，使得程序可以从命令行获取输入参数。
# 通过定义参数和选项，argparse 会自动生成帮助和使用信息，并解析传递给程序的参数。
# 例如：
# parser = argparse.ArgumentParser(description='Example with argparse')
# parser.add_argument('--num_classes', type=int, default=100, help='the number of classes')
# args = parser.parse_args()
# 这段代码定义了一个命令行参数解析器，并添加了一个名为 --num_classes 的参数。
import sys  #用于与系统交互
# sys 是 Python 的标准库模块之一，提供了与 Python 解释器和系统进行交互的功能。
# 主要功能包括访问命令行参数、标准输入输出、错误输出、解释器退出等。
# 例如：sys.argv 获取命令行参数列表。
# 例如：sys.exit() 退出程序。
# 例如：sys.stdout.write('Hello\n') 向标准输出写入内容。
from torch.utils.data import DataLoader  #DataLoader 是 PyTorch 中用于加载数据集的重要工具。
#它提供了一个可迭代对象，可以对给定的数据集进行自动批处理、采样、打乱和多进程数据加载。
#这使得处理大型数据集并在训练和评估过程中将数据集输入到神经网络中变得更加容易。
from tqdm import tqdm  #用于显示进度条，方便在终端中实时查看训练或处理进度
# tqdm 是一个用于显示进度条的库，主要用于在长时间运行的循环中显示进度。
# 它可以在终端、Jupyter Notebook 和 GUI 应用程序中使用。
# 通过简单地将 tqdm 包装在任何可迭代对象上，可以轻松地添加进度条。
# 例如：for i in tqdm(range(100)): 执行一个循环并显示进度条。
import torch #PyTorch的核心库，用于张量计算和自动求导。
import torch.optim as optim  #用于定义优化器
# torch.optim 是 PyTorch 中的一个模块，包含了多种优化算法，用于调整模型的参数以最小化损失函数。
# 常用的优化器包括 SGD（随机梯度下降）、Adam、RMSprop 等。
# 这些优化器通过更新模型的权重来加速和改进训练过程。
# 例如：optimizer = optim.Adam(model.parameters(), lr=0.001) 定义了一个 Adam 优化器，并设置学习率为 0.001。
import torch.nn as nn  #用于定义神经网络
# torch.nn 是 PyTorch 中的一个模块，提供了构建神经网络的基础组件。
# 该模块包含了各种神经网络层（如全连接层、卷积层、循环层等）、激活函数（如ReLU、Sigmoid等）和损失函数（如交叉熵损失、均方误差等）。
# 通过组合这些组件，可以方便地构建和训练复杂的神经网络模型。
# 例如：class Neuralnetwork(nn.Module): 定义了一个神经网络类，继承自 nn.Module。
from data_loader import iris_dataload  #自定义的数据加载模块，用于加载Iris数据集。

#这段代码是用于解析命令行参数，定义类别数、训练的epoch数、训练的batch大小、学习率、数据路径和设备。
parser = argparse.ArgumentParser()  #用于解析命令行参数
parser.add_argument('--num_classes', type=int, default=100, help='the number of classes')  #用于定义类别数
parser.add_argument('--epochs', type=int, default=20, help='the number of training epoch')  #用于定义训练的epoch数（训练次数）
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')  #用于定义训练的batch大小
parser.add_argument('--lr', type=float, default=0.005, help='star learning rate')   #用于定义学习率 
parser.add_argument('--data_path', type=str, default="e:/desktop/vsCode/Python/pytorch2.0_nn/Iris_data.txt")  #用于定义数据路径
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')  #用于定义设备
opt = parser.parse_args()  

# 初始化神经网络模型
class Neuralnetwork(nn.Module):   #初始化定义网络层级结构
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim): 
        super(Neuralnetwork, self).__init__()   
        self.layer1 = nn.Linear(in_dim, n_hidden_1) #全连接层1 （输入张量与权重矩阵相乘并加上偏置向量，然后输出结果）
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2) #全连接层2
        self.layer3 = nn.Linear(n_hidden_2, out_dim) #全连接层3

#张量（Tensor）是一个多维数组，是深度学习和机器学习中的基本数据结构。
#张量可以看作是一个通用的矩阵，可以有任意数量的维度。
#张量可以存储在CPU或GPU上，用于计算和存储数据。
#张量是PyTorch中的基本数据结构，用于表示和操作数据。

    def forward(self, x):   #定义网络功能（当数据输入进来后，要对数据进行什么样的处理）：前向传播
        x = self.layer1(x)  #全连接层1矩阵运算激活
        x = self.layer2(x)  #全连接层2矩阵运算激活
        x = self.layer3(x)  #全连接层3矩阵运算激活
        return x
# self 是类实例的引用，用于在类的方法中访问和操作实例的属性和方法。
# 每个实例方法的第一个参数必须是 self，以便方法能够访问实例的状态。

# 定义当前模型的训练环境
device = torch.device(opt.device if torch.cuda.is_available() else "cpu") 

# 划分数据集并加载 （训练集、验证集和测试集）
custom_dataset = iris_dataload( "./Iris_data.txt") 
train_size = int(len(custom_dataset) * 0.7)   #训练集占70%的数据
validate_size = int(len(custom_dataset) * 0.2)  #验证集占20%的数据
test_size = len(custom_dataset) - validate_size - train_size  #测试集占10%的数据
train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, validate_size, test_size]) #随机划分数据集
#torch.utils.data.random_split 是 PyTorch 中用于随机划分数据集的函数。它将一个数据集随机分割成多个子集，常用于将数据集划分为训练集、验证集和测试集。

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False )  
#训练集的数据加载（batch_size=opt.batch_size: 这是每个批次中样本的数量，来自命令行参数。opt.batch_size。批次大小决定了每次迭代中处理的样本数量。）
validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False )  
#验证集的数据加载 （batch_size=1：每个batch中样本的数量，shuffle=False：是否打乱数据集）
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False )  
#测试集的数据加载 （batch_size=1：每个batch中样本的数量，shuffle=False：是否打乱数据集）
print("Training set data size:", len(train_loader)*opt.batch_size, ",Validating set data size:", len(validate_loader), ",Testing set data size:", len(test_loader)) 
# 打印数据集的大小（训练集，验证集，测试集）

# 定义推理过程，返回准确率。用于验证阶段和测试阶段
def infer(model, dataset, device):
    model.eval() #将模型设置为评估模式，这意味着在推理过程中不会计算梯度，从而减少内存消耗。
    acc_num = 0.0  #初始化，用于计算预测正确的数量
    with torch.no_grad(): #关闭梯度（上下文管理器）
        for  data in dataset:  #遍历数据集中的每一个样本
            datas, labels = data  #解包
            outputs = model(datas.to(device))  #前向传播 （将输入数据传递给模型，得到输出。同时，将数据移动到指定的设备（如 GPU）。）
            predict_y = torch.max(outputs, dim=1)[1]   #获取最大值的索引（从模型的输出中获取每个样本的预测类别。只需要索引）
            acc_num += torch.eq(predict_y, labels.to(device)).sum().item()  #计算出预测正确的数量
    accuratcy = acc_num / len(dataset)  #计算准确率
    return accuratcy 

# 定义训练，验证和测试过程
def main(args): #args 是通过 argparse 模块解析命令行参数后得到的一个命名空间对象。
#这个对象包含了从命令行传递给程序的所有参数及其对应的值。
    print(args)
 
    model = Neuralnetwork(4, 12, 6, 3).to(device) # 实例化模型 （4：输入层，12：隐藏层1，6：隐藏层2，3：输出层）
    loss_function = nn.CrossEntropyLoss() # 定义损失函数 （交叉熵损失函数）
    pg = [p for p in model.parameters() if p.requires_grad] # 定义模型参数（获取所有需要计算梯度的参数）
    optimizer = optim.Adam(pg, lr=args.lr) # 定义优化器（Adam优化器）
    
    # 定义模型权重文件存储路径
    save_path = os.path.join(os.getcwd(), 'results/weights')  #获取当前目录下results/weights的路径（os.getcwd()：获取当前工作目录）
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)  #如果路径不存在，则创建路径

    # 开始训练过程
    for epoch in range(opt.epochs):  #遍历每一个epoch（训练次数）
        ############################################################## train ######################################################
        model.train()  #将模型设置为训练模式
        acc_num = torch.zeros(1).to(device)    # 初始化，用于计算训练过程中预测正确的数量 
        #torch.zeros(1).to(device)：初始化一个大小为1的tensor，并将其移动到指定的设备上
        sample_num = 0   # 初始化，用于记录当前迭代中，已经计算了多少个样本
        
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100) # tqdm是一个进度条显示器，可以在终端打印出现在的训练进度 
        #train_loader：训练集的数据加载器，file=sys.stdout：输出到标准输出，ncols=100：进度条的长度

        for datas in train_bar :   #遍历训练集中的每一个样本
            data, label = datas  #解包（data：输入数据，label：标签）
            label = label.squeeze(-1)  #将label的形状从[batch_size, 1]转换为[batch_size] （把标签的最后一个维度去掉）
            sample_num += data.shape[0]  #计算当前batch中样本的数量

            optimizer.zero_grad()  #清空梯度 （优化器的初始化过程） （防止历史梯度对当前批次梯度的影响）
            outputs = model(data.to(device)) # output_shape: [batch_size, num_classes] （输出形状：[batch_size, 类别数]）
            pred_class = torch.max(outputs, dim=1)[1] # torch.max 返回值是一个tuple（元组），第一个元素是max值，第二个元素是max值的索引。
            acc_num += torch.eq(pred_class, label.to(device)).sum()  #计算预测正确的数量
 
            loss = loss_function(outputs, label.to(device)) # 求损失 （outputs：预测值，label：真实值）
            loss.backward() # 自动求导 （反向传播）
            optimizer.step() # 梯度下降 （更新参数）

            # print statistics 
            train_acc = acc_num.item() / sample_num  #计算当前epoch的准确率 （acc_num.item()：将tensor转换为标量）
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,  opt.epochs, loss)   #打印当前epoch的损失 
            #epoch（索引：从0开始） + 1：当前epoch的编号，opt.epochs：总epoch数，loss：当前epoch的损失

        ############################################################## validate ###################################################### 
        val_accurate = infer(model = model, dataset=validate_loader, device=device)   #计算验证集的准确率
        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, loss, train_acc, val_accurate))    
        #epoch（索引：从0开始） + 1：当前epoch的编号，loss：当前epoch的损失，train_acc：当前epoch的准确率，val_accurate：当前epoch的验证准确率
        torch.save(model.state_dict(), os.path.join(save_path, "AlexNet.pth") )  #保存模型参数

        # 每次迭代后清空这些指标，重新计算 
        train_acc = 0.0
        val_accurate = 0.0 
    print('Finished Training')

    ################################################################# test ############################################################  
    test_accurate = infer(model = model, dataset = test_loader, device= device)   #计算测试集的准确率
    print(' test_accuracy: %.3f' %  ( test_accurate))   #打印测试集的准确率

if __name__ == '__main__':   #主入口
    main(opt)



 