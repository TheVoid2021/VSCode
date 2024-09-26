from torch.utils.data import Dataset  #Dataset是PyTorch中用于处理数据集的抽象类，它提供了一种标准化的方式来加载和处理数据。
import os  #os模块提供了与操作系统交互的功能，包括文件和目录操作、环境变量管理等。
import pandas as pd  #Pandas 是一个用于数据操作和分析的 Python 库。Pandas 提供了两种主要的数据结构：
    #Series：一维数组，可以存储任意数据类型（整数、浮点数、字符串、Python 对象等）。
    #DataFrame：二维表格型数据结构，可以存储不同类型的数据（如整数、浮点数、字符串等），类似于 Excel 表格或 SQL 表。
import numpy as np  #Numpy 是一个用于科学计算的 Python 库，主要用于处理大型多维数组和矩阵，它提供了高性能的多维数组对象和一系列用于处理这些数组的工具。
    #多维数组对象（ndarray）：NumPy 的核心是多维数组对象（ndarray），它提供了高效的数组操作和计算能力。
    #广播机制：NumPy 支持广播（broadcasting），允许不同形状的数组进行算术运算，这在处理多维数据时非常有用。
    #数学函数库：NumPy 提供了大量的数学函数，包括线性代数、傅里叶变换、随机数生成等。
    #数据读写：NumPy 支持从文件中读取和写入数组数据，支持多种文件格式（如 CSV、文本文件、二进制文件等）。
    #线性代数、随机数生成：NumPy 提供了丰富的线性代数函数和随机数生成函数，方便进行科学计算。
import torch  #PyTorch 是一个用于深度学习的开源框架，主要用于构建和训练神经网络模型。
    #张量（Tensor）：PyTorch 的核心是张量，它是一个多维数组，类似于 NumPy 的 ndarray，但提供了 GPU 加速和自动求导功能。
    #自动求导：PyTorch 提供了自动求导机制，可以自动计算张量的梯度，这对于训练神经网络非常有用。
    #神经网络模块：PyTorch 提供了丰富的神经网络模块，包括卷积神经网络（CNN）、循环神经网络（RNN）、长短期记忆网络（LSTM）等。
    #优化器：PyTorch 提供了多种优化器，如随机梯度下降（SGD）、Adam、RMSprop 等，用于训练神经网络。
    #数据加载器：PyTorch 提供了 DataLoader，用于加载和处理数据集，支持批量处理、数据增强等功能。
    #PyTorch 还提供了许多其他功能，如模型保存和加载、可视化工具、分布式训练等，方便进行深度学习研究和开发。

    
#当我们在定义一个数据集加载的时候，要继承Dataset这个副类，就必须实现三个函数：__init__—__len__和__getitem__
class iris_dataload(Dataset):
    def __init__(self, data_path: str, transform=None):  #初始化函数：设置数据集的路径并且加载数据集
        self.data_path = data_path  #数据集的路径
        self.transform = transform  #数据预处理
 
        assert os.path.exists(data_path), "dataset root: {} does not exist.".format(data_path)  #判断数据集路径是否存在
        df=pd.read_csv(self.data_path , names=[0,1,2,3,4])  #读取数据集
        d={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}  #标签映射 （把这些类名都起一个数值版本的别名）
        df[4]=df[4].map(d)   #将数据集中的标签（第四列）映射为数字
        data=df.iloc[:,0:4]  #获取数据 （从第零行到最后一行，第零列到第四列）
        label=df.iloc[:,4:]  #获取标签 （从第零行到最后一行，标签只用第四列）

        data=np.array(data)  #将数据转换为numpy数组  
        data = (data - np.mean(data) )/ np.std(data)  #对数据进行标准化处理（Z值化）。
            #具体来说，它将数据的每个特征值减去该特征的均值，然后除以该特征的标准差。这样处理的目的是使数据的均值为0，标准差为1，
            #从而使得数据在训练模型时更稳定，收敛速度更快。
        label=np.array(label)  #将标签转换为numpy数组

        self.data=torch.from_numpy(np.array(data,dtype='float32') )   #将数据从数组转换为torch张量
        self.label= torch.from_numpy(np.array(label,dtype='int64') )  #将标签从数组转换为torch张量

        self.data_num = len(label)  # 存储训练集的所有图片路径
        print("{} images were found in the dataset.".format(self.data_num))

    def __len__(self):  #获取数据集的大小（数据集中样本的数量）
        return self.data_num
    
    def __getitem__(self, idx):  #通过索引获取数据集中的第idx个元素
        self.data = list(self.data) #将数据转换为列表，利于索引
        self.label = list(self.label) #将标签转换为列表，利于索引
        return self.data[idx], self.label[idx]  #返回数据和标签


 