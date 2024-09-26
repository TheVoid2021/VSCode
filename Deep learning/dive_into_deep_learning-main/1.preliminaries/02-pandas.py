import os
import numpy as np
import pandas as pd
import torch
from numpy import nan as NaN

#• pandas软件包是Python中常用的数据分析工具中，pandas可以与张量兼容。
#• 用pandas处理缺失的数据时，我们可根据情况选择用插值法和删除法。

# 创建一个人工数据集，并储存在house_tiny.csv文件中
os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 在上级目录创建data文件夹
datafile = os.path.join('..', 'data', 'house_tiny.csv')  # 创建文件
with open(datafile, 'w') as f:  # 往文件中写数据
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 第1行的值
    f.write('2,NA,106000\n')  # 第2行的值
    f.write('4,NA,178100\n')  # 第3行的值
    f.write('NA,NA,140000\n')  # 第4行的值

# 用pandas从csv文件中加载原始数据集 
data = pd.read_csv(datafile)  # 可以看到原始表格中的空值NA被识别成了NaN
print('1.原始数据:\n', data)

#通过位置索引iloc，我们将data分成inputs和outputs，其中前者为data的前两列，而后者为data的最后一列。
#对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
inputs, outputs = data.iloc[:, 0: 2], data.iloc[:, 2]  
inputs = inputs.select_dtypes(include=[np.number])  # 选择数值类型的列
inputs = inputs.fillna(inputs.mean())  # 用均值填充NaN
print(inputs)
print(outputs)


# 利用pandas中的get_dummies函数来处理离散值或者类别值。
# [对于 inputs 中的类别值或离散值，我们将 "NaN" 视为一个类别。] 由于 "Alley"列只接受两种类型的类别值 "Pave" 和 "NaN"
inputs = pd.get_dummies(inputs, dummy_na=True)
print('2.利用pandas中的get_dummies函数处理:\n', inputs) 

x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print('3.转换为张量：')
print(x)
print(y)

# 扩展填充函数fillna的用法
df1 = pd.DataFrame([[1, 2, 3], [NaN, NaN, 2], [NaN, NaN, NaN], [8, 8, NaN]])  # 创建初始数据
print('4.函数fillna的用法：')
print(df1)
print(df1.fillna(100))  # 用常数填充 ，默认不会修改原对象
print(df1.fillna({0: 10, 1: 20, 2: 30}))  # 通过字典填充不同的常数，默认不会修改原对象 列：值
print(df1.fillna(method='ffill'))  # 用前面的值来填充


df2 = pd.DataFrame(np.random.randint(0, 10, (5, 5)))  # 随机创建一个5*5
df2.iloc[1:4, 3] = NaN
df2.iloc[2:4, 4] = NaN  # 指定的索引处插入值
print(df2)
print(df2.fillna(method='bfill', limit=2))  # 限制填充个数
print(df2.fillna(method="ffill", limit=1, axis=1))  
# method="ffill"：表示使用前一个有效值来填充缺失值。
# limit=1：限制每一行最多只填充一个缺失值。
# axis=1：表示沿着列的方向进行填充。

#创建包含更多行和列的原始数据集。
data_test_file = os.path.join('..', 'data', 'house_tiny_test.csv')
with open(data_test_file, 'w') as f:
    f.write('NumRooms,Alley,Salary,Price\n') # 列名
    f.write('NA,Pave,4500,127500\n') # 每行表示一个数据样本
    f.write('2,NA,NA,106000\n')
    f.write('4,NA,8000,178100\n')
    f.write('NA,NA,6000,140000\n')
    f.write('5,NA,NA,110000\n')
    f.write('8,NA,7500,163000\n')
    f.write('NA,Pave,5000,125000\n')

data_test = pd.read_csv(data_test_file) #读取自定义数据集
print(data_test)

#1.删除缺失值最多的列。
data_test = data_test.drop('Alley',axis=1)
print(data_test)

#2.处理缺失值
input_data, output_data = data_test.iloc[:, 0:2], data_test.iloc[:, 2] #提取输入和输出
input_data = input_data.fillna(input_data.mean()) #用均值填充输入
print(input_data)
print(output_data)

#3.转换为张量格式
X, y = torch.tensor(input_data.values), torch.tensor(output_data.values)
print(X, y)