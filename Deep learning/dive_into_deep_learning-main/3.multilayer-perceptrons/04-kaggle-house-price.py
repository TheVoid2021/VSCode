import hashlib
import os
import tarfile
import zipfile
import requests

import pandas as pd
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
def download(name, cache_dir=os.path.join('./', 'data')):
  #下载一个DATA_HUB中的文件，返回本地文件名#
  assert name in DATA_HUB, f"{name} 不存在于{DATA_HUB}"
  url, sha1_hash = DATA_HUB[name]
  os.makedirs(cache_dir, exist_ok=True)
  fname = os.path.join(cache_dir, url.split('/')[-1])
  if os.path.exists(fname):
    sha1 = hashlib.sha1()
    with open(fname, 'rb')as f:
      while True:
        data = f.read(1048576)
        if not data:
          break
        sha1.update(data)
      if sha1.hexdigest() == sha1_hash:
        return fname # 命中缓存
  print(f'正在从{url}下载{fname}...')
  r = requests.get(url, stream=True, verify=True)
  with open(fname, 'wb')as f:
    f.write(r.content)
  return fname

def download_extract(name, folder=None):
  #下载并解压zip/tar文件#
  fname = download(name)
  base_dir = os.path.dirname(fname)
  data_dir, ext = os.path.splitext(fname)
  if ext == '.zip':
    fp = zipfile.ZipFile(fname, 'r')
  elif ext in ('.tar', '.gz'):
    fp = tarfile.open(fname, 'r')
  else:
    assert False, '只有zip/tar文件可以被解压缩'
  fp.extractall(base_dir)
  return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
  #下载DATA_HUB中的所有文件#
  for name in DATA_HUB:
    download(name)



# Kaggle房价预测竞赛
# Kaggle房价预测竞赛是Kaggle上非常受欢迎的竞赛之一，它要求我们根据给定的特征预测房价。
# Kaggle房价预测竞赛的数据集包括1460个训练样本和1459个测试样本，每个样本有79个特征。
# 我们将使用这些数据来训练一个线性回归模型，并使用测试数据集进行预测。

# 下载并读取数据集
DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# 下载并读取训练数据集
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
# 训练数据集包括1460个样本，每个样本80个特征和1个标签， 而测试数据集包含1459个样本，每个样本80个特征
print(train_data.shape)  # (1460, 81)
print(test_data.shape)   # (1459, 80)
# 索引器 基于整数位置（integer location）选择数据。它通常用于按照行和列的整数位置进行数据选择和切片操作。
# .iloc[行， 列]  看前四个和最后两个特征，以及相应标签（房价）
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]) #0到3 4行样本 ，以及前四列特征和后两列特征 最后加上一个最后一列的标签
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])) #去掉id列 这是不能用的，提交结果的时候用
print(all_features.shape) # (2919, 79)

# 数据预处理
# 将所有缺失的值替换为相应列的平均值。通过将特征重新缩放到零均值和单位方差来标准化数据
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  #区别文本和数值特征
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
print(all_features.shape) # (2919, 79)
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True) #处理离散值，用一次独热编码替换它们
print(all_features.shape) # (2919, 330) 1460+1459=2919  


n_train = train_data.shape[0]
# 通过values属性，我们可以 从pandas格式中提取NumPy格式，并将其转换为张量表示用于训练
train_np = np.array(all_features[:n_train].values, dtype=np.float32) # NumPy默认浮点数类型为float64，而PyTorch的默认浮点数类型为float32
train_features = torch.tensor(train_np, dtype=torch.float32)
test_np = np.array(all_features[n_train:].values, dtype=np.float32)
test_features = torch.tensor(test_np, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 训练
loss = nn.MSELoss() # 计算均方损失
in_features = train_features.shape[1] # 输入特征维度
def get_net():
  net = nn.Sequential(nn.Linear(in_features,1)) #单层的线性回归
  return net

# 更关心相对误差，用取对数来衡量差异
def log_rmse(net, features, labels):
  # 为了在取对数时进一步稳定该值，将小于1的值设置为1
  clipped_preds = torch.clamp(net(features), 1, float('inf'))
  rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
  return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
  train_ls, test_ls = [], []
  train_iter = d2l.load_array((train_features, train_labels), batch_size)
  # 这里使用的是Adam优化算法 类似比较于平滑的SGD，对学习率不会那么敏感
  optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate,weight_decay = weight_decay)
  for epoch in range(num_epochs):
    for X, y in train_iter:
      optimizer.zero_grad()
      l = loss(net(X), y)
      l.backward()
      optimizer.step()
    train_ls.append(log_rmse(net, train_features, train_labels))
    if test_labels is not None:
      test_ls.append(log_rmse(net, test_features, test_labels))
  return train_ls, test_ls

# K折交叉验证
def get_k_fold_data(k, i, X, y):
  # 这是一个断言语句，用于确保 k 的取值大于1，否则会触发 AssertionError 异常。
  assert k > 1
  # 计算每折的大小：根据样本数和折数 k，计算每折的大小，即每个验证集的样本数
  fold_size = X.shape[0] // k
  X_train, y_train = None, None
  # 分割数据：使用循环遍历每个折，根据当前折的索引 j 计算出当前折的索引范围，
  # 然后从输入的特征数据 X 和标签数据 y 中切片得到当前折的训练数据和验证数据
  for j in range(k):
    idx = slice(j * fold_size, (j + 1) * fold_size)
    X_part, y_part = X[idx, :], y[idx]
    if j == i:
      X_valid, y_valid = X_part, y_part
    elif X_train is None:
      X_train, y_train = X_part, y_part
    else:
      X_train = torch.cat([X_train, X_part], 0)
      y_train = torch.cat([y_train, y_part], 0)
  # 返回数据：根据当前折的索引 i，将当前折的数据作为验证集（X_valid 和 y_valid），
  # 其余折的数据拼接起来作为训练集（X_train 和 y_train），最终返回训练集和验证集的特征数据和标签数据。
  return X_train, y_train, X_valid, y_valid

# K折交叉验证 返回训练集和验证集的平均log rmse
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
  train_l_sum, valid_l_sum = 0, 0
  for i in range(k):
    data = get_k_fold_data(k, i, X_train, y_train)
    net = get_net()
    train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
    train_l_sum += train_ls[-1]
    valid_l_sum += valid_ls[-1]
    #if i == 0:
    d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
    d2l.plt.show()
    print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}', f'验证log rmse{float(valid_ls[-1]):f}')
  return train_l_sum / k, valid_l_sum / k   #求和做平均

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}', f'平均验证log rmse: {float(valid_l):f}') 
