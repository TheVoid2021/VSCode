import torch
import torchvision # torchvision是PyTorch的官方视觉库，它包含了许多视觉图像处理的工具和数据集
from torch.utils import data 
from torchvision import transforms  # transforms:对数据进行操作
from d2l import torch as d2l

#数据读取

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式 并除以255使得所有像素的数值均在0到1之间。
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=False)
# 测试集 预测模型好坏
mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=False)
# transform=trans 使得数据集的每个样本都是经过transform操作后的torch.Tensor对象，而不是一堆图片

# print(len(mnist_train), len(mnist_test))  
# print(mnist_train[0][0].shape) # torch.Size([1, 28, 28])  每个输入图像的高度和宽度均为28像素。 数据集由黑白图像组成，其通道数为1
# print(type(mnist_train), type(mnist_train[0]), len(mnist_train[0])) 

# 展示一下数据集
def get_fashion_mnist_labels(labels):
  #返回Fashion-MNIST数据集的文本标签
  text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
  return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
  # 绘制图像列表
  # 每一个图像画布的大小
  figsize = (num_cols * scale, num_rows * scale)
  print(figsize)
  # d2l.plt.subplots（行数，列数，切成第几个小图）
  # subplots()函数返回值的类型为元组，元组中包含两个元素：第一个元素表示一个画布，第二个是元素表示所包含多个子图像的array数组对象。
  _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
  # 表示把axes数组降到一维,默认为按行的方向降 返回一个一维数组 子图像对象
  axes = axes.flatten()
  # print(axes)
  # enumerate 输出数值顺序和值 (0,data)
  for i, (ax, img) in enumerate(zip(axes, imgs)):
    # print(i, ax, 'img')
    # 判断图片是什么类型数据表示的
    if torch.is_tensor(img):
      # 图片张量
      ax.imshow(img.numpy()) # 绘图张量数据要转成numpy数组
    else:
      # PIL图片 PIL图片可以直接展示
      ax.imshow(img)
    ax.axes.get_xaxis().set_visible(False) # 设置坐标轴是上的刻度不可见,但轴仍在
    ax.axes.get_yaxis().set_visible(False) # 设置坐标轴是上的刻度不可见,但轴仍在
    if titles:
      ax.set_title(titles[i]) # 设置坐标系的title
  return axes # 返回画好图的坐标系

# next: 这里返回第一个batch的数据
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# 根据y的索引值，获取y的标签
show_images(X.reshape(18,28,28), 2, 9, titles=get_fashion_mnist_labels(y)) # reshape(18,28,28) 将数据集的每个样本变成28*28的图像 2行9列 样本数量18 
# titles=get_fashion_mnist_labels(y) 根据y的索引值，获取y的标签(字符串)

# 读取一小批量数据，大小为batch_size
batch_size = 256  
# 减少batch_size（如减少到1）会影响读取性能
# batch_size = 1  
def get_dataloader_workers():
  #使用0个进程来读取数据
  return 0   #windows环境num_workers只能设置为0!!! linux可以设置为0或大于0

# 用来创建一个数据加载器（DataLoader）的，用于在训练神经网络时批量加载数据。
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()) #train_iter是一个迭代器

# 得出扫一遍数据的时间花费
timer = d2l.Timer()
for X, y in train_iter:  #一个一个来访问batch
  continue
print(f'{timer.stop():.2f} sec')  

# 如果模型需要更大输入，可以用resize让图片更大一些，即resize可以调整图片大小
def load_data_fashion_mnist(batch_size, resize=None):
  # 数据预处理
  trans = [transforms.ToTensor()] # 为什么加中括号，放在列表里 方便后面如果有加resize操作，给trans操作能排序且同时执行。
  if resize:
    trans.insert(0, transforms.Resize(resize))
  trans = transforms.Compose(trans) # transforms.Compose()函数将多个步骤整合到一起，方便使用。
  mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
  mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
  return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), 
      data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers())
      )
  
# 调用函数，加载训练集和测试集，每个批次32个样本，图像大小调整为64x64
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
  print(X.shape, X.dtype, y.shape, y.dtype) # 查看数据形状和数据类型
  break
