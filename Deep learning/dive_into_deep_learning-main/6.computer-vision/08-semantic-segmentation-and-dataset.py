import os
import torch
import torchvision
from d2l import torch as d2l


# 下载VOC2012数据集
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                        '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

# 解压VOC2012数据集
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

# 读取VOC图像和标签：将所有输入的图像和标签读入内存
def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    # 根据is_train参数确定读取训练集还是验证集
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                            'train.txt' if is_train else 'val.txt')
    # 设置读取图像的模式为RGB
    mode = torchvision.io.image.ImageReadMode.RGB
    # 打开txt文件，读取图像文件名
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    # 初始化特征和标签列表
    features, labels = [], []
    # 遍历图像文件名列表
    for i, fname in enumerate(images):
        # 读取图像
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        # 读取标签
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    # 返回特征和标签列表
    return features, labels

# 读取训练集图像和标签
train_features, train_labels = read_voc_images(voc_dir, True)

# 显示前5张图像和标签
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);

# 定义VOC颜色映射
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


# 定义VOC类别
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 构建从RGB到VOC类别索引的映射
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


# 将VOC标签中的RGB值映射到它们的类别索引
def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
        + colormap[:, :, 2])
    return colormap2label[idx]

# 将VOC标签中的RGB值映射到它们的类别索引
y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]

# 随机裁剪特征和标签图像
def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    # 获取随机裁剪的参数
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    # 根据参数裁剪特征图像
    feature = torchvision.transforms.functional.crop(feature, *rect)
    # 根据参数裁剪标签图像
    label = torchvision.transforms.functional.crop(label, *rect)
    # 返回裁剪后的特征和标签图像
    return feature, label

# 显示随机裁剪后的图像
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);

# 定义VOC语义分割数据集
class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        # 初始化函数，传入训练标志、裁剪大小和voc目录
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 对图像进行归一化处理
        self.crop_size = crop_size
        # 获取voc目录下的图像和标签
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        # 对图像进行过滤处理
        self.features = [self.normalize_image(feature)
                        for feature in self.filter(features)]
        # 对标签进行过滤处理
        self.labels = self.filter(labels)
        # 获取voc目录下的颜色映射到标签的映射
        self.colormap2label = voc_colormap2label()
        # 打印读取的图像数量
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                    *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

# 定义裁剪大小
crop_size = (320, 480)
# 加载训练集
voc_train = VOCSegDataset(True, crop_size, voc_dir)
# 加载测试集
voc_test = VOCSegDataset(False, crop_size, voc_dir)

# 定义批量大小
batch_size = 64
# 加载训练集迭代器
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
# 打印第一个批次的特征和标签形状
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break

# 加载VOC语义分割数据集
def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    # 下载并解压VOC2012数据集
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    # 获取数据加载器的线程数
    num_workers = 0
    # 加载训练集数据
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    # 加载测试集数据
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    # 返回训练集和测试集数据加载器
    return train_iter, test_iter

# 显示图像
d2l.plt.show()