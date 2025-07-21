import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

"""类别预测层"""
# 定义类别预测器：预测锚框的类别（num_inputs：输入通道数，num_anchors：锚框数量，num_classes：类别数量，
# 输出通道数：num_anchors * (num_classes + 1)，kernel_size：卷积核大小，padding：填充）
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)

"""边界框预测层"""
# 边界框预测层的设计与类别预测层的设计类似。 唯一不同的是，这里需要为每个锚框预测4个偏移量，而不是𝑞+1个类别。
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

"""连结多尺度的预测"""
# 不同尺度下预测输出的形状可能会有所不同， 为了将这两个预测输出链接起来以提高计算效率，
# 我们将把这些张量转换为更一致的格式，方便后面进行loss计算等，而不用对每个不同的尺度做loss。
def forward(x, block):
    return block(x)

# 使用类别预测器进行前向传播
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape

# 除了批量大小这一维度外，其他三个维度都具有不同的尺寸。通道维包含中心相同的锚框的预测结果，
# 我们首先将通道维移到最后一维（这样每个像素预测值是连续的）。
# 然后后三个维度拉平，形状变为（批量大小，高 × 宽 × 通道数），以便后面进行连结。
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

# 定义拼接预测函数
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# 使用拼接预测函数进行前向传播
concat_preds([Y1, Y2]).shape

# 高宽减半block
# 定义下采样块
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

# 使用下采样块进行前向传播
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape

# 基本网络块
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

# 使用基础网络进行前向传播
forward(torch.zeros((2, 3, 256, 256)), base_net()).shape

# 定义获取块函数
"""
完整的SSD由五个模块组成，每个块生成的特征图既用于生成锚框，又用于预测这些锚框的类别和偏移量。
在这五个模块中，第一个是基本网络块，第二个到第四个是高和宽减半块，最后一个模块使用全局最大池将高度和宽度都降到1。
"""
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

# 定义每个块的前向传播
# 与图像分类任务不同，此处的输出包括：CNN特征图Y；在当前尺度下根据Y生成的锚框；
# 预测的这些锚框的类别和偏移量（基于Y）。（图片分类就只有输入X输出Y）
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

# 定义锚框大小和比例 选取合适超参数
"""
在 2.2 多尺度目标检测中，接近顶部的多尺度特征块，特征图较小，是用于检测较大目标的，
因此需要生成更大的锚框（缩放比scale更大，也就是参数size）。
  在下面，0.2和1.05之间的区间被均匀分成五个部分，以确定五个模块的在不同尺度下的较小值：0.2、0.37、0.54、0.71和0.88。
  之后，他们较大的值由\sqrt{0.2 × 0.37} = 0.272 、\sqrt{0.37 × 0.54} = 0.447 等给出。
（每个尺度块size选两个参数，ratios三个取值，则每个像素生成4个锚框）
"""
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

# 定义TinySSD类
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        # 初始化函数，接收num_classes和kwargs参数
        super(TinySSD, self).__init__(**kwargs)
        # 调用父类的初始化函数
        self.num_classes = num_classes
        # 定义一个列表，存储每个block的输入通道数
        idx_to_in_channels = [64, 128, 128, 128, 128]
        # 遍历5个block
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            # 将每个block赋值给self.blk_i
            setattr(self, f'blk_{i}', get_blk(i))
            # 将每个分类预测器赋值给self.cls_i
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            # 将每个边界框预测器赋值给self.bbox_i
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        # 初始化anchors、cls_preds、bbox_preds为5个None
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        # 遍历5个block
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            # 调用blk_forward函数，传入X、self.blk_i、sizes[i]、ratios[i]、self.cls_i、self.bbox_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        # 将anchors按dim=1拼接
        anchors = torch.cat(anchors, dim=1)
        # 调用concat_preds函数，传入cls_preds
        cls_preds = concat_preds(cls_preds)
        # 将cls_preds reshape为(cls_preds.shape[0], -1, self.num_classes + 1)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        # 调用concat_preds函数，传入bbox_preds
        bbox_preds = concat_preds(bbox_preds)
        # 返回anchors、cls_preds、bbox_preds
        return anchors, cls_preds, bbox_preds

# 创建TinySSD实例
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

"""第一个模块输出特征图的形状为32 × 32 32 \times 3232×32。第二到第四个模块为高和宽减半块，第五个模块为全局汇聚层。
由于以特征图的每个单元为中心有4 44个锚框生成，因此在所有五个尺度下，每个图像总共生成( 3 2 2 + 1 6 2 + 8 2 + 4 2 + 1 ) × 4 = 5444 (32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444(32 
2+162+82+4 2+1)×4=5444个锚框（w*h*a）。
"""
print('output anchors:', anchors.shape) 
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)

# 加载香蕉数据集
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

# 将模型和数据移动到GPU上
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

"""
目标检测有两种类型的损失：锚框类别的损失和对于边界框偏移量的损失。
前者使用交叉熵损失函数计算，后者使用L1损失函数。之所以不用L2损失，是因为很多锚框离边界框很远，平方之后数值会特别大。
我们只关心几个比较好的锚框，那些离得远的锚框根本不care，所以也不需要MSE那样讲误差大的进行平方加权。
掩码变量bbox_masks令负类锚框和填充锚框不参与损失的计算。 
最后，我们将锚框类别和偏移量的损失相加，以获得模型的最终损失函数。

"""
# 定义损失函数
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

# 定义计算损失函数
def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

# 由于偏移量使用了L1范数损失，我们使用平均绝对误差来评价边界框的预测结果。
# 这些预测结果是从生成的锚框及其预测偏移量中获得的。

# 定义类别评估函数
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

# 定义边界框评估函数
def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# 定义训练过程
"""
这里的逻辑是每个多尺度锚框经过net预测其类别和对真实边界框的偏移量，这是预测值。
然后通过真实边界框，使用d2l.multibox_target函数标注锚框的真实类别和偏移量，这是真实值。二者的差距就是训练损失。
"""
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')

# 加载测试图片
X = torchvision.io.read_image('Deep learning/dive_into_deep_learning-main/6.computer-vision/img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

# 定义预测函数
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

# 进行预测
output = predict(X)

# 定义显示函数
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

# 显示预测结果
display(img, output.cpu(), threshold=0.9)
d2l.plt.show()