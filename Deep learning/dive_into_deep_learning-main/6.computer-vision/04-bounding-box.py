import torch
from d2l import torch as d2l

# 设置图像大小
d2l.set_figsize()
# 读取图像
img = d2l.plt.imread('Deep learning/dive_into_deep_learning-main/6.computer-vision/img/catdog.jpg')
# 显示图像
d2l.plt.imshow(img);

# 定义在这两种表示法之间进行转换的函数：box_corner_to_center从两角表示法转换为中心宽度表示法，而box_center_to_corner反之亦然。
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    # 获取左上角和右下角的坐标
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # 计算中心点的坐标
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # 计算宽度和高度
    w = x2 - x1
    h = y2 - y1
    # 将中心点、宽度和高度堆叠成一个新的张量
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    # 获取中心点、宽度和高度的坐标
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # 计算左上角的坐标
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    # 计算右下角的坐标
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    # 将左上角和右下角的坐标堆叠成一个新的张量
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# bbox是边界框的英文缩写
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

# 将边界框转换为中间，宽度，高度格式
boxes = torch.tensor((dog_bbox, cat_bbox))
# 将中间，宽度，高度格式的边界框转换回左上，右下格式
box_center_to_corner(box_corner_to_center(boxes)) == boxes


def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：((左上x,左上y),宽,高)
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

# 显示图像
fig = d2l.plt.imshow(img)
# 在图像上添加狗的边界框
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
# 在图像上添加猫的边界框
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));

# 显示图像
d2l.plt.show()