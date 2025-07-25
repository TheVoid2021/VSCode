# from mmdet.apis import init_detector, inference_detector

# config_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
# checkpoint_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'
# model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
# inference_detector(model, 'demo/demo.jpg')
import mmcv
import mmcv._ext
print("MMCV版本:", mmcv.__version__)
print("扩展模块加载成功")