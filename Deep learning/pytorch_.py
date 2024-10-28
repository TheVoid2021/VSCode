import torch

# 检查 PyTorch 和 CUDA 版本
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print("Is CUDA available:", cuda_available)

# 检查 GPU 设备数量
device_count = torch.cuda.device_count()
print("CUDA device count:", device_count)

if cuda_available:
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# 检查PyTorch是否支持CUDA
def try_gpu(i=0): #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
def try_all_gpus(): #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
        for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())

# 检查PyTorch是否支持CUDA，并返回GPU数量
print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))

# 检查PyTorch是否支持CUDA
print(torch.cuda.is_available())

# 检查PyTorch是否支持CUDA，并返回GPU数量
print(torch.cuda.device_count())


x = torch.tensor([1, 2, 3])
print(x.device)

# 尝试运行简单的 CUDA 操作
if cuda_available:
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        y = x ** 2
        print("CUDA computation result:", y)
    except RuntimeError as e:
        print("CUDA computation failed:", e)
