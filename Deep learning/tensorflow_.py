import tensorflow as tf
import timeit

print("TensorFlow version:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())

# 检查是否有可用的GPU
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    print("No GPUs found. Please check your GPU installation.")
else:
    print(f"GPUs: {gpus}")
    try:
        # 设置GPU显存按需增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # 如果有多个GPU，仅使用第0个GPU
        tf.config.set_visible_devices(gpus[0], "GPU")
        
        # 验证是否成功配置
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        print(e)


#指定在cpu上运行
def cpu_run():
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([10000, 1000])
        cpu_b = tf.random.normal([1000, 2000])
        c = tf.matmul(cpu_a, cpu_b)
    return c
 
#指定在gpu上运行 
def gpu_run():
    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([10000, 1000])
        gpu_b = tf.random.normal([1000, 2000])
        c = tf.matmul(gpu_a, gpu_b)
    return c

cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print("cpu:", cpu_time, "  gpu:", gpu_time)
