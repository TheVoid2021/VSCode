# import torch
# import torch.nn.functional as F

# x = torch.tensor([1.0, 2.0, 3.0])
# softmax = F.softmax(x, dim=0)
# log_softmax = F.log_softmax(x, dim=0)

# print("Softmax:", softmax)
# print("Log Softmax:", log_softmax)


# # softmax 和 log_softmax 都是激活函数，用于将神经网络的输出转换为概率分布。它们的主要区别在于计算方式和应用场景。
# # softmax 函数将神经网络的输出转换为概率分布，使得所有输出的和为 1。它通常用于多分类问题，将每个类别的得分转换为概率。
# # softmax 函数的优点是计算简单，易于理解和实现。但是，当输出值非常大时，可能会导致数值溢出。

# # log_softmax 函数是对 softmax 函数取对数，将神经网络的输出转换为对数概率分布。
# # 它通常用于多分类问题，将每个类别的得分转换为对数概率。
# # log_softmax 函数的优点是可以避免 softmax 函数的数值溢出问题。当输出值非常大时，
# # log_softmax 函数的计算结果仍然是一个合理的数值。此外，log_softmax 函数可以与交叉熵损失函数一起使用，从而简化计算。



