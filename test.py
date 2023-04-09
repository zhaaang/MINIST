import mnist_loader
import network2
import matplotlib.pyplot as plt
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.load('./models/net_50.txt')
acc = net.accuracy(test_data)

print('该模型在测试集上的精确度为：', acc/len(test_data))

# 假设您已经有一个训练好的三层全连接神经网络的参数，分别为W1, b1, W2, b2, W3, b3
W1 = net.weights[0]
b1 = net.biases[0].reshape(-1)
W2 = net.weights[1]
b2 = net.biases[1].reshape(-1)
# 将参数组织为一个列表
# params = [W1, b1, W2, b2]
#
# # 遍历每层的参数，并绘制热力图
# for i in range(len(params) // 2):
#     print('第{}层参数'.format(i))
#     W = params[i * 2]
#     b = params[i * 2 + 1]
#     fig, ax = plt.subplots()
#     ax.imshow(W, cmap='gray')
#     ax.set_title('Layer {} Weights'.format(i + 1))
#     ax.set_xlabel('Input Units')
#     ax.set_ylabel('Output Units')
#     plt.show()
#
#     fig, ax = plt.subplots()
#     ax.bar(np.arange(len(b)), b)
#     ax.set_title('Layer {} Biases'.format(i + 1))
#     ax.set_xlabel('Output Units')
#     ax.set_ylabel('Bias Values')
#     plt.show()

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# 绘制左上子图
axs[0, 0].imshow(W1, cmap='gray')
axs[0, 0].set_title('Layer {} Weights'.format(1))
axs[0, 0].set_xlabel('Input Units')
axs[0, 0].set_ylabel('Output Units')

# 绘制右上子图
axs[0, 1].bar(np.arange(len(b1)), b1)
axs[0, 1].set_title('Layer {} Biases'.format(1))
axs[0, 1].set_xlabel('Output Units')
axs[0, 1].set_ylabel('Output Units')

# 绘制左下子图
axs[1, 0].imshow(W2, cmap='gray')
axs[1, 0].set_title('Layer {} Weights'.format(2))
axs[1, 0].set_xlabel('Input Units')
axs[1, 0].set_ylabel('Output Units')

# 绘制右下子图
axs[1, 1].bar(np.arange(len(b2)), b2)
axs[1, 1].set_title('Layer {} Biases'.format(2))
axs[1, 1].set_xlabel('Output Units')
axs[1, 1].set_ylabel('Output Units')
fig.tight_layout()
plt.show()
