import mnist_loader
import network2
import matplotlib.pyplot as plt

# n = 128

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

lr = 1.0
n = 64
lmbda = 10
# for lr in [100]:# 0.001, 0.01, 0.1, 1, 10,
# for lmbda in [0.001, 0.01, 0.1, 1, 10, 100]:

net = network2.Network([784, n, 10], cost=network2.CrossEntropyCost)
net.default_weight_initializer()

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
    net.SGD(
        training_data, 50, 10, lr,
        evaluation_data=test_data,
        lmbda=lmbda,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
# net.save('./models/pic2epo_{}_{}_{}_{}.txt'.format(n, lr, False, lmbda))
epoch = [i for i in range(len(evaluation_accuracy))]
# plt.title('隐藏层个数：{}'.format(n))
plt.figure()
plt.subplot(2, 1, 1)
# plt.xlabel('epoch')
plt.ylabel('cost')
plt.plot(epoch, evaluation_cost, label='evaluation_cost')
plt.plot(epoch, training_cost, label='training_cost')
plt.legend()
plt.subplot(2, 1, 2)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(epoch, evaluation_accuracy, label='evaluation_accuracy')
plt.plot(epoch, training_accuracy, label='training_accuracy')
plt.legend()
plt.savefig('./img/pic_{}_{}_{}_{}.png'.format(n, lr, True, lmbda))
# plt.show()
