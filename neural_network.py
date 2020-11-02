# Creation of a ANN with backpropagation
import numpy as np  # see if it's possible to use, otherwise import math
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative(x):
    return x * (1 - x)


def cost_fct(x, y):
    return 0.5 * (pow(abs(x - y), 2))


def cost_fct2(x, y):
    return x - y


def cost_fct3(x, y):
    return 0.5 * (np.power((x - y), 2))


def create_training_sample(x):
    training_sample = np.zeros(8)
    if x > 8:
        raise Exception("training sample cannot be created")
    training_sample[x] = 1
    return training_sample


training_set = []
for i in range(8):
    training_set.append((create_training_sample(i), create_training_sample(i)))
# TODO initialize near zero (0, 0.01^2)
layer1_weights = np.random.random_sample((9, 3)) * 0.001
layer2_weights = np.random.random_sample((4, 8)) * 0.001
layer1_activation = np.zeros(9).reshape(-1, 1)
layer2_activation = np.zeros(4).reshape(-1, 1)
layer3_activation = np.zeros(8).reshape(-1, 1)
layer1_activation[0] = 1
layer2_activation[0] = 1

# print(layer1_weights)
# print(layer2_weights)

# learning rate and weight decay
learning_rate = 2
const_lambda = 0
cost_iter = np.array([])
training_epochs = 2000


for epoch in range(training_epochs):
    cost = 0
    layer1_df_W = np.zeros((9, 3))
    layer2_df_W = np.zeros((4, 8))
    batch = np.random.permutation(training_set)
    for m in batch:
        layer1_activation[1:, :] = m[0].reshape(-1, 1)

        layer2_activation[1:, :] = sigmoid(np.dot(layer1_activation.T, layer1_weights).T)

        layer3_activation = sigmoid(np.dot(layer2_activation.T, layer2_weights).T)

        if epoch == 1999:
            print(m[0])
            print(layer3_activation.T)
            print()
        # print("layer3", layer3_activation)

        cost += np.sum(cost_fct3(layer3_activation, m[1]))

        delta_3 = np.multiply(layer3_activation.T - m[1], derivative(layer3_activation).T)

        delta_2 = np.multiply(np.dot(delta_3, layer2_weights.T), derivative(layer2_activation).T)

        layer2_df_W += np.dot(layer2_activation, delta_3)
        layer1_df_W += np.dot(layer1_activation, delta_2[:, 1:])

        # print("layer2", layer2_df_W)
        # print("layer1", layer1_df_W)

    m = len(training_set)
    layer1_weights[0, :] -= learning_rate * ((1 / m) * layer1_df_W[0, :])
    layer2_weights[0, :] -= learning_rate * ((1 / m) * layer2_df_W[0, :])
    layer1_weights[1:, :] -= learning_rate * (
            ((1 / m) * layer1_df_W[1:, :]) + (const_lambda * layer1_weights[1:, :]))
    layer2_weights[1:, :] -= learning_rate * (
            ((1 / m) * layer2_df_W[1:, :]) + (const_lambda * layer2_weights[1:, :]))


    cost_Wb = (cost/m) + const_lambda / 2 * (np.sum(pow(layer1_weights, 2)) + np.sum(pow(layer2_weights, 2)))
    print("Total Cost", cost_Wb, epoch)
    cost_iter = np.append(cost_iter, cost_Wb)

plt.plot(cost_iter)
plt.xlabel("Iterations")
plt.ylabel("Cost function")
plt.title(f"Progression of the cost function over {training_epochs} epochs")
plt.show()

