# Creation of a ANN with backpropagation
import numpy as np  # see if it's possible to use, otherwise import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative(x):
    return x * (1 - x)


def cost_fct(x, y):
    return 0.5 * ((x - y) * (x - y))


def create_training_sample(x):
    training_sample = np.zeros(8)
    if x > 8:
        raise Exception("training sample cannot be created")
    training_sample[x] = 1
    return training_sample


# input_values = np.array([[1, 0, 0]])
# input_values = input_values.reshape(input_values.size, 1)
# output_values = input_values
# adding bias to input
# input_values = np.insert(input_values, 0, 1).reshape(input_values.size + 1, 1)
# measurement = input_values

training_set = []
for i in range(8):
    training_set.append((create_training_sample(i), create_training_sample(i)))
# TODO initialize near zero (0, 0.01^2)
layer1_weights = np.random.random_sample((9, 3)) * 0.01
layer2_weights = np.random.random_sample((4, 8)) * 0.01
layer1_activation = np.zeros(9)
layer2_activation = np.zeros(4)
layer3_activation = np.zeros(8)
layer1_activation[0] = 1
layer2_activation[0] = 1

print(layer1_weights)
print(layer2_weights)

# random weight, learning rate
# weight = [0.5]
learning_rate = 0.1
const_lambda = 0.01
cost = 0

for epoch in range(100):
    layer1_df_W = np.zeros((9, 3))
    layer2_df_W = np.zeros((4, 8))
    for m in training_set:
        layer1_activation[1:] = m[0]

        layer2_activation[1:] = np.vectorize(sigmoid)(np.dot(layer1_activation, layer1_weights))

        layer3_activation = np.vectorize(sigmoid)(np.dot(layer2_activation, layer2_weights))
        # print("layer3", layer3_activation)

        # TODO implement proper cost function 1/m sum (1/2 (aL-y)^2 + lambda/2 (W^2)
        cost += np.sum(np.vectorize(cost_fct)(layer3_activation, m[1]))

        error_4 = np.multiply(np.vectorize(derivative)(layer3_activation), (layer3_activation - m[1]))

        error_3 = np.multiply(np.dot(error_4, layer2_weights.T), np.vectorize(derivative)(layer2_activation))

        print("ground truth", m[1])
        print("prediction:", layer3_activation)

        layer1_df_W = layer1_df_W + np.dot(layer1_activation.reshape(-1, 1), error_3[1:].reshape(-1, 1).T)
        layer2_df_W = layer2_df_W + np.dot(layer2_activation.reshape(-1, 1), error_4.reshape(-1, 1).T)

    layer1_weights[0, :] = layer1_weights[0, :] - learning_rate * ((1 / len(training_set)) * layer1_df_W[0, :])
    layer2_weights[0, :] = layer2_weights[0, :] - learning_rate * ((1 / len(training_set)) * layer2_df_W[0, :])
    layer1_weights[1:, :] = layer1_weights[1:, :] - learning_rate * (
            (1 / len(training_set)) * layer1_df_W[1:, :] + const_lambda * layer1_weights[1:, :])
    layer2_weights[1:, :] = layer2_weights[1:, :] - learning_rate * (
            (1 / len(training_set)) * layer2_df_W[1:, :] + const_lambda * layer2_weights[1:, :])

    print(layer3_activation)
    # print(layer1_weights)
    # print(layer2_weights)
    print()
    print("cost", cost / len(training_set), epoch)
    cost = 0
