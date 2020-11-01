# Creation of a ANN with backpropagation
import numpy as np  # see if it's possible to use, otherwise import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative(x):
    if x == 1:
        return 1
    return x * (1 - x)


def cost_fct(x, y):
    return 0.5 * ((x - y) * (x - y))


def create_training_sample(x):
    training_sample = np.zeros(3)
    if x > 8:
        raise Exception("training sample cannot be created")
    training_sample[x] = 1
    return training_sample


training_set = []
for i in range(3):
    training_set.append((create_training_sample(i), create_training_sample(i)))
# TODO initialize near zero (0, 0.01^2)
layer1_weights = np.array([[0.00482732], [0.00345301], [0.00315615], [0.00713093]])
layer2_weights = np.array([[0.00238607, 0.00309865, 0.00011428], [0.00153557, 0.00921564, 0.00350693]])
layer1_activation = np.zeros(4)
layer2_activation = np.zeros(2)
layer3_activation = np.zeros(3)
layer1_activation[0] = 1
layer2_activation[0] = 1

print(layer1_weights)
print(layer2_weights)

# random weight, learning rate
learning_rate = 0.001
const_lambda = 0.01
cost = 0

for epoch in range(1):
    layer1_df_W = np.zeros((4, 1))
    layer2_df_W = np.zeros((2, 3))
    for m in training_set:
        layer1_activation[1:] = m[0]

        print("layer1", layer1_activation)
        layer2_activation[1:] = np.vectorize(sigmoid)(np.dot(layer1_activation, layer1_weights))
        print("layer2", layer2_activation)
        layer3_activation = np.vectorize(sigmoid)(np.dot(layer2_activation, layer2_weights))
        print("layer3", layer3_activation)

        # TODO implement proper cost function 1/m sum (1/2 (aL-y)^2 + lambda/2 (W^2)
        cost += np.sum(np.vectorize(cost_fct)(layer3_activation, m[1]))

        error_4 = np.multiply(np.vectorize(derivative)(layer3_activation), (layer3_activation - m[1]))
        print("error4", error_4)

        print(error_4.shape)
        print(layer2_weights.shape)
        test = np.dot(layer2_weights, error_4)
        print(test)
        error_3 = np.multiply(np.vectorize(derivative)(layer2_activation), np.dot(layer2_weights, error_4))
        print("error3", error_3)

        layer1_df_W = layer1_df_W + np.outer(layer1_activation, error_3[1:])
        layer2_df_W = layer2_df_W + np.outer(layer2_activation, error_4)
        print(layer1_df_W)
        print(layer2_df_W)

    # TODO don't apply weight decay to bias
    layer1_weights = layer1_weights - learning_rate * (
            (1 / len(training_set)) * layer1_df_W + const_lambda * layer1_weights)
    layer2_weights = layer2_weights - learning_rate * (
            (1 / len(training_set)) * layer2_df_W + const_lambda * layer2_weights)

    print(layer3_activation)
    # print(layer1_weights)
    # print(layer2_weights)
    print()
    print("cost", cost / len(training_set), epoch)
    cost = 0
