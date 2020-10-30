# Creation of a ANN with backpropagation
import numpy as np  # see if it's possible to use, otherwise import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


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
for i in range(1):
    training_set.append((create_training_sample(i), create_training_sample(i)))

# 8 nodes and 1 bias node to 3 nodes
layer1_weights = np.zeros(27)
# 3 nodes and 1 bias node to 8 nodes
layer2_weights = np.zeros(32)
layer1_activation = np.zeros(8)
layer2_activation = np.zeros(3)
layer3_activation = np.zeros(8)

# random weight, learning rate
# weight = [0.5]
learning_rate = 0.001

for epoch in range(1):
    for m in training_set:
        layer1_activation = m[0]
        # add bias node
        layer1_activation = np.insert(layer1_activation, 0, 1)
        print(layer1_activation)

        for i in range(layer2_activation.size):
            layer2_activation[i] = sigmoid(np.dot(layer1_activation, layer1_weights[9*i:9*(i+1)]))
        # add bias node
        layer2_activation = np.insert(layer2_activation, 0, 1)
        print(layer2_activation)
        for i in range(layer3_activation.size):
            layer3_activation[i] = sigmoid(np.dot(layer2_activation, layer2_weights[4*i:4*(i+1)]))
        print(layer3_activation)

        error_3 = np.dot(np.dot(layer3_activation, 1-layer3_activation), (layer3_activation-m[1]))
        print(error_3)

        print(layer2_weights.T)
        print(error_3)
        print(layer2_activation)
        for i in range(layer2_activation.size):
            error_2 = np.dot(np.dot(layer2_weights[4*i:4*(i+1)], error_3), np.dot(layer2_activation, 1-layer2_activation))

        # # forward propagation
        # prediction_out = sigmoid(prediction_in)
        # print(prediction_out)
        #
        # # backpropagation
        # # error calculation
        # error = (prediction_out - output_values).sum()
        # print(error)
        #
        # # derivative
        # delta = error * sigmoid_derivative(prediction_out)
        #
        # # updataing weight
        # weight -= learning_rate * np.dot(measurement.T, delta)

