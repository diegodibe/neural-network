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
for i in range(6):
    training_set.append((create_training_sample(i), create_training_sample(i)))

# 8 nodes and 1 bias node to 3 nodes
layer1_weights = np.ones((9, 3))
# 3 nodes and 1 bias node to 8 nodes
layer2_weights = np.ones((4, 8))
layer1_activation = np.zeros(8)
layer2_activation = np.zeros(3)
layer3_activation = np.zeros(8)

# random weight, learning rate
# weight = [0.5]
learning_rate = 0.001

for epoch in range(50):
    for m in training_set:
        layer1_activation = m[0]
        # add bias node
        layer1_activation = np.insert(layer1_activation, 0, 1)
        print("layer1", layer1_activation)

        layer2_activation = np.vectorize(sigmoid)(np.dot(layer1_weights, layer1_activation))
        # add bias node
        layer2_activation = np.insert(layer2_activation, 0, 1)
        print("layer2", layer2_activation)
        layer3_activation = np.vectorize(sigmoid)(np.dot(layer2_weights, layer2_activation))
        print("layer3", layer3_activation)

        error_4 = np.dot(np.dot(layer3_activation, 1 - layer3_activation), (layer3_activation - m[1]))
        print("error 4", error_4)

        error_3 = np.dot(np.dot(layer2_weights, error_4), np.dot(layer2_activation, 1 - layer2_activation))
        print("error 3", error_3)

        error_2 = np.dot(np.dot(layer1_weights, np.delete(error_3, 0)),
                         np.dot(layer1_activation, 1 - layer1_activation))
        print("error 2", error_2)

        layer1_weights = layer1_weights + np.dot(layer1_activation, error_2)
        layer2_weights = layer2_weights + np.dot(layer2_activation, error_3)

        print(layer1_weights)
        print(layer2_weights)

        layer1_activation = np.delete(layer1_activation, 0)
        layer2_activation = np.delete(layer2_activation, 0)

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
