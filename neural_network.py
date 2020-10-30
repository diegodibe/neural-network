# Creation of a ANN with backpropagation
import numpy as np  # see if it's possible to use, otherwise import math


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


input_values = np.array([[1, 0, 0]])
input_values = input_values.reshape(input_values.size, 1)
output_values = input_values


# random weight, learning rate
weight = [0.5]
learning_rate = 0.05

# adding bias to input
input_values = np.insert(input_values, 0, 1).reshape(input_values.size + 1, 1)

measurement = input_values
for epoch in range(5000):
    prediction_in = np.dot(input_values, weight)
    print(prediction_in)

    # forward calculation
    prediction_out = sigmoid(prediction_in)
    print(prediction_out)

    # backpropagation
    # error calculation
    error = (prediction_out - output_values).sum()
    print(error)

    # derivative
    delta = error * sigmoid_derivative(prediction_out)

    # updataing weight
    weight -= learning_rate * np.dot(measurement.T, delta)
    


