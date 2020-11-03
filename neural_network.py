# Creation of a ANN with backpropagation
import numpy as np  # see if it's possible to use, otherwise import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

layer1_weights_final = np.random.random_sample((9, 3)) * 0.01
layer2_weights_final = np.random.random_sample((4, 8)) * 0.01
cost_best = 5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative(x):
    return x * (1 - x)


def cost_fct(x, y):
    return 0.5 * (np.power((x - y), 2))


def create_training_sample(x):
    training_sample = np.zeros(8)
    if x > 8:
        raise Exception("training sample cannot be created")
    training_sample[x] = 1
    return training_sample


def train_network(training_epochs, a, weight_decay, layer1_weights, layer2_weights, training_set, threshold):
    global cost_best, layer2_weights_final, layer1_weights_final
    cost_iter = pd.DataFrame(columns=['epoch', 'cost', 'learning rate', 'weight decay'])

    layer1_activation = np.zeros(9).reshape(-1, 1)
    layer2_activation = np.zeros(4).reshape(-1, 1)
    layer3_activation = np.zeros(8).reshape(-1, 1)
    layer1_activation[0] = 1
    layer2_activation[0] = 1

    for epoch in range(training_epochs):
        cost = 0
        layer1_df_WJxy = np.zeros((9, 3))
        layer2_df_WJxy = np.zeros((4, 8))
        for m in training_set:
            layer1_activation[1:, :] = m[0].reshape(-1, 1)

            layer2_activation[1:, :] = sigmoid(np.dot(layer1_activation.T, layer1_weights)).reshape(-1, 1)

            layer3_activation = sigmoid(np.dot(layer2_activation.T, layer2_weights)).reshape(-1, 1)

            cost += np.sum(cost_fct(layer3_activation.T, m[1]))

            delta_3 = np.multiply(derivative(layer3_activation.T), layer3_activation.T - m[1])

            delta_2 = np.multiply(derivative(layer2_activation.T), np.dot(delta_3, layer2_weights.T))

            layer2_df_WJxy += np.dot(layer2_activation, delta_3)
            layer1_df_WJxy += np.dot(layer1_activation, delta_2[:, 1:])


        m = len(training_set)

        layer1_weights[0, :] -= a * ((1 / m) * layer1_df_WJxy[0, :])
        layer2_weights[0, :] -= a * ((1 / m) * layer2_df_WJxy[0, :])
        layer1_weights[1:, :] -= a * (((1 / m) * layer1_df_WJxy[1:, :]) + (weight_decay * layer1_weights[1:, :]))
        layer2_weights[1:, :] -= a * (((1 / m) * layer2_df_WJxy[1:, :]) + (weight_decay * layer2_weights[1:, :]))

        cost_Wb = (cost / m) + (weight_decay / 2 * (np.sum(pow(layer1_weights, 2)) + np.sum(pow(layer2_weights, 2))))
        cost_iter = cost_iter.append(
            {'epoch': epoch, 'cost': cost_Wb, 'learning rate': a, 'weight decay': weight_decay}, ignore_index=True)

        if cost_Wb < cost_best:
            cost_best = cost_Wb
            layer1_weights_final = layer1_weights
            layer2_weights_final = layer2_weights

        if cost_Wb < threshold:
            print(f"Cost {cost_Wb} reached after {epoch} epochs")
            break

    return cost_iter


def main():
    training_set = []
    for i in range(8):
        training_set.append((create_training_sample(i), create_training_sample(i)))

    # to be set, learning rate and weight decay
    learning_rates = [5.00]
    weight_decay = 0.00001
    training_epochs = 10000
    threshold = 0
    cost_iter = pd.DataFrame(columns=['epoch', 'cost', 'learning rate', 'weight decay'])

    layer1_weights_rad = np.random.random_sample((9, 3)) * 0.01
    layer2_weights_rad = np.random.random_sample((4, 8)) * 0.01

    for a in learning_rates:
        layer1_weights = layer1_weights_rad.copy()
        layer2_weights = layer2_weights_rad.copy()
        cost_iter = cost_iter.append(
            train_network(training_epochs, a, weight_decay, layer1_weights, layer2_weights, training_set, threshold))

    # cost_iter.to_csv("statistics.csv")
    colors = len(cost_iter['learning rate'].unique())
    sns.lineplot(data=cost_iter, x='epoch', y='cost', hue='learning rate', palette=sns.color_palette("husl", colors))
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title(f"Progression of the cost with {training_epochs} epochs and weight decay {weight_decay}")
    plt.show()

    layer1_output = np.zeros(9).reshape(-1, 1)
    layer2_output = np.zeros(4).reshape(-1, 1)
    layer3_output = np.zeros(8).reshape(-1, 1)
    layer1_output[0] = 1
    layer2_output[0] = 1
    for m in training_set:
        layer1_output[1:, :] = m[0].reshape(-1, 1)
        layer2_output[1:, :] = sigmoid(np.dot(layer1_output.T, layer1_weights_final)).reshape(-1, 1)
        layer3_output = sigmoid(np.dot(layer2_output.T, layer2_weights_final)).reshape(-1, 1)

        print("Input:\n", m[0], "\nHidden layer activation\n", np.round(np.reshape(layer2_output, (1, 4)), 2),
              "\nNeural Network output:\n", np.round(np.reshape(layer3_output, (1, 8)), 2), "\n")

    sns.set_palette(sns.color_palette("husl", 15))
    sns.heatmap(layer1_weights_final, linewidth=0.5)
    plt.title(f"Heat map of weights from input layer to hidden layer with learning rate {a}")
    plt.ylabel("Input layer")
    plt.xlabel("Hidden layer")
    plt.show()

    sns.set_palette(sns.color_palette("husl", 15))
    sns.heatmap(layer2_weights_final, linewidth=0.5)
    plt.title(f"Heat map of weights hidden to output layer with learning rate {a}")
    plt.ylabel("Hidden layer")
    plt.xlabel("Output layer")
    plt.show()


if __name__ == "__main__":
    main()
