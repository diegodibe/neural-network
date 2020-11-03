# Creation of a ANN with backpropagation
import numpy as np  # see if it's possible to use, otherwise import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_palette(sns.color_palette("husl", 15))


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


# def gradient_check


training_set = []
for i in range(8):
    training_set.append((create_training_sample(i), create_training_sample(i)))
# TODO initialize near zero (0, 0.01^2)
layer1_weights = np.random.random_sample((9, 3)) * 0.1
layer2_weights = np.random.random_sample((4, 8)) * 0.1
layer1_activation = np.zeros(9).reshape(-1, 1)
layer2_activation = np.zeros(4).reshape(-1, 1)
layer3_activation = np.zeros(8).reshape(-1, 1)
layer1_activation[0] = 1
layer2_activation[0] = 1

# print(layer1_weights)
# print(layer2_weights)

# learning rate and weight decay
learning_rates = [71]
weight_decay = 0.00001
training_epochs = 10000
cost_iter = pd.DataFrame(columns=['epoch', 'cost', 'learning rate', 'weight decay'])
print(cost_iter)

for a in learning_rates:
    for epoch in range(training_epochs):
        cost = 0
        layer1_df_WJxy = np.zeros((9, 3))
        layer2_df_WJxy = np.zeros((4, 8))
        batch = np.random.permutation(training_set)
        for m in batch:
            layer1_activation[1:, :] = m[0].reshape(-1, 1)

            layer2_activation[1:, :] = sigmoid(np.dot(layer1_activation.T, layer1_weights)).reshape(-1, 1)

            layer3_activation = sigmoid(np.dot(layer2_activation.T, layer2_weights)).reshape(-1, 1)

            if epoch == 9999:
                print(m[0])
                print(layer3_activation.T)
                print()
            # print("layer3", layer3_activation)

            cost += np.sum(cost_fct3(layer3_activation.T, m[1]))

            delta_3 = np.multiply(derivative(layer3_activation.T), layer3_activation.T - m[1])

            delta_2 = np.multiply(derivative(layer2_activation.T), np.dot(delta_3, layer2_weights.T))

            layer2_df_WJxy += np.dot(layer2_activation, delta_3)
            layer1_df_WJxy += np.dot(layer1_activation, delta_2[:, 1:])

            # print("layer2", layer2_df_W)
            # print("layer1", layer1_df_W)

        m = len(training_set)
        layer1_df_WJ = np.zeros((9, 3))
        layer2_df_WJ = np.zeros((4, 8))
        layer1_df_WJ[0, :] = (1 / m) * layer1_df_WJxy[0, :]
        layer2_df_WJ[0, :] = (1 / m) * layer2_df_WJxy[0, :]
        layer1_df_WJ[1:, :] = ((1 / m) * layer1_df_WJxy[1:, :]) + (weight_decay * layer1_weights[1:, :])
        layer2_df_WJ[1:, :] = ((1 / m) * layer2_df_WJxy[1:, :]) + (weight_decay * layer2_weights[1:, :])

        layer1_weights[0, :] -= a * ((1 / m) * layer1_df_WJxy[0, :])
        layer2_weights[0, :] -= a * ((1 / m) * layer2_df_WJxy[0, :])
        layer1_weights[1:, :] -= a * (((1 / m) * layer1_df_WJxy[1:, :]) + (weight_decay * layer1_weights[1:, :]))
        layer2_weights[1:, :] -= a * (((1 / m) * layer2_df_WJxy[1:, :]) + (weight_decay * layer2_weights[1:, :]))

        cost_Wb = (cost / m) + (weight_decay / 2 * (np.sum(pow(layer1_weights, 2)) + np.sum(pow(layer2_weights, 2))))
        print("Total Cost", cost_Wb, epoch)
        cost_iter = cost_iter.append({'epoch':epoch, 'cost': cost_Wb, 'learning rate': a, 'weight decay': weight_decay}, ignore_index=True)

    sns.heatmap(layer1_weights, linewidth=0.5)
    plt.title(f"Heat map of weights input to hidden layer with learning rate {a}")
    plt.show()

    sns.heatmap(layer2_weights, linewidth=0.5)
    plt.title(f"Heat map of weights hidden to output layer with learning rate {a}")
    plt.show()


print(cost_iter)
sns.lineplot(data=cost_iter, x='epoch', y='cost', hue='learning rate')
plt.xlabel("Epochs")
plt.ylabel("Cost function")
plt.title(f"Progression of the cost function over {training_epochs} epochs and weight decay {weight_decay}")
plt.show()



