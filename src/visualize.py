import networkx as nx
import toml
import utils
from perceptron import perceptron_net
from layers import dense_layer, activation_layer
import matplotlib.pyplot as plt


if __name__ == "__main__":
    with open('config/config.toml', 'r') as file:
        config = toml.load(file)

    # training data
    X_train, y_train, X_test, y_test = utils.load_data(**config['data'])
    # if config['data']['objective'] == 'classification':
    #     utils.plot_dataset_classification(X_train, y_train)
    # else:
    #     utils.plot_dataset_regression(X_train, y_train)

    # network
    net = perceptron_net()
    net.add(dense_layer(2, 3))
    net.add(activation_layer(utils.relu, utils.relu_prime))
    net.add(dense_layer(3, 3))
    net.add(activation_layer(utils.relu, utils.relu_prime))
    net.add(dense_layer(3, 1))
    net.add(activation_layer(utils.relu, utils.relu_prime))
    # net.add(dense_layer(10, 1))
    # net.add(activation_layer(utils.softmax, utils.softmax_prime))
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    net.draw_network_weights(axs[0])
    net.draw_losses(axs[1])
    fig.show()