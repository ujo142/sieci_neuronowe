import networkx as nx
import toml
import utils
from perceptron import perceptron_net
from src.layers import dense_layer, activation_layer
import matplotlib.pyplot as plt


def draw_network(network: perceptron_net):
    G = nx.Graph()
    i = 0
    for idx, layer in enumerate(network.layers):
        if not isinstance(layer, dense_layer):
            continue
        first_layer, second_layer = layer.weights.shape
        edges = []
        for j in range(first_layer):
            for k in range(second_layer):
                edges.append((i + j, i + first_layer + k, layer.weights[j][k]))
        i += len(edges)
        first_nodes = [edge[0] for edge in edges]
        second_nodes = [edge[1] for edge in edges]
        G.add_weighted_edges_from(edges)
        for node in G.nodes:
            if node in first_nodes:
                G.nodes[node]["subset"] = idx
            elif node in second_nodes:
                G.nodes[node]["subset"] = idx+1
    pos = nx.multipartite_layout(G)
    nx.draw_networkx(G, pos=pos)
    plt.show()


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
    net.add(dense_layer(3, 1))
    net.add(activation_layer(utils.softmax, utils.softmax_prime))
    draw_network(net)