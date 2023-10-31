import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from layers import dense_layer
import utils
from tqdm import tqdm


class perceptron_net:
    def __init__(self, objective, seed, **kwargs):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.objective = objective
        self.seed = seed
        
    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def set_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
    
    def test(self, x_test, y_test):
        # sample dimension first
        samples = len(x_test)
        outputs = []
        for j in range(samples):
            # forward propagation
            output = x_test[j]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            outputs.append(np.argmax(output[0]))
        if self.objective == 'binary_classification':
            self.num_correct = sum([1 if outputs[i] == y_test[i] else 0 for i in range(len(y_test))])
        elif self.objective == 'multi_classification':
            self.num_correct = sum([1 if np.argmax(outputs[i]) == np.argmax(y_test[i]) else 0 for i in range(len(y_test))])
        accuracy = self.num_correct / samples
        return accuracy, outputs
    
    def test_mse(self, x_test, y_test):
        # sample dimension first
        samples = len(x_test)
        outputs = []
        for j in range(samples):
            # forward propagation
            output = x_test[j]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            outputs.append(output[0])
        mse = self.loss(y_test, outputs)
        return mse, outputs
    
    # train the network
    def fit(self, x_train, y_train, x_test, y_test, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        result = {
            "Accuracy": [],
            "Train loss": []
        }
        outputs = None
        # training loop
        for i in range(epochs):
            if i == 14 or i == 24:
                learning_rate /= 10
            err = 0
            
            # shuffle samples in each epoch with certain seed
            np.random.seed(self.seed)
            perm = np.random.permutation(len(x_train))
            x_train = x_train[perm]
            y_train = y_train[perm]
            print(f"Epoch: {i + 1}")
            for j in tqdm(range(samples)):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output[0])

                
                # backward propagation
                error = self.loss_prime(y_train[j], output[0])     # err - int, error - 1,3. Czy to na pewno dobrze?
                for layer in reversed(self.layers): 
                    error = layer.backward_propagation(error, learning_rate)
                    
            err /= samples
             # calculate average error on all samples
            if self.objective == 'binary_classification' or self.objective == 'multi_classification':
                metric, outputs = self.test(x_test, y_test)
                print(f"Epoch {i+1}, Loss: {err}, Test accuracy: {metric * 100:.2f}%")
            else: 
                metric, outputs = self.test_mse(x_test, y_test)
                print(f"Epoch {i+1}, train_mse: {err}, train_rmse: {np.sqrt(err)}, test_mse: {metric}, test_rmse {np.sqrt(metric)}")

            result["Accuracy"].append(metric)
            result["Train loss"].append(err)
            
        return result, outputs

    def draw_losses(self, ax):
        G = nx.Graph()
        i = 0
        for idx, layer in enumerate(self.layers):
            if not isinstance(layer, dense_layer):
                continue
            first_layer, second_layer = layer.weights.shape
            edges = []
            for j in range(first_layer):
                for k in range(second_layer):
                    edges.append((i + j, i + first_layer + k, layer.weights_errors[j][k]))
            i += first_layer
            first_nodes = [edge[0] for edge in edges]
            second_nodes = [edge[1] for edge in edges]
            G.add_weighted_edges_from(edges)
            for node in G.nodes:
                if node in first_nodes:
                    G.nodes[node]["subset"] = idx
                elif node in second_nodes:
                    G.nodes[node]["subset"] = idx + 1
        pos = nx.multipartite_layout(G)
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        nx.draw(G, pos=pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Reds, with_labels=True, ax=ax)
        labels = {e: f"{G.edges[e]['weight']:.2f}" for e in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, label_pos=0.75)
        ax.set_title("Propagated_losses")

    def draw_network_weights(self, ax):
        G = nx.Graph()
        i = 0
        for idx, layer in enumerate(self.layers):
            if not isinstance(layer, dense_layer):
                continue
            first_layer, second_layer = layer.weights.shape
            edges = []

            for j in range(first_layer):
                for k in range(second_layer):
                    edges.append([i + j, i + first_layer + k, layer.weights[j][k]])
            i += first_layer
            first_nodes = [edge[0] for edge in edges]
            second_nodes = [edge[1] for edge in edges]

            G.add_weighted_edges_from(edges)
            for node in G.nodes:
                if node in first_nodes:
                    G.nodes[node]["subset"] = idx
                elif node in second_nodes:
                    G.nodes[node]["subset"] = idx + 1
        pos = nx.multipartite_layout(G)
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        nx.draw(G,pos=pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Reds, with_labels=True, ax=ax)
        labels = {e: f"{G.edges[e]['weight']:.2f}" for e in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, label_pos=0.75)
        ax.set_title("Network weights")
