import toml
from perceptron import perceptron_net
import utils
from layers import dense_layer, activation_layer
import cProfile


if __name__ == "__main__":
    
    with open('config/config.toml', 'r') as file:
        config = toml.load(file)
        
    # training data
    X_train, y_train, X_test, y_test = utils.load_data(**config['data'])
    
    # network
    net = perceptron_net()
    net.add(dense_layer(2, 3))
    net.add(activation_layer(utils.relu, utils.relu_prime))
    net.add(dense_layer(3, 1))
    net.add(activation_layer(utils.softmax, utils.softmax_prime))

    # train
    net.set_loss(utils.CrossEntropy, utils.CrossEntropy_prime)
    net.fit(X_train, y_train, X_test, y_test, epochs=5, learning_rate=0.001)