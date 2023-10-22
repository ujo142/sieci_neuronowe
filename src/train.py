import toml
from perceptron import perceptron_net
import utils
from layers import dense_layer, activation_layer
import cProfile


if __name__ == "__main__":
    
    with open('src/config/config.toml', 'r') as file:
        config = toml.load(file)
        
    # training data
    X_train, y_train, X_test, y_test = utils.load_data(**config['data'])
    
    # network
    net = perceptron_net()
    net.add(dense_layer(2, 3)) 
    net.add(activation_layer(utils.relu, utils.relu_prime))
    net.add(dense_layer(3, 1))
    net.add(activation_layer(utils.sigmoid, utils.sigmoid))

    # train
    net.set_loss(utils.BinaryCrossEntropy, utils.BinaryCrossEntropy_prime)
    net.fit(X_train, y_train, X_test, y_test, epochs=500, learning_rate=0.0001)
    
    
    # Epoch 127, Loss: 0.032956917601535934, Test accuracy: 99.70%