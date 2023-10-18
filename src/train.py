import toml
from perceptron import perceptron_net
import utils
from layers import dense_layer, activation_layer



if __name__ == "__main__":
    
    with open('/Users/ben/python_projects/mini_pw/sieci_neuronowe/src/config/config2.toml', 'r') as file:
        config = toml.load(file)
        
    
    # training data
    X_train, y_train = utils.load_data(config['data']['path'])
   
    # network
    net = perceptron_net()
    net.add(dense_layer(2, 3))
    net.add(activation_layer(utils.relu, utils.relu_prime))
    net.add(dense_layer(3, 1))
    net.add(activation_layer(utils.sigmoid, utils.sigmoid_prime))

    # train
    net.use(utils.BinaryCrossEntropy, utils.BinaryCrossEntropy_prime)
    net.fit(X_train, y_train, epochs=1000, learning_rate=0.001)