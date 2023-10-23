import toml
from perceptron import perceptron_net
import utils
from layers import dense_layer, activation_layer
import cProfile
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
if __name__ == "__main__":
    
    with open('src/config/config.toml', 'r') as file:
        config = toml.load(file)
        
    # training data
    X_train, y_train, X_test, y_test = utils.load_data(**config['data'])
    
    encoder = OneHotEncoder(sparse=False)
    y_train_enc = encoder.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test_enc = encoder.fit_transform(np.array(y_test).reshape(-1, 1))
    
    # network
    net = perceptron_net()
    net.add(dense_layer(2, 6)) 
    net.add(activation_layer(utils.relu, utils.relu_prime))
    net.add(dense_layer(6, 6))
    net.add(activation_layer(utils.relu, utils.relu_prime))
    net.add(dense_layer(6, 3))
    net.add(activation_layer(utils.softmax, utils.softmax_prime))

    # train
    net.set_loss(utils.CrossEntropy, utils.CrossEntropy_prime)
    net.fit(X_train, y_train_enc, X_test, y_test_enc, epochs=500, learning_rate=0.001)
    
    
    # Epoch 127, Loss: 0.032956917601535934, Test accuracy: 99.70%