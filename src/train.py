import pandas as pd
import toml
from perceptron import perceptron_net
import utils
from layers import dense_layer, activation_layer
import cProfile
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gui
import pickle

if __name__ == "__main__":
    
    with open('src/config/config.toml', 'r') as file:
        config = toml.load(file)
        
    # Prepare data
    X_train, y_train, X_test, y_test = utils.load_data(**config)
    #X_train = utils.load_mnist()
    if config['objective'] == 'multi_classification':
        encoder = OneHotEncoder(sparse_output=False)
        y_train = encoder.fit_transform(np.array(y_train).reshape(-1, 1))
        y_test = encoder.fit_transform(np.array(y_test).reshape(-1, 1))
    
    if config['activation_fn'] == 'relu':
        activation_fn = utils.relu
        activation_fn_prime = utils.relu_prime
    elif config['activation_fn'] == 'leaky_relu':
        activation_fn = utils.leaky_relu
        activation_fn_prime = utils.leaky_relu_prime
    elif config['activation_fn'] == 'tanh':
        activation_fn = utils.tanh
        activation_fn_prime = utils.tanh_prime
    elif config['activation_fn'] == 'linear':
        activation_fn = utils.linear
        activation_fn_prime = utils.linear_prime
    elif config['activation_fn'] == 'sigmoid':
        activation_fn = utils.sigmoid
        activation_fn_prime = utils.sigmoid_prime
        
    if config['last_activation_fn'] == 'sigmoid':
        last_activation_fn = utils.sigmoid
        last_activation_fn_prime = utils.sigmoid_prime
    elif config['last_activation_fn'] == 'softmax':
        last_activation_fn = utils.softmax
        last_activation_fn_prime = utils.softmax_prime
    elif config['last_activation_fn'] == 'tanh':
        last_activation_fn = utils.tanh
        last_activation_fn_prime = utils.tanh_prime
    elif config['last_activation_fn'] == 'linear':
        last_activation_fn = utils.linear
        last_activation_fn_prime = utils.linear_prime
        
    if config['loss_function'] == 'categorical_crossentropy':
        loss = utils.CrossEntropy
        loss_prime = utils.CrossEntropy_prime
    elif config['loss_function'] == 'binary_crossentropy':
        loss = utils.BinaryCrossEntropy
        loss_prime = utils.BinaryCrossEntropy_prime
    elif config['loss_function'] == 'BinaryCrossEntropy':
        loss = utils.BinaryCrossEntropy
        loss_prime = utils.BinaryCrossEntropy_prime
    elif config['loss_function'] == 'mse':
        loss = utils.mse
        loss_prime = utils.mse_prime
    
    # network
    net = perceptron_net(config['objective'], config['seed'])
    np.random.seed(config['seed'])
    
    # set layers
    for i in range(len(config['neurons_per_layer']) - 1):
        net.add(dense_layer(config['neurons_per_layer'][i], config['neurons_per_layer'][i + 1], config['seed']))
        if i < len(config['neurons_per_layer']) - 2:
            net.add(activation_layer(activation_fn, activation_fn_prime))
        else:
            net.add(activation_layer(last_activation_fn, last_activation_fn_prime))
    
    # set loss
    net.set_loss(loss, loss_prime)

    result = net.fit(X_train, y_train, X_test, y_test, epochs=config['epochs'], learning_rate=config['learning_rate'])
    pd.DataFrame(result).to_csv(f"last.csv")
    with open("test.pkl", 'wb') as f:
        pickle.dump(net, f)
    # gui.run_gui(config['epochs'], figs)