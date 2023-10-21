import os

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt



def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

def BinaryCrossEntropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return (y_pred - y_true) / (y_pred * (1.0 - y_pred))

def CrossEntropy(y_true, y_pred):
    epsilon = 1e-10
    loss = -np.mean(np.sum(y_pred * np.log(y_true + epsilon), axis=0))
    return loss

def CrossEntropy_prime(y_true, y_pred):
    epsilon = 1e-10
    return -np.mean(y_true / (y_pred + epsilon), axis=0)
    
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(values):
    exp_values = np.exp(values)
    exp_values_sum = np.sum(exp_values)
    return exp_values/exp_values_sum

def softmax_prime(x):
    return softmax(x)*(1-softmax(x))
    
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
    
def load_data(data_dir, objective, data_size, data_name):
    # load csv data
    # Zrobić to potem porządnie
    train_data_path = os.path.join(data_dir, objective, f"data.{data_name}.train.{data_size}.csv")
    test_data_path = os.path.join(data_dir, objective, f"data.{data_name}.test.{data_size}.csv")
    try:
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
    except:
        raise Exception("File not found")
    
    X_tr = train_data.iloc[:, :-1].values
    y_tr = train_data.iloc[:, -1].values
    y_tr = y_tr - 1
    
    X_te = test_data.iloc[:, :-1].values
    y_te = test_data.iloc[:, -1].values
    y_te = y_te - 1
    
    # reshape data to match batch size
    X_train_reshaped = np.expand_dims(X_tr, 1)
    Y_train_reshaped = np.expand_dims(y_tr, 1)
    X_te_reshaped = np.expand_dims(X_te, 1)
    y_te_reshaped = np.expand_dims(y_te, 1)
    return X_train_reshaped, Y_train_reshaped, X_te_reshaped, y_te_reshaped

def plot_dataset_classification(ds_X, ds_Y):
    X_squeezed = np.squeeze(ds_X)
    Y_squeezed = np.squeeze(ds_Y)
    x = X_squeezed[:, 0]
    y = X_squeezed[:, 1]
    colors = Y_squeezed
    plt.scatter(x, y, s=50, c=colors, alpha=0.8, cmap='viridis')
    plt.show()

def plot_dataset_regression(ds_X, ds_Y):
    x = np.squeeze(ds_X)
    y = np.squeeze(ds_Y)
    plt.scatter(x, y, s=2)
    plt.show()

