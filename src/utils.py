import numpy as np
import pandas as pd



def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

def BinaryCrossEntropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return (y_pred - y_true) / (y_pred * (1.0 - y_pred))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
    
# loss function and its derivative
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
    
def load_data(path):
    # load csv data
    try:
        data = pd.read_csv(path)
    except:
        raise Exception("File not found")
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = y - 1
    
    # reshape data
    X_reshaped = X.reshape(len(X), 1, 2)
    y_reshaped = y.reshape(len(y), 1, 1)
    return X_reshaped, y_reshaped