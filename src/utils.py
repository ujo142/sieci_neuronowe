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

def CrossEntropy(y_true, y_pred):
    epsilon = 1e-10
    loss = -np.mean(np.sum(y_pred * np.log(y_true + epsilon), axis=1))
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
    
def load_data(train_data_path, test_data_path):
    # load csv data
    # Zrobić to potem porządnie 
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
    X_te_reshaped = X_te.reshape(len(X_te), 1, 2)
    y_te_reshaped = y_te.reshape(len(y_te), 1, 1)
    return X_te_reshaped, y_te_reshaped, X_te_reshaped, y_te_reshaped