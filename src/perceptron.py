import numpy as np 

class perceptron_net:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

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
            outputs.append(np.round(output[0]))
        self.num_correct = sum([1 if outputs[i] == y_test[i] else 0 for i in range(len(y_test))])
        accuracy = self.num_correct / samples
        return accuracy
    
    # train the network
    def fit(self, x_train, y_train, x_test, y_test, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        
        # training loop
        for i in range(epochs):
            err = 0
            outputs = []
            
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output[0])
                
                # append index of max output to outputs
                outputs.append(np.argmax(output))
                
                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers): 
                    error = layer.backward_propagation(error, learning_rate)
                
            # calculate average error on all samples
            test_accuracy = self.test(x_test, y_test)
            err /= samples
            print(f"Epoch {i+1}, Loss: {err}, Test accuracy: {test_accuracy * 100:.2f}%")