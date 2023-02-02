import numpy as np


class FlowerTester:
    def __init__(self):
        np.random.seed(1)
        self.weights1 = 2 * np.random.random((3, 3)) - 1
        self.weights2 = 2 * np.random.random((3, 3)) - 1
        self.weights3 = 2 * np.random.random((3, 3)) - 1
        self.weights4 = 2 * np.random.random((3, 1)) - 1
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def predict(self, inputs):
        #inputs = inputs.astype(float)
        hidden_layer1 = self.sigmoid(np.dot(inputs, self.weights1))
        hidden_layer2 = self.sigmoid(np.dot(hidden_layer1, self.weights2))
        hidden_layer3 = self.sigmoid(np.dot(hidden_layer2, self.weights3))
        output = self.sigmoid(np.dot(hidden_layer3, self.weights4))
        return output
    
    def train(self, inputs, labels, iterations):
        inputs = inputs.astype(float)
        labels = labels.astype(float)
        
        for i in range(iterations):
            hidden_layer1 = self.sigmoid(np.dot(inputs, self.weights1))
            hidden_layer2 = self.sigmoid(np.dot(hidden_layer1, self.weights2))
            hidden_layer3 = self.sigmoid(np.dot(hidden_layer2, self.weights3))
            output = self.sigmoid(np.dot(hidden_layer3, self.weights4))
            
            error = labels - output
            adjustments = error * self.sigmoid_derivative(output)
            self.weights4 += np.dot(hidden_layer3.T, adjustments)
            
            hidden_error3 = np.dot(adjustments, self.weights4.T)
            hidden_adjustments3 = hidden_error3 * self.sigmoid_derivative(hidden_layer3)
            self.weights3 += np.dot(hidden_layer2.T, hidden_adjustments3)
            
            hidden_error2 = np.dot(hidden_adjustments3, self.weights3.T)
            hidden_adjustments2 = hidden_error2 * self.sigmoid_derivative(hidden_layer2)
            self.weights2 += np.dot(hidden_layer1.T, hidden_adjustments2)
            
            hidden_error1 = np.dot(hidden_adjustments2, self.weights2.T)
            hidden_adjustments1 = hidden_error1 * self.sigmoid_derivative(hidden_layer1)
            self.weights1 += np.dot(inputs.T, hidden_adjustments1)

# nn = FlowerTester()
# inputs = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0]])
# labels = np.array([[1], [0], [0], [1], [0]])
# nn.train(inputs, labels, 10000)