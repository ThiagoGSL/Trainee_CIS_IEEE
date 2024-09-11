import numpy as np

class Dense:
    def __init__(self, n_in, n_out): #n_prev: previous layer dim, n_next = next_layer_dim 
        self.W = 0.01 * np.random.randn(n_out, n_in) #Weights
        self.b = np.zeros((n_h, 1)) #Biases
    
    def forward(self, A): #Activation from previous layer -- A[0] = X
        self.A = A
        self.Z = np.dot(self.W,A.T)+self.b
        
    def backward(self, dZ):
        m = self.A.shape[0]
        self.dW = np.dot(dZ,self.A.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True)/m
        dA = np.dot(self.W.T,dZ)
        return dA
    
    def update_parameters(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db