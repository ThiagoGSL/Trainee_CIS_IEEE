import numpy as np

class Dense:
    def __init__(self, n_in, n_out, initialization = 'He', lambd = 0): 
        if initialization == None:
            self.W = 0.01 * np.random.randn(n_out, n_in) #random weights initialization
        elif initialization == 'He':
            self.W = np.random.randn(n_out, n_in) * np.sqrt(2. / n_in) # He Initialization
        self.b = np.zeros((n_out, 1)) #Biases
    
    def forward(self, A): #Activation from previous layer -- A[0] = X
        self.A = A
        self.Z = np.dot(self.W,A)+self.b
        return self.Z
        
    def backward(self, dZ): #IMPLEMENTAR REGULARIZATION
        m = self.A.shape[1]
        self.dW = np.dot(dZ, self.A.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True)/m
        dA = np.dot(self.W.T,dZ)
        return dA
    
    def update_parameters(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db