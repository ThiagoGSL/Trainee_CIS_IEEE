import numpy as np

class ReLU:
    def forward(self, X):
        self.A = np.maximum(0,X)
        return self.A
    
    def backward(self, dA):
        dZ = dA * (self.A>0).astype(float)
        return dZ
    
class Sigmoid:
    def forward(self, X):
        self.A = (1 / (1 + np.exp(-X)))
        return self.A
    
    def backward(self, dA):
        dZ = dA * (self.A *(1 - self.A))
        return dZ
    
class Softmax:
    pass

class Tanh:
    pass