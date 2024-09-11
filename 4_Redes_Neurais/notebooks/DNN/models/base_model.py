class BaseModel:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            
    def update_parameters(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'update_parameters'):
                layer.update_parameters(learning_rate)
                