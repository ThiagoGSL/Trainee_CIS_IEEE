class Trainer:
    def __init__(self, model, loss_func, optimizer):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer

    def train(self, X_train, y_train, epochs=100, show=False, period_to_show = 50):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.model.forward(X_train)
            loss = self.loss_func.forward(y_pred, y_train)
    
            # Backward pass
            dA = self.loss_func.backward()
            self.model.backward(dA)
    
            # Update parameters
            self.optimizer.step(self.model)
            
            if epoch%period_to_show==0 and show:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}')