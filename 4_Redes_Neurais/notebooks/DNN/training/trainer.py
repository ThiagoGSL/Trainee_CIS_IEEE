class Trainer:
    def __init__(self, model, loss_func, optimizer):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metrics = {'accuracy':[]}#IMPLEMENTAR CAPTURA DE ACURÁCIA

    def train(self, X_train, y_train, epochs=100):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.model.forward(X_train)
            loss = self.loss_func.forward(y_pred, y_train)

            # Backward pass
            dA = self.loss_func.backward()
            self.model.backward(dA)

            # Update parameters
            self.optimizer.step(self.model)
            
            if epoch%50==0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}')
            