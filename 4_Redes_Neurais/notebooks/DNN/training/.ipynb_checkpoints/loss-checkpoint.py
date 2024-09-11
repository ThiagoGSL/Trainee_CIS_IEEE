import numpy as np

class MeanSquaredError:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.squeeze(np.mean(np.power(y_true - y_pred, 2)))

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.squeeze(-np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9)))

    def backward(self):
        return - (np.divide(self.y_true, self.y_pred + 1e-9) - np.divide(1 - self.y_true, 1 - self.y_pred + 1e-9))
    
class CategoricalCrossEntropyLoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))