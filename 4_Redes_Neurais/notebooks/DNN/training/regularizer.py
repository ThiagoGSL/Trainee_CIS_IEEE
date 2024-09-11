# regularizers.py
import numpy as np

class L2Regularizer:
    def __init__(self, lambd=0.01):
        self.lambd = lambd

    def compute_penalty(self, weights):
        return self.lambd * np.sum(np.square(weights))

    def gradient(self, weights):
        return 2 * self.lambd * weights


class L1Regularizer:
    def __init__(self, lambd=0.01):
        self.lambd = lambd

    def compute_penalty(self, weights):
        return self.lambd * np.sum(np.abs(weights))

    def gradient(self, weights):
        return self.lambd * np.sign(weights)