import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'update_parameters'):
                layer.update_parameters(self.learning_rate)

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None

    def initialize(self, params):
        # Initialize the first moment vector (m) and second moment vector (v)
        self.m = [np.zeros_like(param) for param in params]
        self.v = [np.zeros_like(param) for param in params]

    def step(self, model):
        if self.m is None or self.v is None:
            self.initialize([layer.W for layer in model.layers if hasattr(layer, 'W')] +
                            [layer.b for layer in model.layers if hasattr(layer, 'b')])
        
        self.t += 1  # Increment time step

        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'W'):
                # Update for weights
                self._update_params(layer.W, layer.dW, i * 2)
                # Update for biases
                self._update_params(layer.b, layer.db, i * 2 + 1)

    def _update_params(self, param, grad, index):
        # Update biased first moment estimate
        self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * grad
        # Update biased second moment estimate
        self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * np.square(grad)

        # Compute bias-corrected first moment estimate
        m_hat = self.m[index] / (1 - np.power(self.beta1, self.t))
        # Compute bias-corrected second moment estimate
        v_hat = self.v[index] / (1 - np.power(self.beta2, self.t))

        # Update parameters
        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)