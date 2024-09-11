class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'update_parameters'):
                layer.update_parameters(self.learning_rate)