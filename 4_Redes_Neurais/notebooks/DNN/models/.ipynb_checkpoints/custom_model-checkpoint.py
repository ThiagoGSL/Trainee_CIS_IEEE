from .base_model import BaseModel
from ..layers.layer import Dense
from ..layers.activation import ReLU, Sigmoid

class CustomModel(BaseModel):
    def __init__(self, layers_config):
        super().__init__()
        for layer in layers_config:
            self.add(layer)
            
            
    #IMPLEMENTAR OPTMIZER, TRAINING e EVALUATE aqui