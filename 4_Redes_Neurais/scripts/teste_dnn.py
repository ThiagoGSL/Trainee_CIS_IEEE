from DNN.models.custom_model import CustomModel
from DNN.layers.layer import Dense
from DNN.layers.activation import ReLU, Sigmoid
from DNN.training.loss import CrossEntropyLoss
from DNN.training.optimizer import GradientDescent
from DNN.training.trainer import Trainer



#Initializing model, loss function and optimizer 
model = CustomModel([
    Dense(784, X.shape[0]),
    Relu(),
    Dense(128,2),
    Sigmoid()])

loss_func = CrossEntropyLoss()
optimizer = GradientDescent(learning_rate=0.01)

trainer = Trainer(model, loss_func, optimizer)
trainer.train(X_train, y_train, epochs=10)