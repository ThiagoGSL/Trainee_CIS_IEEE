import numpy as np
from ..models.custom_model import CustomModel
from ..training.trainer import Trainer
from .evaluate import evaluate

class CrossValidation:
    def __init__(self, model, trainer):
        self.metrics = {'accuracy':[], 'precision':[], 'recall': [], 'f1': []}
        self.model = model
        self.trainer = trainer

    def validate(self, folds, epochs = 100, show= False):
        #if type(folds) == list:
        #   pass 
        for fold in folds: #Expecting a dictionary like: {'Fold_1': {'train': {'X': X_train, 'y': y_train},'test':  {'X': X_test, 'y': y_test}}}
            self.model.reinitialize()
            
            X_train = folds[fold]['train']['X']
            y_train = folds[fold]['train']['y']
            X_test = folds[fold]['test']['X']
            y_test = folds[fold]['test']['y']

            self.trainer.train(X_train.T, y_train, epochs=epochs, show = True, period_to_show = 10)
            accuracy, precision, recall, f1 = evaluate(self.model, X_test, y_test)

            self.metrics['accuracy'].append(accuracy)
            self.metrics['precision'].append(precision)
            self.metrics['recall'].append(recall)
            self.metrics['f1'].append(f1)

        avg_accuracy = np.mean(self.metrics['accuracy'])
        avg_precision = np.mean(self.metrics['precision'])
        avg_recall = np.mean(self.metrics['recall'])
        avg_f1 = np.mean(self.metrics['f1'])
        
        if show:
            # Mostrando métricas
            print(f"Accuracy: {avg_accuracy* 100:.2f}%")
            print(f"Precision: {avg_precision:.2f}")
            print(f"Recall: {avg_recall:.2f}")
            print(f"F1 Score: {avg_f1:.2f}")

        return avg_accuracy, avg_precision, avg_recall, avg_f1
        
            
            
            