import numpy as np
from ..models.custom_model import CustomModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, X_test, y_test, show=False):
    # Forward pass para obter as previsões
    y_pred_prob = model.forward(X_test.T)

    '''
    if y_pred_prob.shape[0]>1 or y_pred_prob.shape[1]>1: #Verificando se o modelo é multiclasse
        y_pred = np.argmax(y_pred_prob, axis=0) #O formato da matriz y_pred_prob esperado é (n_y,m)
        y_true = np.argmax(y_test, axis=0) 
    
    else:
    '''
    y_pred = np.squeeze((y_pred_prob>0.5).astype(int))
    y_true = y_test 

    # Calculo da acurácia
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculo de métricas adicionais
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')


    if show:
        # Mostrando métricas
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

    return accuracy, precision, recall, f1