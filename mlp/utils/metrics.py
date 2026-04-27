import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    prediction = np.argmax(y_pred, axis=1)
    return np.mean(prediction == y_true)

def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int=10) -> np.ndarray:
    prediction = np.argmax(y_pred, axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for t,p in zip(y_true, prediction):
        cm[t,p] += 1
        
    return cm
