from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        
        
        if y_prob.shape[1] == 2:  
            y_prob = y_prob[:, 1]
        else:  
            y_prob = y_prob  
    else:
        y_prob = None  
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1_score": f1_score(y_test, y_pred, average='macro')
    }
    
    if y_prob is not None:
        if y_prob.ndim == 1:  # binaire
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        else:  # multi-classes
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob, multi_class='ovr')
    
    
    return metrics , y_pred

def get_confusion_matrix(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 0:
            return np.zeros((2, 2), dtype=int)
        return cm
    except Exception as e:
        return np.zeros((2, 2), dtype=int)