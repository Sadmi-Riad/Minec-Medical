import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

def cross_val(function , cv , X , y, params): 
    if not params : 
        model = function()
    else : 
        model = function(**params)
    scores = cross_val_score(model, X, y, cv=cv)
    score_mean = np.mean(scores)
    scores_str = ", ".join([f"{s:.4f}" for s in scores])
    
    return model , score_mean , scores_str

def supplied_test_file(X , y , test_df , outcome): 
    y_test = test_df[outcome]
    X_test = test_df.drop(columns=[outcome])
    X_train, y_train = X, y 
    
    return X_train , y_train , X_test , y_test

def pourcentage_split(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42,shuffle=True)
    return X_train, y_train, X_test, y_test


