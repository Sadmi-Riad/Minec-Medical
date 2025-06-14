from sklearn.linear_model import LogisticRegression
from scripts.preprocessing.data_loader import get_target
from scripts.validation.validation import *
from scripts.evaluation.model_evaluation import evaluate_model , get_confusion_matrix
from scripts.evaluation.save_model import apply_pre_on_supplied_simple
from scripts.visualization.logistic_equation import logistic_regression_equation
import pandas as pd
import os


def apply_logistic_regression(self, cv, supplied, test_size ,params):
    y, X = get_target(self , 1)
    self.is_test_file = False #Flag
    
    cv_used = False  
    
    if not params : 
        model = LogisticRegression(max_iter=500)
    else :
        model = LogisticRegression(**params)
    
    if cv is not None:
        model, score_mean, scores_str = cross_val(function=LogisticRegression, cv=cv, X=X, y=y ,params=params)
        cv_used = True
        model.fit(X, y)
    elif supplied is not None:
        test_df=pd.read_csv(supplied)
        if self.pipeline_manager.has_preprocessing_steps():
            text , test_df = apply_pre_on_supplied_simple(self.pipeline_manager , test_df,self.df)
            test_df = test_df[self.df_filtred.columns]
            summary = "Preprocess of Supplied File : \n" + text
        else : 
            test_df = test_df[self.df_filtred.columns]
            summary="No Preprocess of the Supplied File \n"
        numeric_cols = [col for col in self.df_filtred.columns if pd.api.types.is_numeric_dtype(self.df_filtred[col])]
        numeric_test_cols = [col for col in numeric_cols if col in test_df.columns]
        test_df=test_df[numeric_test_cols]
        X_train, y_train, X_test, y_test = supplied_test_file(X=X, y=y, test_df=test_df , outcome=self.outcome_column)
        model.fit(X_train, y_train)
        self.is_test_file = True
    elif test_size is not None:
        X_train, y_train, X_test, y_test = pourcentage_split(X=X, y=y, test_size=test_size)
        model.fit(X_train, y_train)
    else:
        model.fit(X, y)
        X_train = X
        y_train = y
        X_test = X
        y_test = y
    
    output_text = "==== Logistic Regression Results ====\n\n"
    
    output_text += f"Total number of samples: {X.shape[0]}\n"
    output_text += f"Number of features: {X.shape[1]}\n\n"
    
    if not cv_used:
        if supplied is not None:
            output_text += "Validation method used: Supplied test file\n"
            output_text += summary + "\n"
        elif test_size is not None:
            output_text += f"Validation method used: Percentage split (test size = {test_size})\n"
        output_text += f"Number of test set samples: {X_test.shape[0]}\n"
        output_text += "Model evaluation on the test set:\n"
        metrics, y_pred = evaluate_model(model, X_test=X_test, y_test=y_test)
        for metric, value in metrics.items():
            output_text += f"   {metric.capitalize()}: {value:.4f}\n"
        matrix = get_confusion_matrix(model, X_test, y_test)  # confusion matrix
        if supplied is None: 
            new_df = self.df.copy()
            new_df.loc[X_test.index, "Outcome_Predicted"] = y_pred 
        else: 
            test_df.loc[X_test.index, "Outcome_Predicted"] = y_pred 
            new_df = test_df
    else:
        output_text += "Validation method used: Cross-validation\n"
        output_text += f"Number of folds: {cv}\n"
        output_text += f"Mean score: {score_mean:.4f}\n"
        output_text += f"Scores per fold: {scores_str}\n"
        matrix = get_confusion_matrix(model, X, y)  # confusion matrix
        metrics, y_pred = evaluate_model(model, X, y)
        new_df = self.df.copy()
        new_df["Outcome_Predicted"] = y_pred  
    
    output_text += "\n==== End of Results ====\n"
    feature_names = X.columns.tolist()
    equation = logistic_regression_equation(model, feature_names)

    return output_text,model, equation, matrix, new_df