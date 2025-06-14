from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scripts.evaluation.save_model import apply_pre_on_supplied_simple
import pandas as pd
import os

def apply_linear_regression(self, cv, supplied, test_size ,params):
    try:
        # Get target and features
        target = self.ui.comboBox_Att_Estim.currentText()
        
        self.is_test_file = False
        
        if not target:
            raise ValueError("Please select a target variable")
                
        # Ensure target is numeric
        if not pd.api.types.is_numeric_dtype(self.df_filtred[target]):
            raise ValueError("Target variable must be numeric")
        
        # Select only numeric features
        X = self.df_filtred.select_dtypes(include=['int64', 'float64']).drop(columns=[target], errors='ignore')
        y = self.df_filtred[target]
                
        cv_used = False
        
        if not params : 
            model = LinearRegression()
        else : 
            model = LinearRegression(**params)
        
        if cv is not None:
            # Cross validation case
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            cv_used = True
            model.fit(X, y)
            score_mean = scores.mean()
            scores_str = ", ".join([f"{s:.4f}" for s in scores])
        elif supplied is not None:
            # Supplied test file case
            if not os.path.exists(supplied):
                raise FileNotFoundError(f"Test file not found at: {supplied}")
            test_df = pd.read_csv(supplied)
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
            X_test = test_df.drop(columns=[target], errors='ignore')
            y_test = test_df[target]
            model.fit(X, y)
            self.is_test_file = True
        elif test_size is not None:
            # Percentage split case
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
            model.fit(X_train, y_train)
        else:
            # Training set case
            model.fit(X, y)
            X_test = X
            y_test = y
        
        # Generate output text
        output_text = "==== Linear Regression Results ====\n\n"
        output_text += f"Target variable: {target}\n"
        output_text += f"Total samples: {X.shape[0]}\n"
        output_text += f"Number of features: {X.shape[1]}\n\n"
        
        if cv_used:
            output_text += "Validation method: Cross-validation\n"
            output_text += f"Number of folds: {cv}\n"
            output_text += f"Mean R² score: {score_mean:.4f}\n"
            output_text += f"Fold scores: {scores_str}\n"
        else:
            y_pred = model.predict(X_test)
            
            if supplied is not None:
                output_text += "Validation method: Supplied test set\n"
                output_text += summary
            elif test_size is not None:
                output_text += "Validation method: Percentage split\n"
                output_text += f"Test size: {test_size*100:.1f}%\n"
            else:
                output_text += "Validation method: Training set\n"
            
            output_text += f"Test samples: {X_test.shape[0]}\n\n"
            output_text += "Evaluation metrics:\n"
            output_text += f"   MAE: {mean_absolute_error(y_test, y_pred):.4f}\n"
            output_text += f"   MSE: {mean_squared_error(y_test, y_pred):.4f}\n"
            output_text += f"   R²: {r2_score(y_test, y_pred):.4f}\n"
        
        # Handle predictions
        if cv_used:
            new_df = self.df_filtred.copy()
            new_df[f"Predicted_{target}"] = model.predict(X)
        elif supplied is None:
            new_df = self.df_filtred.copy()
            new_df.loc[X_test.index, f"Predicted_{target}"] = y_pred
        else:
            test_df[f"Predicted_{target}"] = y_pred
            new_df = test_df
        
        return output_text, model, new_df
        
    except Exception as e:
        print(f"Error in linear regression: {str(e)}")  # Debug print
        raise  # Re-raise the exception with original stack trace