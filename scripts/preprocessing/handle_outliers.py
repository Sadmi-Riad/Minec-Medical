import numpy as np
from sklearn.ensemble import IsolationForest
from PyQt5.QtWidgets import QMessageBox

def detect_outliers_iqr(main_window, selected_columns):
    try:
        if main_window.df is None:
            QMessageBox.warning(main_window, "Warning", "No dataframe loaded!")
            return "Error: No dataframe loaded", []
        
        df_numeric = main_window.df[selected_columns].select_dtypes(include=[np.number])
        if df_numeric.empty:
            QMessageBox.warning(main_window, "Warning", "No numerical column selected!")
            return "Error: No numerical column selected", []

        processed_columns = []
        total_outliers = 0
        output_text = "Outliers detected (IQR method):\n"
        bounds = {}  # To store bounds for each column
        
        for col in df_numeric.columns:
            Q1 = df_numeric[col].quantile(0.25)
            Q3 = df_numeric[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = ((main_window.df[col] < lower) | (main_window.df[col] > upper)).sum()
            total_outliers += outliers

            if outliers > 0:
                main_window.df[col] = np.where(main_window.df[col] < lower, lower, main_window.df[col])
                main_window.df[col] = np.where(main_window.df[col] > upper, upper, main_window.df[col])
                processed_columns.append(col)
                bounds[col] = {'lower': lower, 'upper': upper}  # Store bounds
                output_text += f"{col}: {outliers} values capped\n"
            else:
                output_text += f"{col}: No outliers found\n"

        # Add to pipeline
        if processed_columns:
            main_window.pipeline_manager.add_preprocessing_step(
                "iqr_outlier_handling",
                bounds,  # Store the bounds dictionary
                processed_columns
            )
            # Update feature names
            if main_window.pipeline_manager.feature_names:
                main_window.pipeline_manager.feature_names.extend(processed_columns)
            else:
                main_window.pipeline_manager.feature_names = processed_columns

        main_window.df[selected_columns] = main_window.df[selected_columns].round(2)
        output_text += f"\nTotal outliers processed: {total_outliers} values"
        
        return output_text, processed_columns

    except Exception as e:
        QMessageBox.critical(main_window, "Error", f"IQR outlier detection failed: {str(e)}")
        return f"Error during IQR outlier detection: {str(e)}", []

def detect_outliers_isolation_forest(main_window, selected_columns):
    try:
        if main_window.df is None:
            QMessageBox.warning(main_window, "Warning", "No dataframe loaded!")
            return "Error: No dataframe loaded", []
        
        df_numeric = main_window.df[selected_columns].select_dtypes(include=[np.number])
        if df_numeric.empty:
            QMessageBox.warning(main_window, "Warning", "No numerical column selected!")
            return "Error: No numerical column selected", []

        processed_columns = []
        total_outliers = 0
        output_text = "Outliers detected (Isolation Forest method):\n"
        models = {}  # To store models for each column
        
        for col in df_numeric.columns:
            values = main_window.df[[col]].dropna()
            if values.empty:
                output_text += f"{col}: No data available\n"
                continue

            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(values)

            outlier_mask = (preds == -1)
            nb_outliers = outlier_mask.sum()
            total_outliers += nb_outliers

            if nb_outliers > 0:
                median_value = main_window.df[col].median()
                main_window.df.loc[values.index[outlier_mask], col] = median_value
                processed_columns.append(col)
                models[col] = iso  # Store the trained model
                output_text += f"{col}: {nb_outliers} values replaced with median\n"
            else:
                output_text += f"{col}: No outliers found\n"

        # Add to pipeline
        if processed_columns:
            main_window.pipeline_manager.add_preprocessing_step(
                "isolation_forest_outlier_handling",
                models,  # Store the models dictionary
                processed_columns
            )
            # Update feature names
            if main_window.pipeline_manager.feature_names:
                main_window.pipeline_manager.feature_names.extend(processed_columns)
            else:
                main_window.pipeline_manager.feature_names = processed_columns

        main_window.df[selected_columns] = main_window.df[selected_columns].round(2)
        output_text += f"\nTotal outliers processed: {total_outliers} values"
        
        return output_text, processed_columns

    except Exception as e:
        QMessageBox.critical(main_window, "Error", f"Isolation Forest failed: {str(e)}")
        return f"Error during Isolation Forest execution: {str(e)}", []