from sklearn.impute import SimpleImputer , KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np

def handle_missing_data(self, strategy):

    if self.df is None:
        return "Error: No dataframe loaded", []

    selected_columns = self.ui_helper.selected_preprocessing_columns
    if not selected_columns:
        return "Error: No column selected for missing value handling.", []

    imputer_num = SimpleImputer(strategy=strategy)
    processed_columns = []  
    output_text = "Missing values processing:\n"
    numeric_cols = [col for col in selected_columns if self.df[col].dtype.kind in 'iufc']

    for col in numeric_cols:
        missing_before = self.df[col].isnull().sum()
        output_text += f"{col}: {missing_before} missing values\n"

        if missing_before > 0:
            self.df[col] = imputer_num.fit_transform(self.df[[col]]).round(2)
            processed_columns.append(col)

    if processed_columns:
        output_text += f"\n{len(processed_columns)} column(s) imputed using '{strategy}':\n" + "\n".join(processed_columns)
        # Ajout automatique au pipeline
        self.pipeline_manager.add_preprocessing_step(
            f"missing_values_{strategy}",
            imputer_num,
            processed_columns
        )
        self.pipeline_manager.feature_names = processed_columns
    else:
        output_text += f"\nNo missing values found to handle with strategy '{strategy}'."

    if hasattr(self.ui, 'outputs'):
        self.ui.outputs.clear()
        self.ui.outputs.append(output_text)

    return output_text, processed_columns


def handle_missing_data_knn(self):
    if self.df is None:
        return "Error: No dataframe loaded", []
    n_neighbors = max(1, int(np.sqrt(self.df.shape[0])))
    selected_columns = self.ui_helper.selected_preprocessing_columns
    if not selected_columns:
        return "Error: No column selected for missing value handling.", []
    
    numeric_cols = [col for col in selected_columns if self.df[col].dtype.kind in 'iufc']
    categorical_cols = [
        col for col in selected_columns
        if self.df[col].dtype == 'object' or str(self.df[col].dtype).startswith('category')
    ]
    if not numeric_cols and not categorical_cols:
        return "Error: No valid columns selected for KNN imputation.", []
    
    output_text = "KNN missing values processing:\n"
    all_cols = numeric_cols + categorical_cols
    missing_counts = self.df[all_cols].isnull().sum()
    for col in all_cols:
        output_text += f"{col}: {missing_counts[col]} missing values\n"

    # Only impute columns that actually have missing values
    cols_to_impute = [col for col in all_cols if missing_counts[col] > 0]
    cols_to_impute_num = [col for col in numeric_cols if col in cols_to_impute]
    cols_to_impute_cat = [col for col in categorical_cols if col in cols_to_impute]
    processed_columns = []

    if cols_to_impute:
        df_impute = self.df[cols_to_impute].copy()

        # Handle categorical: encode with OrdinalEncoder (preserving NaN)
        encoder = None
        if cols_to_impute_cat:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
            df_impute[cols_to_impute_cat] = encoder.fit_transform(df_impute[cols_to_impute_cat])

        # KNNImputer: works on all columns (numeric + encoded categorical)
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_array = imputer.fit_transform(df_impute)
        imputed_df = pd.DataFrame(imputed_array, columns=cols_to_impute, index=self.df.index)

        # Restore numeric columns (rounded if needed)
        for col in cols_to_impute_num:
            self.df[col] = imputed_df[col].round(2)
        # Decode categorical columns back to original labels
        if cols_to_impute_cat:
            imputed_df[cols_to_impute_cat] = imputed_df[cols_to_impute_cat].round(0)
            self.df[cols_to_impute_cat] = encoder.inverse_transform(imputed_df[cols_to_impute_cat])

        processed_columns = cols_to_impute_num + cols_to_impute_cat

        output_text += (
            f"\n{len(processed_columns)} column(s) imputed using KNN (n_neighbors={n_neighbors}):\n"
            + "\n".join(processed_columns)
        )
        self.pipeline_manager.add_preprocessing_step(
            f"missing_values_knn_{n_neighbors}",
            imputer,
            processed_columns
        )
        self.pipeline_manager.feature_names = processed_columns
    else:
        output_text += "\nNo missing values found to handle with KNN imputation."

    if hasattr(self.ui, 'outputs'):
        self.ui.outputs.clear()
        self.ui.outputs.append(output_text)

    return output_text, processed_columns