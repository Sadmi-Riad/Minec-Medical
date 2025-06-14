from PyQt5.QtWidgets import QInputDialog
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

def handle_imbalanced_data(self):
    if self.df_filtred is None:
        output_text = "Error: No DataFrame loaded."
        if hasattr(self.ui, 'outputs'):
            self.ui.outputs.clear()
            self.ui.outputs.append(output_text)
        return

    selected_columns = self.ui_helper.selected_preprocessing_columns
    if not selected_columns:
        output_text = "Error: No columns selected for SMOTE processing."
        if hasattr(self.ui, 'outputs'):
            self.ui.outputs.clear()
            self.ui.outputs.append(output_text)
        return

    colonnes_disponibles = selected_columns
    classe_colonne, ok = QInputDialog.getItem(
        self, "Select Target Column", "Choose the target column for balancing:", colonnes_disponibles, 0, False
    )
    if not ok or classe_colonne not in self.df_filtred.columns:
        output_text = "Error: Invalid or no target column selected."
        if hasattr(self.ui, 'outputs'):
            self.ui.outputs.clear()
            self.ui.outputs.append(output_text)
        return

    if not pd.api.types.is_numeric_dtype(self.df_filtred[classe_colonne]):
        self.df_filtred[classe_colonne] = LabelEncoder().fit_transform(self.df_filtred[classe_colonne])
        output_text = f"Target column '{classe_colonne}' encoded automatically."
        if hasattr(self.ui, 'outputs'):
            self.ui.outputs.clear()
            self.ui.outputs.append(output_text)

    features = [col for col in selected_columns if col != classe_colonne]
    X = self.df_filtred[features].copy()
    y = self.df_filtred[classe_colonne]

    counts = y.value_counts()
    if counts.nunique() == 1:
        output_text = (
            f"Dataset already balanced for target '{classe_colonne}' "
            f"({counts.iloc[0]} samples in each of {len(counts)} classes)."
        )
        if hasattr(self.ui, 'outputs'):
            self.ui.outputs.clear()
            self.ui.outputs.append(output_text)
        processed_columns = []
        return output_text, processed_columns

    non_numeric_cols = X.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        encoders = {}
        output_text = "Non-numeric columns detected and will be encoded automatically:\n" + ", ".join(non_numeric_cols)
        if hasattr(self.ui, 'outputs'):
            self.ui.outputs.clear()
            self.ui.outputs.append(output_text)
        for col in non_numeric_cols:
            le =LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
        self.pipeline_manager.add_preprocessing_step(
            "label_encoding",
            encoders, 
            non_numeric_cols
            )
            # Update feature names (append to existing ones if any)
        if self.pipeline_manager.feature_names:
            self.pipeline_manager.feature_names.extend(non_numeric_cols)
        else:
            self.pipeline_manager.feature_names = non_numeric_cols
    
    if X.select_dtypes(include=['object']).shape[1] > 0:
        output_text = "Error: Some features are still non-numeric after encoding. Cannot apply SMOTE."
        if hasattr(self.ui, 'outputs'):
            self.ui.outputs.clear()
            self.ui.outputs.append(output_text)
        return

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=features)
    df_resampled[classe_colonne] = y_resampled

    df_resampled[features] = df_resampled[features].round(3)

    self.df = df_resampled
    output_text = (
        f"SMOTE applied successfully.\n"
        f"Original samples: {len(X)}, Resampled samples: {len(X_resampled)}."
    )
    if hasattr(self.ui, 'outputs'):
        self.ui.outputs.clear()
        self.ui.outputs.append(output_text)

    processed_columns = list(self.df_filtred.columns)
    return output_text, processed_columns
