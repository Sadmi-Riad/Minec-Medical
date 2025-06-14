from PyQt5.QtWidgets import QMessageBox
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalize_data(self):
    try:
        selected_columns = self.ui_helper.selected_preprocessing_columns
        df = self.df

        if df is None:
            QMessageBox.warning(self, "Warning", "No DataFrame loaded!")
            return "Error: No DataFrame loaded.", []

        if not selected_columns:
            QMessageBox.warning(self, "Warning", "No column selected for normalization!")
            return "Error: No column selected for normalization.", []

        processed_columns = []
        already_normalized = []
        non_numeric = []

        # 1) Classify columns
        for col in selected_columns:
            if np.issubdtype(df[col].dtype, np.number):
                col_min, col_max = df[col].min(), df[col].max()
                if col_min >= 0 and col_max <= 1:
                    already_normalized.append(col)
                else:
                    processed_columns.append(col)
            else:
                non_numeric.append(col)

        # 2) Apply scaling only to those that need it
        if processed_columns:
            scaler = MinMaxScaler()
            df[processed_columns] = scaler.fit_transform(df[processed_columns])
            df[processed_columns] = df[processed_columns].round(3)
            
            # Add to pipeline
            self.pipeline_manager.add_preprocessing_step(
                "minmax_normalization",
                scaler,
                processed_columns
            )
            self.pipeline_manager.feature_names = processed_columns

        # 3) Build the output message
        lines = []
        if processed_columns:
            lines.append(
                f"{len(processed_columns)} column(s) normalized and added to pipeline:\n"
                + "\n".join(processed_columns)
            )
        if already_normalized:
            lines.append(
                f"Skipped already normalized column(s):\n"
                + "\n".join(already_normalized)
            )
        if non_numeric:
            lines.append(
                f"Skipped non-numeric column(s):\n"
                + "\n".join(non_numeric)
            )

        output_txt = "\n\n".join(lines) if lines else "No columns were normalized."

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Normalization failed: {str(e)}")
        output_txt = f"Error during normalization: {str(e)}"
        processed_columns = []

    if hasattr(self.ui, 'outputs'):
        self.ui.outputs.clear()
        self.ui.outputs.append(output_txt)

    return output_txt, processed_columns