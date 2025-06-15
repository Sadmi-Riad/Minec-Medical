from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from PyQt5.QtWidgets import QInputDialog
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMessageBox

def encode_label(self):
    try:
        columns = self.ui_helper.selected_preprocessing_columns
        df = self.df

        if df is None:
            QMessageBox.warning(self, "Warning", "No DataFrame loaded!")
            return "Error: No DataFrame loaded.", []

        if not columns:
            QMessageBox.warning(self, "Warning", "No column selected for encoding!")
            return "Error: No column selected for encoding.", []

        processed_columns = []
        already_numeric = []
        non_categorical = []
        skipped_empty = []
        encoders = {}  # To store encoders for each column

        for col in columns:
            dtype = df[col].dtype
            if df[col].dropna().empty:
                skipped_empty.append(col)
                continue

            if dtype == object or dtype.name == "category":
                le = LabelEncoder()
                # Create a mask for non-null values
                not_null_mask = df[col].notnull()
                # Encode only non-null values
                encoded = pd.Series(index=df.index, dtype='float64')
                encoded[not_null_mask] = le.fit_transform(df[col][not_null_mask])
                # Keep NaN for missing values
                df[col] = encoded
                processed_columns.append(col)
                encoders[col] = le  # Store the encoder

            elif np.issubdtype(dtype, np.number):
                already_numeric.append(col)

            else:
                non_categorical.append(col)

        # Add encoding step to pipeline
        if processed_columns:
            self.pipeline_manager.add_preprocessing_step(
                "label_encoding",
                encoders,  # Pass the dictionary of encoders
                processed_columns
            )
            # Update feature names (append to existing ones if any)
            if self.pipeline_manager.feature_names:
                self.pipeline_manager.feature_names.extend(processed_columns)
            else:
                self.pipeline_manager.feature_names = processed_columns
            self.boolean = True 
        # Build output message
        lines = []
        if processed_columns:
            lines.append(f"{len(processed_columns)} column(s) encoded and added to pipeline:\n{', '.join(processed_columns)}")
        if already_numeric:
            lines.append(f"Skipped already numeric column(s):\n{', '.join(already_numeric)}")
        if non_categorical:
            lines.append(f"Skipped non-categorical column(s):\n{', '.join(non_categorical)}")
        if skipped_empty:
            lines.append(f"Skipped empty column(s):\n{', '.join(skipped_empty)}")
        
        # Add label encoding mapping for each processed column
        if processed_columns:
            mapping_lines = ["\nLabel encoding mapping:"]
            for col in processed_columns:
                le = encoders[col]
                pairs = [f"{cls} → {code}" for code, cls in enumerate(le.classes_)]
                mapping_lines.append(f"{col}: " + ", ".join(pairs))
            lines.append("\n".join(mapping_lines))

        output_text = "\n\n".join(lines) if lines else "No columns were encoded."

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Encoding failed: {str(e)}")
        output_text = f"Error during encoding: {str(e)}"
        processed_columns = []

    if hasattr(self.ui, 'outputs'):
        self.ui.outputs.clear()
        self.ui.outputs.append(output_text)
        
    return output_text, processed_columns

def decode(self):
    columns = self.ui_helper.selected_preprocessing_columns
    df = self.df

    if df is None:
        return self._output_error("Error: No DataFrame loaded.")
    if not columns:
        return self._output_error("Error: No column selected.")

    mode, ok = QInputDialog.getItem(
        None,
        "Choose Conversion Mode",
        "Select a conversion mode:",
        ["auto", "bins", "manual"],
        editable=False
    )
    if not ok:
        return "Conversion canceled.", []

    processed_columns = []
    report_lines = []

    for col in columns:
        if not np.issubdtype(df[col].dtype, np.number):
            report_lines.append(f"{col} skipped (not numeric)")
            continue

        try:
            if mode == 'auto':
                unique_vals = sorted(df[col].unique())
                val_map = {val: f"G{i+1}" for i, val in enumerate(unique_vals)}
                df[col] = df[col].map(val_map)
                report_lines.append(f"{col}: {len(unique_vals)} groups created")

            elif mode == 'bins':
                n = len(df[col].dropna())
                default_bins = max(1, int(np.ceil(np.log2(n) + 1)))

                nb_bins, ok1 = QInputDialog.getInt(
                    None,
                    f"Number of bins for {col}",
                    "Choose the number of bins:",
                    value=default_bins,
                    min=1,
                    max=50
                )
                if not ok1:
                    return "Conversion canceled during binning.", processed_columns

                min_val = df[col].min()
                max_val = df[col].max()
                bins = list(np.linspace(min_val, max_val, nb_bins + 1))

                labels = []
                for i in range(nb_bins):
                    label, ok2 = QInputDialog.getText(
                        None,
                        f"Label for bin {i+1} of {col}",
                        f"Enter label for bin from {bins[i]:.2f} to {bins[i+1]:.2f}:"
                    )
                    if not ok2 or label.strip() == "":
                        label = f"Bin {i+1}"
                    labels.append(label.strip())

                df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
                report_lines.append(f"{col}: {nb_bins} bins created with custom labels")

            elif mode == 'manual':
                unique_vals = sorted(df[col].unique())
                df[col] = df[col].astype(object)

                canceled = False
                for val in unique_vals:
                    new_label, ok = QInputDialog.getText(
                        None,
                        f"Manual Conversion - {col}",
                        f"Value {val} becomes:\n(Leave empty to skip, Cancel to stop column)",
                        text=str(val)
                    )
                    if not ok:
                        canceled = True
                        break
                    elif new_label.strip() == "":
                        continue
                    else:
                        df.loc[df[col] == val, col] = new_label.strip()

                if canceled:
                    report_lines.append(f"{col}: manual conversion canceled")
                    continue
                else:
                    report_lines.append(f"{col}: {len(unique_vals)} values processed manually")

            processed_columns.append(col)

        except Exception as e:
            report_lines.append(f"{col}: error - {str(e)}")

    output_text = "Conversion report:\n" + "\n".join(report_lines)
    return output_text, processed_columns

def encode_one_hot(self):
    try:
        columns = self.ui_helper.selected_preprocessing_columns
        df = self.df

        if df is None:
            QMessageBox.warning(self, "Warning", "No DataFrame loaded!")
            return "Error: No DataFrame loaded.", []

        if not columns:
            QMessageBox.warning(self, "Warning", "No column selected for encoding!")
            return "Error: No column selected for encoding.", []

        processed_columns = []       
        new_feature_names = []       
        already_numeric = []         
        non_categorical = []         
        encoders = {}                

        for col in columns:
            dtype = df[col].dtype
            if dtype == object or dtype.name == "category":
                # fit and transform
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                arr = ohe.fit_transform(df[[col]])

                cats = ohe.categories_[0]
                dummy_cols = [f"{col}_{cat}" for cat in cats]

                # create DataFrame of dummies and replace the original column
                df_dummies = pd.DataFrame(arr, columns=dummy_cols, index=df.index)
                df = pd.concat([df.drop(columns=[col]), df_dummies], axis=1)

                processed_columns.append(col)
                new_feature_names.extend(dummy_cols)
                encoders[col] = ohe
                if self.ui_helper.selected_columns :
                    for col in dummy_cols :
                        self.ui_helper.selected_columns.append(col)

            elif np.issubdtype(dtype, np.number):
                already_numeric.append(col)
            else:
                non_categorical.append(col)

        # reassign the transformed DataFrame back to the instance
        self.df = df

        # register this preprocessing step in the pipeline
        if processed_columns:
            self.pipeline_manager.add_preprocessing_step(
                "one_hot_encoding",
                encoders,
                processed_columns
            )
            # update feature names in the pipeline
            if self.pipeline_manager.feature_names:
                self.pipeline_manager.feature_names.extend(new_feature_names)
            else:
                self.pipeline_manager.feature_names = new_feature_names

        # build output message
        lines = []
        if processed_columns:
            lines.append(
                f"{len(processed_columns)} column(s) one-hot encoded and added to pipeline:\n"
                f"{', '.join(processed_columns)}"
            )
        if already_numeric:
            lines.append(f"Skipped already numeric column(s):\n{', '.join(already_numeric)}")
        if non_categorical:
            lines.append(f"Skipped non-categorical column(s):\n{', '.join(non_categorical)}")

        output_text = "\n\n".join(lines) if lines else "No columns were encoded."

    except Exception as e:
        QMessageBox.critical(self, "Error", f"One-hot encoding failed: {str(e)}")
        output_text = f"Error during encoding: {str(e)}"
        processed_columns = []

    # display in the UI if available
    if hasattr(self.ui, 'outputs'):
        self.ui.outputs.clear()
        self.ui.outputs.append(output_text)
    return output_text, processed_columns