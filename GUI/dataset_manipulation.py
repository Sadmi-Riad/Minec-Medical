from PyQt5.QtWidgets import QInputDialog

def columns_rename(self):
    columns = self.ui_helper.selected_preprocessing_columns
    output_text = "Column renaming results:\n\n"
    processed_columns = []
    selected = self.ui_helper.selected_columns
    if selected :
        self.stack_columns.append(selected.copy())
    for col in columns:
        new_name, ok = QInputDialog.getText(
            None, 
            "Rename Column", 
            f"Enter new name for column '{col}':"
        )
        if ok and new_name:
            self.df = self.df.rename(columns={col: new_name})
            self.df_filtred = self.df_filtred.rename(columns={col: new_name})
            if getattr(self.ui_helper, "selected_columns", None):
                # Update selected_columns list if defined.
                if col in self.ui_helper.selected_columns:
                    index = self.ui_helper.selected_columns.index(col)
                    self.ui_helper.selected_columns[index] = new_name
            processed_columns.append(new_name)
            output_text += f"Column '{col}' renamed to '{new_name}' - succeeded\n"
        else:
            output_text += f"Column '{col}' was not renamed (operation canceled or no input provided).\n"
        
    return output_text, processed_columns

def convert_float_to_int(self):
    columns = self.ui_helper.selected_preprocessing_columns
    df = self.df
    output_text = "Float to Int conversion results:\n\n"
    processed_columns = []

    for col in columns:
        # Check if the column exists in df and if it is of float type.
        if col in df.columns and df[col].dtype.kind == 'f':
            # Convert the column: fill missing values if necessary, then convert to int.
            df[col] = df[col].fillna(0).astype(int)
            processed_columns.append(col)
            output_text += f"Column '{col}' converted from float to int successfully.\n"
        elif col in df.columns:
            output_text += f"Column '{col}' is not of float type and was not converted.\n"
        else:
            output_text += f"Column '{col}' does not exist in the dataframe and was skipped.\n"

    # Update self.df with the changes.
    self.df = df

    # Update self.df_filtred so that it reflects only the columns present in the updated self.df.
    existing_cols = [col for col in self.df_filtred.columns if col in self.df.columns]
    self.df_filtred = self.df[existing_cols].copy()

    return output_text, processed_columns