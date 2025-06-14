from PyQt5.QtWidgets import QMessageBox
def delete_duplicates(self):

    if self.df is None:
        QMessageBox.warning(self, "Warning", "No DataFrame loaded!")
        return "Error: No DataFrame loaded.", 0

    before_count = len(self.df)

    df_no_dup = self.df.drop_duplicates().reset_index(drop=True)
    removed = before_count - len(df_no_dup)

    self.df = df_no_dup

    if removed > 0:
        output_text = f"{removed} duplicate row(s) removed successfully."
    else:
        output_text = "No duplicate rows found."

    # 6) Display in UI if available
    if hasattr(self.ui, 'outputs'):
        self.update_file_info(self.main_window.current_file_path, self.main_window.df_filtred)
        self.ui.outputs.clear()
        self.ui.outputs.append(output_text)

    return output_text, removed
