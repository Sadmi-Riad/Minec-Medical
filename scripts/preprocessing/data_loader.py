import pandas as pd
from PyQt5.QtWidgets import QFileDialog
from scripts.evaluation.save_model import PipelineManager

def load_csv(self , number):
    #Load a CSV file and return the dataset
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)

    if file_path:
        try:
            df = pd.read_csv(file_path)
            if df is not None :
                self.never_touch = df.copy()
                self.df= df
                self.df_filtred = df
                self.pipeline_manager = PipelineManager()
                self.em_model = None
                self.em_k = None
                clear_all(self)
                if number == 1 : #for maria's version
                    self.ui_helper.update_file_info(self.df)
                    self.ui_helper.populate_checkboxes()
                    self.ui_helper.update_outcome_selection()
                    self.ui_helper.histogram_plotter.clear_histogram()
                    self.ui_helper.handle_groupbox_claustering() 
                    self.ui_helper.selected_columns = []
                    # self.ui_helper.main_window.current_file_path = file_path
                    file_name = file_path.split("/")[-1].rsplit(".", 1)[0]
                    self.ui.label_filename.setText(file_name)
        except Exception as e:
            print("Error loading CSV:", e)
            return

def get_target(self , choice):
    if choice == 1 : 
        X = self.df_filtred.select_dtypes(include=['int64', 'float64']).drop(columns=[self.outcome_column], errors='ignore')
        y = self.df_filtred[self.outcome_column]
    elif choice == 2 : 
        if self.outcome_fs == 'No Target' : 
            X = self.df_filtred
            y = None
        else : 
            X = self.df_filtred.select_dtypes(include=['int64', 'float64']).drop(columns=[self.outcome_fs], errors='ignore')
            y = self.df_filtred[self.outcome_fs]
    return y, X

def clear_all(self):
    self.ui.op_msg.setPlainText("")
    self.ui.text_estimaterOutput.setPlainText("")
    self.ui.text_classifierOutput.setPlainText("")
    self.ui.text_clustererOutput.setPlainText("")
    self.ui.text_sugg.setPlainText("")
    self.ui.text_FSOutput.setPlainText("")
    self.ui.btn_show_esti.setEnabled(False)
    self.ui.btn_show_2.setEnabled(False)
    self.ui.btn_show.setEnabled(False)
    self.ui.btn_saveResultClust.setEnabled(False)
    self.ui_helper.setup_show_more_toggle(False , "" , "")