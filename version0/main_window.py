from PyQt5.QtWidgets import QMainWindow, QFileDialog , QMessageBox , QLineEdit ,QDialog
import pandas as pd
import sys
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt
from scripts.preprocessing.data_loader import load_csv
from version0.interface import Ui_MainWindow
from version0.ui_functions import UIHelper
from scripts.modeling.supervised.logistic_regression import apply_logistic_regression
from scripts.visualization.logistic_equation import EquationDialog
from scripts.visualization.confusion_matrix import show_confusion_matrix_dialog
from scripts.visualization.K_Means_Vis import FenetreGraphiqueKMeans
from scripts.modeling.supervised.decision_tree import apply_decision_tree
from scripts.modeling.unsupervised.DBSCAN import apply_dbscan
from scripts.modeling.unsupervised.Bicluster import apply_biclustering
from scripts.modeling.unsupervised.K_Means import apply_kmean
from scripts.modeling.supervised.linear_regression import apply_linear_regression
from scripts.preprocessing.feature_selection import apply_feature_selection
from scripts.visualization.linear_equation import EquationDialogEsti
from scripts.visualization.decision_tree_test import show_decision_tree
from scripts.visualization.decision_tree_test2 import DecisionTreeWindow
from version0.ModelParams import DynamicParameterDialog
from version0.UserGuide import show_user_guide_dialog
from scripts.evaluation.save_model import load_entire_pipeline ,apply_pipeline_and_predict,save_entire_pipeline  
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        # self.current_file_path = None
        self.pipeline_manager = None
        self.ui.setupUi(self)
        self.new_df = None
        self.df = None
        self.never_touch=None
        self.outcome_column = None
        self.target=None # to be checked
        self.outcome_fs = None
        self.df_filtred = None
        self.matrix = None
        self.undo_stack = []
        self.remove_stack = [] 
        self.saved_params = {
            'logistic_regression': {},
            'decision_tree': {},
            'linear_regression' : {},
            'k_means': {},
            'em_algorithm': {}}
        self.equation = None
        self.lineEdit_supplied = QLineEdit()
        self.ui_helper = UIHelper(self)
        
        
        #class upload saved model
        self.ui.btn_uploadModClass.clicked.connect(lambda : self.apply_pipe_load(1))
        #esti upload saved model
        self.ui.btn_uploadModEsti.clicked.connect(lambda : self.apply_pipe_load(2))
        
        #temporary
        self.ui.btn_saveResultEsti.setVisible(False)
        self.ui.btn_savResultClass.setVisible(False)
        
        self.ui.actionNew.triggered.connect(lambda : load_csv(self, 1))
        self.ui.actionNew.setShortcut(QKeySequence.Open) # keyboard shortcut open file
        self.ui.actionUndo.triggered.connect(self.undo_action) 
        self.ui.actionUndo.setShortcut(QKeySequence.Undo) # keyboard shortcut undo action 
        self.ui.actionSave_As.triggered.connect(self.save_as)  
        self.ui.actionSave_As.setShortcut(QKeySequence.Save)# keyboard shortcut Save file
        self.ui.actionSave.triggered.connect(self.save)
        self.ui.actionModel_params.triggered.connect(self.handle_changing_params) 
        mod =( Qt.MetaModifier if sys.platform == 'darwin' else Qt.ControlModifier)
        self.ui.actionModel_params.setShortcut( QKeySequence(int(mod) | Qt.Key_E))
        self.ui.actionAbout.triggered.connect(self.show_about_dialog)
        self.ui.actionHelp_Contents.triggered.connect(self.show_user_guide)
        self.ui.actionHelp_Contents.setShortcut(QKeySequence.HelpContents) #keyboard shortcut user guide
        
        self.ui.actionClose_window.setShortcut(QKeySequence.Quit)
        
        # Connect buttons classifcation 
        self.ui.btn_apply.clicked.connect(self.apply_classification)
        self.ui.btn_file.clicked.connect(self.choose_test_file)

        
        self.ui.btn_show.clicked.connect(self.show_results)
        self.ui.btn_show.setEnabled(False)
        
        self.ui.btn_savResultClass.setEnabled(False)
        self.ui.btn_saveResultClust.clicked.connect(self.save_results)
        
        #Connect buttons Claustering 
        self.ui.btn_apply_2.clicked.connect(self.apply_clustering)
        self.ui.btn_file_2.clicked.connect(self.choose_test_file)
        self.ui.btn_saveResultClust.setEnabled(False)
        self.ui.btn_show_2.clicked.connect(self.show_kmeans_results)
        self.ui.btn_show_2.setEnabled(False)
        
        #Connect Prepoccessing buttons 
        self.ui.btn_apply_tr.clicked.connect(self.ui_helper.handle_comboBox_transf)
        self.ui.btn_apply_cl.clicked.connect(self.ui_helper.handle_comboBox_clean)

        #Feature selection 
        self.ui.btn_applyFS.clicked.connect(self.handling_feature_selection)

        #Connect buttons Estimation
        self.ui.btn_apply_esti.clicked.connect(self.apply_estimation)
        self.ui.btn_file_3.clicked.connect(self.choose_test_file)
        self.ui.btn_saveResultEsti.setEnabled(False)
        self.ui.btn_show_esti.setEnabled(False)
        self.ui.btn_show_esti.clicked.connect(self.show_results_esti)
        self.ui.comboBox_Att_Estim.clear()

        #save model
        self.ui.btn_saveModel_Esti.clicked.connect(self.save_estimation_model)
        self.ui.btn_saveModel.clicked.connect(self.save_classification_model)

    def choose_test_file(self):
        chemin_fichier, _ = QFileDialog.getOpenFileName(self,"Select test file","", "CSV Files (*.csv);;All Files (*)")
        if chemin_fichier:
            self.lineEdit_supplied.setText(chemin_fichier)
    
    def show_user_guide(self):
        show_user_guide_dialog(self)

    def apply_classification(self):
        if self.df_filtred is not None : 
            self.matrix = None # restore the matrix 
            self.equation = None #restor the equation
            if self.ui.btn_crossVal.isChecked() : 
                cv = self.ui.nbrFoldsBox.value()
                supplied = None
                test_size = None
            elif self.ui.btn_perecentSplit.isChecked():
                cv = None
                test_size = self.ui.PsplitSpinBox.value() / 100
                supplied = None
            elif self.ui.btn_useTrainSet.isChecked():
                cv = None 
                test_size = None
                supplied = None
            elif self.ui.btn_testSet.isChecked():
                if not self.lineEdit_supplied.text().strip() :
                    QMessageBox.warning(self,"Warning" , "Please import the supplied test set")
                    return
                else : 
                    cv = None
                    supplied = self.lineEdit_supplied.text()
                    test_size = None

            try : 
                model = self.ui.comboBoxCla_Algo.currentText().replace(" ", "").lower()
                
                if model == "logisticregression":
                    output_text,self.model_class, self.equation , self.matrix , self.new_df = apply_logistic_regression(self, cv, supplied, test_size,self.saved_params["logistic_regression"])
                elif model == "decisiontree":
                    output_text, self.model_class, self.matrix , self.new_df = apply_decision_tree(self, cv, supplied, test_size, self.saved_params["decision_tree"])
                    self.tree_plot = self.model_class
                self.ui.btn_savResultClass.setEnabled(True)
                self.ui.btn_show.setEnabled(True)
                self.ui.text_classifierOutput.setText(output_text) 
            except Exception as e : 
                QMessageBox.critical(self,"Error", f"An error occurred during execution: {e}")
        else : 
            QMessageBox.warning(self, "Warning", "No file loaded for processing.")
        
        self.lineEdit_supplied.clear()

    def save_results(self):
        if getattr(self, 'is_test_file', False):
            save_path, _ = QFileDialog.getSaveFileName(self,"Save predictions to...","updated_predictions.csv","CSV Files (*.csv);;All Files (*)")
            if save_path:
                try:
                    self.new_df.to_csv(save_path, index=False)
                    self.df = self.new_df
                    self.df_filtred = self.df
                    QMessageBox.information(
                        self,"Success",f"Predictions have been saved in the Supplied TestSet, and the dataset has been labeled.")
                    self.is_test_file = False
                except Exception as e:
                    QMessageBox.critical(self,"Error",f"Failed to save predictions: {e}")
        else :        
            self.df = self.new_df
            QMessageBox.information(self,"Success","The dataset has been updated with the predictions")
        self.ui_helper.selected_columns.append("Cluster_Assigned")
        self.df_filtred = self.new2
        self.never_touch = self.new1
        self.ui_helper.populate_checkboxes()
        
    def show_results(self):
        if self.ui.comboBox_visualize.currentText().replace(" ", "").lower() == "logisticequation":
            if self.equation is None:
                QMessageBox.information(self, "Results", "The logistic regression model has not been trained yet.")
            else:
                dialog = EquationDialog(self.equation, parent=self)
                dialog.resize(1000, 200)  
                dialog.exec_()
        elif self.ui.comboBox_visualize.currentText().replace(" ", "").lower() == "confusionmatrix" : 
            show_confusion_matrix_dialog(self.matrix , parent=self)
        elif self.ui.comboBox_visualize.currentText().replace(" ", "").lower() == "decisiontree":
            if self.tree_plot is None:
                QMessageBox.information(self, "Results", "The decision tree has not been trained yet.")
            else:
                class_names = [str(cls) for cls in self.tree_plot.classes_]
                feature_names = self.tree_plot.feature_names_in_
                show_decision_tree(self.tree_plot, feature_names, class_names) #<- je dois enlever ca 
                #self.tree_window = DecisionTreeWindow(model=self.tree_plot,feature_names=feature_names,class_names=class_names)
                #self.tree_window.show()
    
    
    def apply_clustering(self): 
        if self.df_filtred is not None: 
            if self.ui.btn_perecentSplit_2.isChecked(): 
                test_size = self.ui.PsplitSpinBox_2.value() / 100
                supplied = None
            elif self.ui.radioButton_2.isChecked(): 
                if not self.lineEdit_supplied.text().strip() :
                    QMessageBox.warning(self,"Warning" , "Please import the supplied test set")
                    return 
                else : 
                    supplied = self.lineEdit_supplied.text()
                    test_size = None
            elif self.ui.btn_useTrainSet_2.isChecked(): 
                test_size = None
                supplied = None

            try: 
                model = self.ui.chooseAlgoCombo_2.currentText().replace(" ", "").lower()

                if model == "dbscan":
                    output_text, output_final , model, new_df, medoids_df ,self.new1 ,self.new2 = apply_dbscan(self, supplied=supplied, test_size=test_size)
                    self.new_df = new_df

                elif model == "k-means":
                    n_clusters_input = self.ui.NbrClusters.text()          
                    if n_clusters_input.strip() == "":
                        
                        QMessageBox.warning(self, "Erreur", "Veuillez entrer un nombre de clusters.")
                        return
                    else:
                        try:
                            n_clusters = int(n_clusters_input)
                            if n_clusters < 1:
                                raise ValueError
                        except ValueError:
                            QMessageBox.warning(self, "Error", "Please enter a valid number of clusters (integer > 0).")
                            return
                    output_text, output_final , model, new_df, medoids_df ,self.new1 ,self.new2 = apply_kmean(self, n_clusters, supplied=supplied, test_size=test_size)
                    self.kmeans_model = model
                    self.df_clustered = new_df
                    self.new_df = new_df
                    self.medoids_df = medoids_df

                else:
                    raise ValueError("Algorithme unknown.")

                
                self.ui.btn_saveResultClust.setEnabled(True)
                self.ui.btn_show_2.setEnabled(True)
                self.ui_helper.setup_show_more_toggle(True,output_text,output_final)
                self.ui.text_clustererOutput.setText(output_text) 

            except Exception as e: 
                QMessageBox.critical(self, "Error", f"An error occurred during execution: {e}")
        else: 
            QMessageBox.warning(self, "Warning", "No file loaded for processing.")
        
        self.lineEdit_supplied.clear()



    def apply_estimation(self):
        if self.df_filtred is not None:
            # Get validation parameters
            if self.ui.btn_crossVal_2.isChecked():
                cv = self.ui.nbrFoldsBox_2.value()
                supplied = None
                test_size = None
            elif self.ui.btn_perecentSplit_3.isChecked():
                cv = None
                test_size = self.ui.PsplitSpinBox_3.value() / 100
                supplied = None
            elif self.ui.btn_testSet_esti.isChecked():
                if not self.lineEdit_supplied.text().strip() :
                    QMessageBox.warning(self,"Warning" , "Please import the supplied test set")
                    return
                else : 
                    cv = None
                    supplied = self.lineEdit_supplied.text()
                    test_size = None
            elif self.ui.btn_useTrainSet_3.isChecked(): 
                cv = None
                test_size = None
                supplied = None

            try : 
                # Call the linear regression function
                output_text, model, new_df = apply_linear_regression(self, cv, supplied, test_size , self.saved_params["linear_regression"])
                
                self.linear_model = model
                self.linear_feature_names = model.feature_names_in_         
                # Update UI
                self.ui.btn_saveResultEsti.setEnabled(True)
                self.ui.btn_show_esti.setEnabled(True)
                self.ui.text_estimaterOutput.setPlainText(output_text)
                self.new_df = new_df
                        
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}\n")
        else : 
            QMessageBox.warning(self, "Warning", "No file loaded for processing.")
        
        self.lineEdit_supplied.clear()

    def show_kmeans_results(self):
        if self.ui.comboBox_visualize_2.currentText().replace(" ", "").lower() == "clusters&centroids":
            if not hasattr(self, "kmeans_model") or self.kmeans_model is None:
                QMessageBox.information(self, "Results", "The K-Means model has not been run yet.")
                return

            self.kmeans_window = FenetreGraphiqueKMeans(
                df=self.df_clustered,
                df_filtred=self.df_filtred,
                kmeans_model=self.kmeans_model,
                medoids_df=getattr(self, "medoids_df", None)
            )
            self.kmeans_window.show()

    def show_results_esti(self):
        if self.ui.comboBox_visualize_3.currentText().replace(" ", "").lower() == "linearequation":
            if not hasattr(self, "linear_model") or not hasattr(self.linear_model, "coef_"):
                QMessageBox.information(self, "Results", "The linear regression model is not trained yet")
            else:
                from scripts.visualization.linear_equation import linear_regression_equation
                equation = linear_regression_equation(self.linear_model, self.linear_feature_names)
                dialog = EquationDialogEsti(equation, parent=self)
                dialog.resize(1000, 200)
                dialog.exec_()

    def save_and_process(self, function, description):
        if self.df_filtred is not None:
                try:
                    copy = self.df.copy()
                    if callable(function):
                        result = function()
                        if isinstance(result, type(lambda: None)):
                            output_text , self.processed_column= result()
                        else:
                            output_text,self.processed_column = result
                    else:
                        output_text ,self.processed_column= function(self)
                    if isinstance(self.processed_column, list) and self.processed_column:
                        if description != "One Hot Encoder" :
                            copy = copy[self.processed_column]
                        self.undo_stack.append((copy, description, False))
                    if not self.ui_helper.selected_columns:
                        self.df_filtred = self.df
                    else:
                        if description == "One Hot Encoder" : 
                            for col in self.processed_column : 
                                self.ui_helper.selected_columns.remove(col)
                        self.df_filtred = self.df[self.ui_helper.selected_columns]
                    return output_text
                except Exception as e:
                        QMessageBox.critical(self, "Error", f"An error occurred during processing: {e}")
        else:
                QMessageBox.warning(self, "Warning", "No file loaded for processing.")

    def show_about_dialog(self):
     about_text = (
        "<div style='display: flex; align-items: center; gap: 15px; margin-bottom: 15px;'>"
        "<img src='minecLogo.png' width='80'>"
        "<div>"
        "<span style='font-size: 18px; font-weight: bold;'>Minec - Data Mining Toolkit</span><br>"
        "<span style='font-size: 14px;'>Version 1.0</span>"
        "</div>"
        "</div>"
        "<div style='line-height: 1.6;'>"
        "Developed by Team Minec (2025)<br>"
        "Final Year Project, USTHB<br>"
        "Powered by Python + PyQt5<br><br>"
        "<b>Supervisor:</b> Mr BABALI Riadh<br>"
        "<b>Team Members:</b><br>"
        "• SADMI Mohamed Riad<br>"
        "• AIT AHCENE Melissa<br>"
        "• BOUTAGHOU Maria Ghalia<br>"
        "• MENTIZI Rayane Rafik<br><br>"
        "<b>Contact:</b> groupepfedatamining@gmail.com<br>"
        "<b>GitHub:</b> <a href='https://github.com/team-minec/minec'>github.com/team-minec/minec</a>"
        "</div>"
     )

     msg_box = QMessageBox(self)
     msg_box.setWindowTitle("About Minec")
     msg_box.setTextFormat(Qt.TextFormat.RichText)
     msg_box.setText(about_text)
    
      # Style the dialog
     msg_box.setStyleSheet("""
        QMessageBox {
            background-color: white;
            font-family: Segoe UI, Arial;
        }
        QMessageBox QLabel {
            color: #333333;
        }
     """)
    
     ok_button = msg_box.addButton(QMessageBox.Ok)
     ok_button.setStyleSheet("""
        QPushButton {
            color: white;
            background-color: #023C5A;
            padding: 8px 16px;
            border-radius: 4px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #035B7E;
        }
     """)

     msg_box.exec_()
    
    def save(self):
    # First check if we have data to save
     if not hasattr(self, 'df_filtred') or self.df_filtred is None:
        QMessageBox.warning(self, "Warning", "No dataset loaded to save.")
        return
    
     # If we have a current file path, try to save there
     if hasattr(self, 'fichier_actuel') and self.fichier_actuel:
        try:
            self.df_filtred.to_csv(self.fichier_actuel, index=False)
            QMessageBox.information(self, "Success", f"File successfully saved to:\n{self.fichier_actuel}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving file:\n{e}")
     else:
        # No current file path, do Save As instead
        self.save_as()

    def save_as(self):
    # First check if we have data to save
     if not hasattr(self, 'df_filtred') or self.df_filtred is None:
        QMessageBox.warning(self, "Warning", "No dataset loaded to save.")
        return
    
     options = QFileDialog.Options()
     file_path, _ = QFileDialog.getSaveFileName(
        self, 
        "Save As", 
        "", 
        "CSV Files (*.csv);;All Files (*)", 
        options=options
     )

     if file_path:
        try:
            self.df_filtred.to_csv(file_path, index=False)
            self.fichier_actuel = file_path  # Update current file path
            QMessageBox.information(self, "Success", f"File successfully saved as:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving file:\n{e}")
    
    def undo_action(self):
        if not self.undo_stack:
            QMessageBox.information(self, "Undo", "No action to undo.")
            return
        df_prev, description, is_removing  = self.undo_stack.pop()
        if is_removing : 
            removed = self.remove_stack.pop()
            df = self.df
            full_order = list(self.never_touch.columns)
            cols_to_include = [
            col for col in full_order
            if col in df.columns or col in removed]
            new_df = pd.DataFrame(index=df.index)
            for col in cols_to_include:
                if col in df.columns:
                    new_df[col] = df[col]
                else:
                    new_df[col] = self.never_touch[col]
            self.df = new_df
            self.df_filtred=self.df
        else : 
            if description == "One Hot Encoder" : 
                self.df = df_prev
                self.df_filtred = self.df
                self.ui_helper.selected_columns = []
            else : 
                df = self.df.columns
                for col in self.df :
                    if col in df_prev : 
                        self.df[col] = df_prev[col]
                if description == "Handle Imbalanced Data":
                    self.df.dropna(how="all", inplace=True)
                self.df_filtred = self.df
                if len(self.ui_helper.selected_columns) != 0: 
                    self.df_filtred = self.df_filtred[self.ui_helper.selected_columns]
            if description not in ("From Numerical to Categorical" , "Handle Imbalanced Data"):
                self.pipeline_manager.remove_last_preprocessing_step()
            self.ui.op_msg.setPlainText("")
        self.ui_helper.populate_checkboxes()
        # self.ui_helper.update_file_info(self.current_file_path, self.df)
        QMessageBox.information(self, "Undo", f"The action '{description}' has been undone.")

    def handling_feature_selection(self): 
        output_text , output_suggestion = apply_feature_selection(self)
        self.ui.text_sugg.setPlainText(output_suggestion)
        self.ui.text_FSOutput.setPlainText(output_text)
    
    def handle_changing_params(self):
        active_tab = self.ui.tabWidget.currentIndex()  

        model_dialogs = {
                "classification": {
                        "logisticregression": "logistic_regression",
                        "decisiontree": "decision_tree",
                },
                "regression": {
                        "linearregression": "linear_regression",
                },
                "clustering": {
                        "k-means": "k_means",
                        "emalgorithm": "em_algorithm",
                }
        }
        if active_tab == 0 or active_tab == 1 or active_tab == 5 :
            QMessageBox.information(self , "Edit Params" , "There is no model params to edit in this tab")
            return
        else : 
            
            if active_tab == 3:  #"classification"
                model_classif = self.ui.comboBoxCla_Algo.currentText().replace(" ", "").lower()
                model = model_dialogs["classification"].get(model_classif)
            elif active_tab == 2:  # "regression"
                model_reg = self.ui.AlgoEstimCombo.currentText().replace(" ", "").lower()  
                model = model_dialogs["regression"].get(model_reg)
            elif active_tab == 4:  # "clustering"
                model_clust = self.ui.chooseAlgoCombo_2.currentText().replace(" ", "").lower() 
                model = model_dialogs["clustering"].get(model_clust)
        if model:
                saved_params = self.saved_params.get(model, {})
                dialog = DynamicParameterDialog(self, model_type=model, saved_params=saved_params)
                if dialog.exec_() == QDialog.Accepted:
                        params = dialog.get_params()
                        self.saved_params[model] = params
        else:
            QMessageBox.warning(self , "Warning","Model not found for the selected tab.")


    def save_estimation_model(self):
        if hasattr(self, 'linear_model'):
            if len(self.ui_helper.selected_columns) == 0 :
                self.pipeline_manager.set_feature_names(self.df.columns)
            else : 
                self.pipeline_manager.set_feature_names(self.ui_helper.selected_columns)
            self.pipeline_manager.set_model(self.linear_model)
            save_entire_pipeline(self.pipeline_manager)
        else:
            QMessageBox.warning(self, "Model Not Found", "No trained linear regression model found.")

    def save_classification_model(self):
        if hasattr(self, 'model_class'):
            if len(self.ui_helper.selected_columns) == 0 :
                self.pipeline_manager.set_feature_names(self.df.columns)
            else : 
                self.pipeline_manager.set_feature_names(self.ui_helper.selected_columns)
            self.pipeline_manager.set_model(self.model_class)
            save_entire_pipeline(self.pipeline_manager)
        else:
            QMessageBox.warning(self, "Model Not Found", "No trained classification model found.")
    
    def apply_pipe_load(self, number):
        pm = load_entire_pipeline(self)
        if pm is None:
                return
        if pm.model is None:
            QMessageBox.warning(self, "No model", "The loaded pipeline does not contain a model.")
            return
        if not hasattr(self, "df") or self.df is None or self.df.empty:
            QMessageBox.warning(self, "Missing data", "Please load a dataset first.")
            return
        logs, new_df,self.df = apply_pipeline_and_predict(pm, self.df, detailed_log=True)
        if number == 1:
            self.ui.text_classifierOutput.setPlainText(logs)
        else:
            self.ui.text_estimaterOutput.setPlainText(logs)
        if new_df is None:
            QMessageBox.critical(self, "Prediction error", "The prediction failed. Please check the logs for more details.")
            return
        self.pipe_loaded = pm
        self.df_filtred = new_df
        common_cols = [col for col in self.df.columns if col in new_df.columns]
        uncommon_cols = [col for col in self.df.columns if col not in self.never_touch] 
        inter = self.df.copy()
        for col in common_cols:
            if pd.api.types.is_integer_dtype(inter[col].dtype) and pd.api.types.is_float_dtype(new_df[col].dtype):
                inter[col] = inter[col].astype(float)
            inter.loc[:, col] = new_df[col]
        # to add the new columns to never touch for an undo use
        if uncommon_cols:
            self.never_touch = pd.concat([self.never_touch, self.df[uncommon_cols]], axis=1)
        common = [col for col in self.df.columns if col in self.never_touch.columns]
        others = [col for col in self.never_touch.columns if col not in common]
        self.never_touch = self.never_touch[common + others]
        self.df=inter
        self.ui_helper.selected_columns=list(self.df_filtred.columns)
        self.ui_helper.populate_checkboxes()
