from PyQt5.QtCore import Qt ,QSize
from PyQt5.QtWidgets import QCheckBox, QLabel, QHBoxLayout, QWidget, QVBoxLayout 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scripts.preprocessing.normalize_data import normalize_data
from scripts.preprocessing.encode_data import *
from scripts.visualization.correlation_matrix import show_correlation_matrix
from scripts.visualization.box_plot import show_boxplot
from scripts.preprocessing.handle_missing_data import *
from scripts.preprocessing.imbalanced_data import handle_imbalanced_data
from scripts.preprocessing.delete_duplicate import delete_duplicates
from scripts.preprocessing.handle_outliers import (
    detect_outliers_iqr,
    detect_outliers_isolation_forest
)
from GUI.dataset_manipulation import *
from GUI.dataset_viewer import DatasetViewer
from GUI.histogram import HistogramPlotter
from PyQt5.QtGui import QPixmap, QPainter, QIcon, QColor 



class UIHelper:
    def __init__(self, main_window):
        self.main_window = main_window
        self.df_window=None # df window 
        self.selected_columns = []
        
        self.setup_show_more_toggle(False , "" ,"")
        
        # for the vizualsation combobox
        self.update_visualize_options() #to get updates as soon as the app is launched.
        self.main_window.ui.comboBoxCla_Algo.currentTextChanged.connect(self.update_visualize_options)
        
        #for the clausters number and vizulation for claustering
        self.handle_groupbox_claustering() 
        self.main_window.ui.chooseAlgoCombo_2.currentTextChanged.connect(self.handle_groupbox_claustering)
        
        # Connect menu actions
        self.main_window.ui.actionClose_window.triggered.connect(self.main_window.close)
        self.main_window.ui.actionMinimize_2.triggered.connect(self.main_window.showMinimized)
        self.main_window.ui.actionMaximize.triggered.connect(self.toggle_maximize_restore)
        
        # Matplotlib canvas
        self.canvas = FigureCanvas(plt.figure())
        self.main_window.ui.widget_histo.layout().addWidget(self.canvas)
        
        # Histogram Plotter
        self.histogram_plotter = HistogramPlotter(self.canvas, self.main_window.ui.textStat)
        
        # Checkbox Layout
        self.checkbox_layout = QVBoxLayout(self.main_window.ui.checkboxes_container)
        self.checkboxes = []
        
        self.checkbox_layout_pre =QVBoxLayout(self.main_window.ui.checkboxes_container2)
        self.selected_preprocessing_columns = []
        self.checkboxes_pre = []
        
        # tab displaying 
        self.main_window.ui.tabWidget.currentChanged.connect(self.on_tab_changed)
        self.on_tab_changed(self.main_window.ui.tabWidget.currentIndex())
        
        # Connect buttons importation tab
        self.main_window.ui.btn_select_all.clicked.connect(self.select_all_checkboxes)
        self.main_window.ui.btn_unselect_all.clicked.connect(self.deselect_all_checkboxes)
        self.main_window.ui.btn_saveHisto.clicked.connect(self.histogram_plotter.save_histogram)
        self.main_window.ui.btn_remove.clicked.connect(self.remove_selected_columns)
        self.main_window.ui.btn_display.clicked.connect(self.display_df)
        self.main_window.ui.btn_matrix.clicked.connect(self.display_corr)
        self.main_window.ui.btn_box.clicked.connect(self.display_boxplot)
        #preprocessing tab
        self.main_window.ui.btn_select_all_pre.clicked.connect(self.select_all_checkboxes_pre)
        self.main_window.ui.btn_unselect_all_pre.clicked.connect(self.deselect_all_checkboxes_pre)        
        self.main_window.ui.btn_display_2.clicked.connect(self.display_df)
        
        
        #classification select label
        self.main_window.ui.comboBox_Att_Class.currentIndexChanged.connect(self.handle_user_selection_outcome)
        #Estimation select label
        self.main_window.ui.comboBox_Att_Estim.currentIndexChanged.connect(self.handle_user_selection_outcome)
        #Feature selection select label 
        self.main_window.ui.comboBox_outcome.currentIndexChanged.connect(self.handle_user_selection_outcome)
    
    def populate_checkboxes(self): 
        # Dynamically create checkboxes for df columns
        self.clear_checkboxes()
        df = self.main_window.df
        df_filt = self.main_window.df_filtred # filtered DataFrame
        if df is not None:
            all_cols = list(df.columns)
            filt_cols = list(df_filt.columns) if df_filt is not None else []
            selective_check = set(filt_cols) != set(all_cols)
            for column in all_cols:
                container = QWidget()
                container.setFixedHeight(24)
                        
                hbox = QHBoxLayout(container)
                hbox.setContentsMargins(2, 1, 2, 1)
                hbox.setSpacing(30)
                        
                checkbox = QCheckBox()
                checkbox.setFixedSize(16, 16)
                # Pre-check if in filtered columns and not all columns
                if selective_check and column in self.selected_columns:
                    checkbox.setChecked(True)
                label = QLabel(column)
                label.setMargin(0)
                label.setIndent(0)
                                        
                checkbox.stateChanged.connect(lambda state, col=column: self.on_checkbox_changed(col, state))
                label.mousePressEvent = lambda event, col=column: self.histogram_plotter.display(col,df=df,df_filt=df_filt)
                label.setCursor(Qt.PointingHandCursor)
                        
                hbox.addWidget(checkbox)
                hbox.addWidget(label)
                hbox.addStretch()
                        
                self.checkbox_layout.addWidget(container)
                self.checkboxes.append((checkbox, label))
            self.checkbox_layout.setSpacing(2)
        self.update_outcome_selection()
        self.populate_checkboxes_pre()
        self.update_file_info(self.main_window.df_filtred)
        
    def on_checkbox_changed(self, col, state):
        if state == Qt.Checked and len(self.selected_columns) == len(self.main_window.df.columns):
            self.selected_columns = []
        if state == Qt.Checked :
            if col not in self.selected_columns:
                self.selected_columns.append(col)
        else:
            if col in self.selected_columns:
                self.selected_columns.remove(col)
        if not self.selected_columns:
            self.selected_columns = list(self.main_window.df.columns)
            self.main_window.df_filtred = self.main_window.df
        else: 
            ordered_selected = [column for column in self.main_window.df.columns if column in self.selected_columns]
            self.main_window.df_filtred = self.main_window.df[ordered_selected]
            self.selected_columns = ordered_selected
        self.update_outcome_selection()
        self.populate_checkboxes_pre()
        self.update_file_info(self.main_window.df_filtred)

    def remove_selected_columns(self):
        if self.main_window.df is not None:
            to_remove = [label.text() for checkbox, label in self.checkboxes if checkbox.isChecked()]
            self.main_window.undo_stack.append((self.main_window.df.copy(), f"removing : {to_remove}",True))
            self.main_window.remove_stack.append((to_remove))
            if to_remove:
                self.main_window.df.drop(columns=to_remove, inplace=True)
                if self.histogram_plotter.last_selected_column in to_remove:
                    self.histogram_plotter.clear_histogram()
                self.main_window.ui.textStat.setText(f"{len(to_remove)} deleted column(s)")
                self.selected_columns=[]
                self.main_window.df_filtred = self.main_window.df
                self.populate_checkboxes()  
                self.update_outcome_selection()
                # self.update_file_info(self.main_window.current_file_path, self.main_window.df)
                
    def display_df(self):
        if self.main_window.df_filtred is not None:
            self.df_window = DatasetViewer(self.main_window.df_filtred)
            self.df_window.show()

    def display_corr(self) : 
        if self.main_window.df_filtred is not None:
            show_correlation_matrix(self.main_window.df_filtred)
    
    def display_boxplot(self) : 
        if self.main_window.df_filtred is not None:
            show_boxplot(self.main_window.df_filtred)
    def update_file_info(self, dataset):
        #Update UI labels with file info
        self.main_window.ui.label_columns.setText(str(dataset.shape[1]))
        self.main_window.ui.label_rows.setText(str(dataset.shape[0]))

    def clear_checkboxes(self):
    # Remove existing checkbox containers from layout
        for checkbox, label in self.checkboxes:
            container = checkbox.parentWidget()
            if container:
                self.checkbox_layout.removeWidget(container)
                container.deleteLater()
        self.checkboxes.clear()
    # importation tab
    def select_all_checkboxes(self):
        for item in self.checkboxes:
            if isinstance(item, tuple): 
                checkbox = item[0]
            else:  
                checkbox = item
            checkbox.setChecked(True)

    def deselect_all_checkboxes(self):
        for item in self.checkboxes:
            if isinstance(item, tuple):
                checkbox = item[0]
            else:
                checkbox = item
            checkbox.setChecked(False)
        self.selected_columns = []
        
    # preprossing tab
    def select_all_checkboxes_pre(self):
        for item in self.checkboxes_pre:
            if isinstance(item, tuple): 
                checkbox = item[0]
            else:  
                checkbox = item
            checkbox.setChecked(True)

    def deselect_all_checkboxes_pre(self):
        for item in self.checkboxes_pre:
            if isinstance(item, tuple):
                checkbox = item[0]
            else:
                checkbox = item
            checkbox.setChecked(False)
        self.selected_columns_pre = []

    def toggle_maximize_restore(self):
        #Toggle between maximized and normal window state
        if self.main_window.isMaximized():
            self.main_window.showNormal()
        else:
            self.main_window.showMaximized()
    
    def update_visualize_options(self):
        algo = self.main_window.ui.comboBoxCla_Algo.currentText().replace(" ", "").lower()
        self.main_window.ui.comboBox_visualize.clear()
        
        if algo == "logisticregression":
                self.main_window.ui.comboBox_visualize.addItems(["Logistic Equation", "Confusion Matrix"])
                self.main_window.ui.text_classifierOutput.setText("")
        elif algo == "decisiontree":
                self.main_window.ui.comboBox_visualize.addItems(["Confusion Matrix", "Decision Tree"])
                self.main_window.ui.text_classifierOutput.setText("")
        else:
                self.main_window.ui.comboBox_visualize.addItem("No Vizualisation")
        self.main_window.ui.btn_show.setEnabled(False)
        self.main_window.matrix = None
        self.main_window.equation = None
        
    def update_outcome_selection(self):
        if self.main_window.df_filtred is not None:
                combo = self.main_window.ui.comboBox_Att_Class
                combo_num = self.main_window.ui.comboBox_Att_Estim  #estimation tab combo box ( numerical columns only)
                combo_fs = self.main_window.ui.comboBox_outcome
                
                combo.blockSignals(True)
                combo_num.blockSignals(True)
                combo_fs.blockSignals(True)
                
                columns_num = self.main_window.df_filtred.select_dtypes(include=['number']).columns.tolist() 
                # Filter only numeric columns
                numeric_columns = self.main_window.df_filtred.select_dtypes(include=['number']).columns.tolist()        
                copyFs = columns_num.copy()
                copyFs.append("No Target") 
                
                combo.clear()
                combo_num.clear()
                combo_fs.clear()
                
                combo.addItems(columns_num)
                combo_num.addItems(numeric_columns)
                combo_fs.addItems(copyFs)
                if columns_num:
                    combo.setCurrentIndex(len(columns_num) - 1)
                    self.main_window.outcome_column = combo.currentText()
                if numeric_columns:
                    combo_num.setCurrentIndex(len(numeric_columns) - 1)
                    self.main_window.target = combo_num.currentText()
                if copyFs : 
                    combo_fs.setCurrentIndex(len(copyFs)-1)
                    self.main_window.outcome_fs = combo_fs.currentText()
                
                combo.blockSignals(False)
                combo_num.blockSignals(False)
                combo_fs.blockSignals(False)

    def handle_user_selection_outcome(self, index):
        combo = self.main_window.ui.comboBox_Att_Class
        combo_num=self.main_window.ui.comboBox_Att_Estim
        combo_fs = self.main_window.ui.comboBox_outcome
        
        self.main_window.outcome_column = combo.currentText()
        self.main_window.target = combo_num.currentText()
        self.main_window.outcome_fs = combo_fs.currentText()
    
    def handle_groupbox_claustering(self) : 
        if self.main_window.ui.chooseAlgoCombo_2.currentText().replace(" ", "").lower()=="emalgorithm" : 
            self.main_window.ui.groupBox_12.hide()
            self.main_window.ui.visualizeBox_2.hide()
            self.main_window.ui.btn_saveResultClust.hide()
            self.main_window.ui.text_clustererOutput.setText("")
            self.setup_show_more_toggle(False , "" , "")
        else : 
            self.main_window.ui.groupBox_12.show()
            self.main_window.ui.visualizeBox_2.show()
            self.main_window.ui.btn_saveResultClust.show()
            self.main_window.ui.text_clustererOutput.setText("")
            self.setup_show_more_toggle(False , "" , "")
    
    def handle_comboBox_transf(self) : 
        output_text =None
        selected_item = self.main_window.ui.treeTranf.currentItem()
        if selected_item : 
            selected_text = selected_item.text(0)
            if selected_text =="Normalize Data" : 
                output_text =self.main_window.save_and_process(lambda: normalize_data(self.main_window), selected_text)
            elif selected_text == "One Hot Encoder":
                output_text = self.main_window.save_and_process(lambda: encode_one_hot(self.main_window), selected_text)
            elif selected_text =="Label Encoder" :
                output_text = self.main_window.save_and_process(lambda: encode_label(self.main_window), selected_text)
            elif selected_text == "From Numerical to Categorical":
                output_text = self.main_window.save_and_process(lambda: decode(self.main_window), selected_text)
            elif selected_text == "Convert From Float To Integer":
                output_text = self.main_window.save_and_process(lambda: convert_float_to_int(self.main_window), selected_text)
            elif selected_text == "Change Variable's Name" : 
                output_text = self.main_window.save_and_process(lambda: columns_rename(self.main_window), selected_text)
        if output_text : 
            self.main_window.ui.op_msg.setPlainText(output_text)
        else :
            self.main_window.ui.op_msg.setPlainText("Method not implemented yet")
        self.populate_checkboxes()

    def handle_comboBox_clean(self):
        output_text=None
        selected_item = self.main_window.ui.treeClean.currentItem()
        if selected_item :
            selected_text = selected_item.text(0)
            if selected_text == "Replace with Mean":
                output_text = self.main_window.save_and_process(lambda :handle_missing_data(self.main_window, strategy="mean"),selected_text)
            elif selected_text == "Handle Imbalanced Data":
                output_text = self.main_window.save_and_process(lambda: handle_imbalanced_data(self.main_window), selected_text)
            elif selected_text =="Replace with Median" : 
                output_text = self.main_window.save_and_process(lambda :handle_missing_data(self.main_window, strategy="median"),selected_text)
            elif selected_text =="Using KNN" : 
                output_text = self.main_window.save_and_process(lambda : handle_missing_data_knn(self.main_window) , selected_text)
            elif selected_text =='Delete Duplicate' : 
                output_text = self.main_window.save_and_process(lambda : delete_duplicates(self.main_window),selected_text)
            elif selected_text == "IQR Method":
                 output_text = self.main_window.save_and_process(
             lambda: detect_outliers_iqr(self.main_window, self.selected_preprocessing_columns),
                selected_text
            )
            elif selected_text == "Isolation Forest":
                output_text = self.main_window.save_and_process(
                lambda: detect_outliers_isolation_forest(self.main_window, self.selected_preprocessing_columns),
                selected_text
            )
        if output_text : 
            self.main_window.ui.op_msg.setPlainText(output_text)
        else :
            self.main_window.ui.op_msg.setPlainText("Method not implemented yet")
        self.populate_checkboxes()

    def populate_checkboxes_pre(self):
        for cb in getattr(self, 'checkboxes_pre', []):
            self.checkbox_layout_pre.removeWidget(cb)
            cb.deleteLater()
        self.checkboxes_pre = []
        if len(self.main_window.df_filtred.columns) != 0 :
            df = self.main_window.df_filtred
        else : 
            df = self.main_window.df
        if df is not None:
            self.selected_preprocessing_columns = list(df.columns)
            
            for column in df.columns:
                checkbox = QCheckBox(column)
                checkbox.setChecked(False)  

                def _on_state_changed(state, col=column):
                    if state == Qt.Checked and len(self.selected_preprocessing_columns) == len(df.columns):
                        self.selected_preprocessing_columns = [col]  
                    else:
                        if state == Qt.Checked:
                            if col not in self.selected_preprocessing_columns:
                                self.selected_preprocessing_columns.append(col)
                        else:
                            if col in self.selected_preprocessing_columns:
                                self.selected_preprocessing_columns.remove(col)
                
                    if not self.selected_preprocessing_columns:
                        self.selected_preprocessing_columns = list(df.columns)
                

                checkbox.stateChanged.connect(_on_state_changed)
                self.checkbox_layout_pre.addWidget(checkbox)
                self.checkboxes_pre.append(checkbox)
    
    def on_tab_changed(self , index) : 
        displaying = (index in (0 , 1 ,4, 5))
        self.main_window.ui.actionModel_params.setVisible( not displaying )

    def setup_show_more_toggle(self , enable , output_text , output_final):
        btn = self.main_window.ui.btn_showMore

        btn.setCheckable(True)
        btn.setChecked(False)

        if enable :
            btn.setEnabled(True)
        else : 
            btn.setEnabled(False)
        
        btn.setFixedSize(60, 25) 
        btn.setCursor(Qt.PointingHandCursor)

        diameter = 10 
        pix = QPixmap(diameter, diameter)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor("#FFFFFF"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, diameter, diameter)
        painter.end()
        btn.setIcon(QIcon(pix))
        btn.setIconSize(QSize(diameter, diameter))

        btn.setText("show more")
        btn.setLayoutDirection(Qt.LeftToRight)

        btn.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 12px;
                background: #35586D;
                color: white;
                font: bold 7px Arial;
                padding: 1px 3px;
                margin: 0px;
                min-width: 60px;
                min-height: 25px;
                }
                QPushButton:checked {
                    background: #EE6843;
                }
                QPushButton:disabled {
                    background: #e0e0e0;
                    color: #aaa;
                }
                QPushButton::icon {
                    width: 10px;
                    height: 10px;
                    padding-left: 2px;
                    padding-right: 2px;
                }
        """)

        def on_toggled(checked: bool):
            btn.setText("show less" if checked else "show more")
            if  not checked:
                self.main_window.ui.text_clustererOutput.setText(output_text) 
            else:
                self.main_window.ui.text_clustererOutput.setText(output_final) 

        btn.toggled.connect(on_toggled)