from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView

class DatasetViewer(QWidget):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Dataset Viewer")
        self.setGeometry(200, 200, 800, 600)

        self.df = df
        self.layout = QVBoxLayout()

        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        self.setLayout(self.layout)
        self.load_data()

    def load_data(self):
        if self.df is None:
            return

        self.table.setRowCount(len(self.df))
        self.table.setColumnCount(len(self.df.columns))
        self.table.setHorizontalHeaderLabels(self.df.columns)

        for i in range(len(self.df)):
            for j in range(len(self.df.columns)):
                valeur = str(self.df.iloc[i, j])
                self.table.setItem(i, j, QTableWidgetItem(valeur))

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
   