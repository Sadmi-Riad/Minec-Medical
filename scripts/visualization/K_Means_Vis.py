from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QHBoxLayout, QComboBox, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class FenetreGraphiqueKMeans(QWidget):
    def __init__(self, df, df_filtred, kmeans_model, medoids_df=None):
        super().__init__()
        self.setWindowTitle("Cluster Visualization")
        self.setGeometry(150, 150, 800, 600)

        self.df = df  
        self.df_filtred = df_filtred 
        self.kmeans = kmeans_model
        self.medoids_df = medoids_df

        self.numeric_columns = self.df_filtred.select_dtypes(include=["number"]).columns.tolist()
        if len(self.numeric_columns) < 2:
            raise ValueError("The DataFrame must contain at least two numeric columns.")

        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        
        self.checkbox_layout = QHBoxLayout()
        self.cb_clusters = QCheckBox("Show Clusters")
        self.cb_centroids = QCheckBox("Show Centroides")
        self.cb_medoids = QCheckBox("Show the medoids")

        self.checkbox_layout.addWidget(self.cb_clusters)
        self.checkbox_layout.addWidget(self.cb_centroids)
        self.checkbox_layout.addWidget(self.cb_medoids)

        # zone de selection des axes
        self.axis_selection_layout = QHBoxLayout()
        self.combo_x = QComboBox()
        self.combo_y = QComboBox()
        self.combo_x.addItems(self.numeric_columns)
        self.combo_y.addItems(self.numeric_columns)
        self.combo_y.setCurrentIndex(1 if len(self.numeric_columns) > 1 else 0)

        self.axis_selection_layout.addWidget(QLabel("Axe X :"))
        self.axis_selection_layout.addWidget(self.combo_x)
        self.axis_selection_layout.addWidget(QLabel("Axe Y :"))
        self.axis_selection_layout.addWidget(self.combo_y)

        
        self.layout.addLayout(self.checkbox_layout)
        self.layout.addLayout(self.axis_selection_layout)

        
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        
        self.cb_clusters.stateChanged.connect(self.mettre_a_jour_graphe)
        self.cb_centroids.stateChanged.connect(self.mettre_a_jour_graphe)
        self.cb_medoids.stateChanged.connect(self.mettre_a_jour_graphe)
        self.combo_x.currentIndexChanged.connect(self.mettre_a_jour_graphe)
        self.combo_y.currentIndexChanged.connect(self.mettre_a_jour_graphe)

        self.mettre_a_jour_graphe()

    def mettre_a_jour_graphe(self):
        self.ax.clear()

        if "Cluster_Assigned" not in self.df.columns:
            print("La colonne 'Cluster_Assigned' est absente du DataFrame.")
            return

        x_col = self.combo_x.currentText()
        y_col = self.combo_y.currentText()

        if x_col == y_col:
            self.ax.set_title("Please select two different dimensions.")
            self.canvas.draw()
            return

        X = self.df[[x_col, y_col]]
        labels = self.df["Cluster_Assigned"]
        n_clusters = len(np.unique(labels))
        colors = plt.cm.get_cmap("tab10", n_clusters)

        
        self.ax.scatter(X[x_col], X[y_col], label="Data", color='gray', alpha=0.4)

        # affichage des clusters
        if self.cb_clusters.isChecked():
            for i in range(n_clusters):
                cluster_points = self.df[labels == i][[x_col, y_col]]
                self.ax.scatter(
                    cluster_points[x_col],
                    cluster_points[y_col],
                    label=f"Cluster {i}",
                    color=colors(i),
                    alpha=0.8
                )

        # centroides
        if self.cb_centroids.isChecked():
            centers = pd.DataFrame(self.kmeans.cluster_centers_, columns=self.df_filtred.columns[:self.kmeans.cluster_centers_.shape[1]])
            self.ax.scatter(
                centers[x_col],
                centers[y_col],
                c='black',
                marker='X',
                s=150,
                label="Centroids"
            )

        # medoides
        if self.cb_medoids.isChecked() and self.medoids_df is not None:
            self.ax.scatter(
                self.medoids_df[x_col],
                self.medoids_df[y_col],
                c='gold',
                marker='D',
                s=100,
                label="Medoids"
            )

        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.legend()
        self.ax.set_title("Clustering K-Means")
        self.canvas.draw()
