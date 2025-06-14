from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QMainWindow, QSlider, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QByteArray
from sklearn.tree import export_graphviz
import graphviz
import tempfile
import os

class DecisionTreeWindow(QMainWindow):
    def __init__(self, model, feature_names, class_names):
        super().__init__()
        self.setWindowTitle("Arbre de Décision")

        # Generer l image de l arbre
        self.tree_pixmap = self.generate_tree_image(model, feature_names, class_names)

        
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)

        self.pixmap_item = QGraphicsPixmapItem(self.tree_pixmap)
        self.scene.addItem(self.pixmap_item)

        # Creer un curseur pour zoomer
        zoom_slider = QSlider(Qt.Horizontal)
        zoom_slider.setRange(1, 200)  
        zoom_slider.setValue(100)  
        zoom_slider.valueChanged.connect(self.update_zoom)

        
        layout = QVBoxLayout()
        layout.addWidget(zoom_slider)

        container = QWidget()
        container.setLayout(layout)
        self.setMenuWidget(container)

    def update_zoom(self, value):
        zoom_factor = value / 100
        self.view.resetTransform()  
        self.view.scale(zoom_factor, zoom_factor)

    def generate_tree_image(self, model, feature_names, class_names):
       
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dot") as dot_file:
            export_graphviz(
                model,
                out_file=dot_file.name,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                special_characters=True,
                impurity=False
            )

        
        graph = graphviz.Source.from_file(dot_file.name)
        image_data = graph.pipe(format='png')  
        
        os.unlink(dot_file.name)  

        
        pixmap = QPixmap()
        pixmap.loadFromData(QByteArray(image_data))
        return pixmap
