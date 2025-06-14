from PyQt5.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QMainWindow,
    QSlider, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QByteArray
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import io


class DecisionTreeWindow(QMainWindow):
    def __init__(self, model, feature_names, class_names):
        super().__init__()
        self.setWindowTitle("Decision Tree")

        self.tree_pixmap = self.generate_tree_image(model, feature_names, class_names)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)

        self.pixmap_item = QGraphicsPixmapItem(self.tree_pixmap)
        self.scene.addItem(self.pixmap_item)

        # Zoom slider
        zoom_slider = QSlider(Qt.Horizontal)
        zoom_slider.setRange(1, 300)
        zoom_slider.setValue(10)
        zoom_slider.valueChanged.connect(self.update_zoom)

        save_button = QPushButton("Save Tree")
        save_button.clicked.connect(self.save_tree_image)

        layout = QHBoxLayout()
        layout.addWidget(zoom_slider)
        layout.addWidget(save_button)

        container = QWidget()
        container.setLayout(layout)
        self.setMenuWidget(container)

        self.view.setTransform(self.view.transform().scale(0.1, 0.1))

    def update_zoom(self, value):
        zoom_factor = value / 100
        self.view.resetTransform()
        self.view.scale(zoom_factor, zoom_factor)

    def save_tree_image(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Tree Image", "", "PNG Images (*.png)")
        if filename:
            self.tree_pixmap.save(filename, "PNG")

    def generate_tree_image(self, model, feature_names, class_names):
        buf = io.BytesIO()

        plt.figure(figsize=(30, 15), dpi=300)

        plot_tree(
            model,
            feature_names=feature_names,
            class_names=model.classes_.astype(str),
            filled=True,
            rounded=True,
            fontsize=6,
            impurity=False,
            label="all",
            proportion=False
        )

        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=300)
        plt.close()
        buf.seek(0)

        image_data = buf.read()
        pixmap = QPixmap()
        pixmap.loadFromData(QByteArray(image_data))
        return pixmap
