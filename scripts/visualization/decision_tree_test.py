import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from PyQt5.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


def show_decision_tree(model, feature_names, class_names, parent=None):

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        filled=True,
        rounded=True,
        fontsize=10,
        impurity=False,
        label="all",
        proportion=False,
        ax=ax
    )
    
    fig.tight_layout()

    dialog = QDialog(parent)
    dialog.setWindowTitle("Decision Tree")
    dialog.resize(1200, 800)
    layout = QVBoxLayout(dialog)

    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, dialog)

    layout.addWidget(toolbar)
    layout.addWidget(canvas)

    dialog.exec_()
    plt.close(fig)
