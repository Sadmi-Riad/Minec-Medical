import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


def show_confusion_matrix_dialog(confusion_matrix_data, parent=None):
    cm = np.asarray(confusion_matrix_data, dtype=int)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        QMessageBox.information(
            parent,
            "Confusion Matrix",
            f"The confusion matrix must be square (received shape: {cm.shape})."
        )
        return

    n = cm.shape[0]
    k = n // 2

    annot = np.empty_like(cm, dtype=object)
    for i in range(n):
        for j in range(n):
            if   i == k and j == k: label = "TP"
            elif i == k:            label = "FN"
            elif j == k:            label = "FP"
            else:                   label = "TN"
            annot[i, j] = f"{label}\n{cm[i, j]}"

    # Decorative checkerboard
    color_matrix = (np.indices((n, n)).sum(axis=0) % 2)
    cmap = ListedColormap(["#35586D", "#EE6843"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))
    sns.heatmap(
        color_matrix,
        annot=annot,
        fmt="",
        cmap=cmap,
        norm=norm,
        cbar=False,
        linewidths=1,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title(f"Confusion Matrix", fontsize=14, pad=12)
    ax.set_xticklabels([f"Class {j}" for j in range(n)], rotation=45, ha="right")
    ax.set_yticklabels([f"Class {i}" for i in range(n)], rotation=0)
    fig.tight_layout()

    # Display in a Qt dialog with navigation toolbar
    dialog = QDialog(parent)
    dialog.setWindowTitle("Confusion Matrix")
    layout = QVBoxLayout(dialog)

    # Create canvas and toolbar
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, dialog)

    # Add toolbar and canvas to the layout
    layout.addWidget(toolbar)
    layout.addWidget(canvas)

    dialog.exec_()
    plt.close(fig)
