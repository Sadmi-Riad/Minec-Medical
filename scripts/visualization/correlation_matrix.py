import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from PyQt5.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

def show_correlation_matrix(df, parent=None):
    numeric_df = df.select_dtypes(include=[float, int])
    
    correlation_matrix = numeric_df.corr()
    
    custom_cmap = LinearSegmentedColormap.from_list("custom", ["#35586D", "#EE6843"], N=256)
    
    n_vars = len(correlation_matrix.columns)
    base_size = 8  
    figsize = (max(8, n_vars * 0.8), max(6, n_vars * 0.7))  
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap=custom_cmap, 
                fmt='.2f', 
                linewidths=0.5, 
                cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                annot_kws={"size": 9 if n_vars <= 15 else 7}, 
                vmin=-1, 
                vmax=1,
                square=True,  
                ax=ax)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    plt.subplots_adjust(bottom=0.25, left=0.25)
    
    ax.set_title('Correlation Matrix', fontsize=14, pad=20)
    
    dialog = QDialog(parent)
    dialog.setWindowTitle("Correlation Matrix")
    layout = QVBoxLayout(dialog)
    
    dialog.setMinimumSize(600, 500)
    
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, dialog)
    
    layout.addWidget(toolbar)
    layout.addWidget(canvas)
    
    dialog.exec_()
    plt.close(fig)