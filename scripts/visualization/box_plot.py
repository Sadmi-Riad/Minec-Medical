import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

def show_boxplot(df, parent=None):
    # Select only numeric columns, show message box if none found
    numeric_df = df.select_dtypes(include=[float, int])
    if numeric_df.empty:
        QMessageBox.warning(
            parent,
            "No Numeric Columns",
            "The DataFrame contains no numeric columns."
        )
        return

    # Set up a custom colormap using your two specified colors
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_palette", ["#35586D", "#EE6843"], N=256
    )
    n_vars = len(numeric_df.columns)
    # Use the first color for box fill, second for median, outliers, etc.
    box_color = "#35586D"
    highlight_color = "#EE6843"
    palette = [box_color] * n_vars

    # Figure size adjusts to the number of columns
    figsize = (max(10, n_vars * 1.1), 6.5)

    # Set seaborn theme
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.18)

    fig, ax = plt.subplots(figsize=figsize, dpi=120, facecolor="#fff")

    # Draw the boxplot
    sns.boxplot(
        data=numeric_df,
        palette=palette,
        ax=ax,
        orient="v",
        linewidth=1.2,
        fliersize=4,
        boxprops=dict(edgecolor=box_color, facecolor="#f9f9f9", linewidth=1.2),
        medianprops=dict(color=highlight_color, linewidth=2.2),
        whiskerprops=dict(color=box_color, linewidth=1.1),
        capprops=dict(color=box_color, linewidth=1.1),
        flierprops=dict(markerfacecolor=highlight_color, markeredgecolor=box_color, markersize=5, marker="o"),
    )

    # Title and labels
    ax.set_title(
        "Boxplot Diagram",
        fontsize=20,
        fontweight="bold",
        color="#2c2c2c",
        pad=25,
        family="Arial",
    )
    ax.set_ylabel(
        "Values",
        fontsize=14,
        color="#666666",
        family="Arial",
        labelpad=15,
    )

    # X labels styling (use plt.setp to avoid warnings)
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        color="#555555",
        fontsize=12,
        family="Arial",
    )

    # Y tick styling
    ax.tick_params(axis='y', colors='#777777', labelsize=11)

    # Grid styling
    ax.grid(axis='y', color='#dedede', linestyle='--', linewidth=0.7)
    ax.grid(axis='x', visible=False)

    # Remove top/right borders for a clean look
    sns.despine(ax=ax)

    # Extra margin for x labels
    plt.subplots_adjust(bottom=0.3)

    # Create PyQt5 dialog window with modern style
    dialog = QDialog(parent)
    dialog.setWindowTitle("Boxplot Diagram")
    dialog.setMinimumSize(700, 550)
    dialog.setStyleSheet("""
        QDialog {
            background: #fff;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
        }
    """)

    # Layout
    layout = QVBoxLayout(dialog)
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, dialog)
    layout.addWidget(toolbar)
    layout.addWidget(canvas)

    # Show the dialog
    dialog.exec_()
    plt.close(fig)