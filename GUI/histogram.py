import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog

class HistogramPlotter:
    def __init__(self, canvas, textStat):
        self.canvas = canvas
        self.textStat = textStat
        self.last_selected_column = None
        self.current_figure = None

    def display(self, column, df, df_filt):
        dataset = df_filt if column in df_filt.columns else df
        if dataset is None or column not in dataset.columns:
            self.clear_histogram()
            return

        self.last_selected_column = column
 
        self.canvas.figure.clf()

        ax = self.canvas.figure.add_subplot(111)

        if pd.api.types.is_numeric_dtype(dataset[column]):
            dtype = "Numeric"
            data = dataset[column].dropna()
            ax.hist(data, bins=30, color="#3C7E90", edgecolor="black")

            desc = data.describe().round(2)
            stats_text = (
                f"count : {desc['count']}\n"
                f"mean : {desc['mean']}\n"
                f"std : {desc['std']}\n"
                f"min : {desc['min']}\n"
                f"25% : {desc['25%']}\n"
                f"50% : {desc['50%']}\n"
                f"75% : {desc['75%']}\n"
                f"max : {desc['max']}"
            )
            ax.set_title(f"Histogram of {column}")
        else:
            dtype = "Categorical"
            counts = dataset[column].value_counts()
            ax.bar(counts.index.astype(str), counts.values, color="#3C7E90", edgecolor="black")

            mode = dataset[column].mode()[0]
            unique_count = dataset[column].nunique()
            top5 = counts.head()

            stats_text = (
                f"Mode : {mode}\n"
                f"Unique Count : {unique_count}\n"
                f"Top 5 Values :\n"
                + "\n".join([f"{k} : {v}" for k, v in top5.items()])
            )
            ax.set_title(f"Bar Chart of {column}")


        ax.grid(False)

        self.canvas.draw()

        self.current_figure = self.canvas.figure
        self.textStat.setText(f"Name : {column}\nType : {dtype}\n{stats_text}")

    def save_histogram(self):
        if self.current_figure:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save Histogram", "", "PNG Files (*.png)")
            if file_path:
                self.current_figure.savefig(file_path)
                print(f"Histogram saved to {file_path}")

    def clear_histogram(self):
        self.canvas.figure.clf()
        self.textStat.clear()
        self.current_figure = None
        self.last_selected_column = None
        self.canvas.draw()
