import numpy as np
import pandas as pd
from sklearn.cluster import SpectralBiclustering
from sklearn.model_selection import train_test_split
from scripts.evaluation.save_model import apply_pre_on_supplied_simple

def apply_biclustering(self, n_clusters=3, supplied=None, test_size=None):
    X = self.df_filtred.select_dtypes(include=["number"])
    summary = ""
    if supplied:
        try:
            self.is_test_file = True
            df_supplied = pd.read_csv(supplied)
            if self.pipeline_manager.has_preprocessing_steps():
                text, df_supplied = apply_pre_on_supplied_simple(self.pipeline_manager, df_supplied, self.df)
                df_supplied = df_supplied[self.df_filtred.columns]
                summary = "Preprocess of Supplied File : \n" + text
            else:
                df_supplied = df_supplied[self.df_filtred.columns]
                summary = "No Preprocess of the Supplied File\n"
            numeric_cols = [col for col in self.df_filtred.columns if pd.api.types.is_numeric_dtype(self.df_filtred[col])]
            numeric_test_cols = [col for col in numeric_cols if col in df_supplied.columns]
            df_supplied = df_supplied[numeric_test_cols]
            X_train = X
            X_test = df_supplied
        except Exception as e:
            raise ValueError(f"Error loading the supplied file: {e}")
    elif test_size is not None:
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
    else:
        X_train = X
        X_test = None

    bicluster = SpectralBiclustering(n_clusters=n_clusters, random_state=42)
    bicluster.fit(X_train)
    row_labels = bicluster.row_labels_
    col_labels = bicluster.column_labels_

    output_text = "==== Spectral Bi-Clustering Results ====\n"
    output_text += summary
    output_text += f"Total number of training samples: {X_train.shape[0]}\n"
    output_text += f"Number of features: {X_train.shape[1]}\n"
    output_text += f"Number of clusters: {n_clusters}\n\n"
    output_text_second = "Row cluster assignments (first 20 rows):\n"
    output_text_second += ", ".join(str(x) for x in row_labels[:20]) + "\n"
    output_text_second += "Column cluster assignments:\n"
    output_text_second += ", ".join(str(x) for x in col_labels) + "\n"
    output_text_second += "==== End of Results ====\n"

    # No centroids/medoids for biclustering
    medoids_df = None

    self.biclustering_model = bicluster

    return output_text + output_text_second, output_text + output_text_second, bicluster, None, medoids_df, None, None