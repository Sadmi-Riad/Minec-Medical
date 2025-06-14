import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scripts.evaluation.save_model import apply_pre_on_supplied_simple

def estimate_dbscan_params(X, min_samples=None):
    n_features = X.shape[1]
    # Automate min_samples if not provided: choose the greater between (2 * n_features) and 5
    min_samples = min_samples or max(2 * n_features, 5)
    # Automate eps using the k-distance method (90th percentile)
    neigh = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    eps = np.percentile(k_distances, 90)
    return eps, min_samples

def apply_dbscan(self, eps=None, min_samples=None, supplied=None, test_size=None):
    X = self.df_filtred.select_dtypes(include=["number"])
    summary = ""
    
    # Data preparation and extraction
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
            numeric_cols = [
                col for col in self.df_filtred.columns
                if pd.api.types.is_numeric_dtype(self.df_filtred[col])
            ]
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

    # Scale training data for better DBSCAN performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Automate parameter selection if necessary using scaled data
    if eps is None or min_samples is None:
        eps, min_samples = estimate_dbscan_params(X_train_scaled, min_samples)
    
    # Create and fit the DBSCAN model with parallel computation enabled (if available)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    dbscan.fit(X_train_scaled)
    labels = dbscan.labels_
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Compute silhouette score if possible
    if n_clusters_found > 1 and np.sum(labels != -1) > 0:
        sil_score = silhouette_score(X_train_scaled[labels != -1], labels[labels != -1])
    else:
        sil_score = float('nan')
    
    # Compose output summary text
    output_text = "==== DBSCAN Clustering Results ====\n"
    output_text += summary
    output_text += f"Parameters used: eps={eps:.3f}, min_samples={min_samples}\n"
    output_text += f"Silhouette Score: {sil_score:.3f}\n"
    output_text += f"Total number of training samples: {X_train.shape[0]}\n"
    output_text += f"Number of features: {X_train.shape[1]}\n"
    output_text += f"Number of clusters found: {n_clusters_found}\n"
    noise_points = np.sum(labels == -1)
    output_text += f"Number of noise points: {noise_points}\n\n"
    output_text_second = "==== End of Results ====\n"
    
    # Assign labels to DataFrames preserving original indices
    new_df = self.df.copy()
    new_df["Cluster_Assigned"] = np.nan
    new_df.loc[X_train.index, "Cluster_Assigned"] = labels
    new_df_touched = self.never_touch.copy()
    new_df_touched["Cluster_Assigned"] = np.nan
    new_df_touched.loc[X_train.index, "Cluster_Assigned"] = labels
    new_filtered = self.df_filtred.copy()
    new_filtered["Cluster_Assigned"] = np.nan
    new_filtered.loc[X_train.index, "Cluster_Assigned"] = labels
    
    # No centroids/medoids are computed for DBSCAN
    medoids_df = None
    
    self.dbscan_model = dbscan
    
    return (
        output_text + output_text_second,
        output_text + output_text_second,
        dbscan,
        new_df,
        medoids_df,
        new_df_touched,
        new_filtered,
    )