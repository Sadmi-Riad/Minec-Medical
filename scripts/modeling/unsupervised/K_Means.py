import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from scripts.evaluation.save_model import apply_pre_on_supplied_simple
from scipy.spatial.distance import cdist

def apply_kmean(self, n_clusters, supplied=None, test_size=None):
    X = self.df_filtred.select_dtypes(include=["number"])
    summary = ""
    if supplied:
        try:
            self.is_test_file=True
            df_supplied=pd.read_csv(supplied)
            if self.pipeline_manager.has_preprocessing_steps():
                text , df_supplied = apply_pre_on_supplied_simple(self.pipeline_manager , df_supplied,self.df)
                df_supplied = df_supplied[self.df_filtred.columns]
                summary = "Preprocess of Supplied File : \n" + text
            else : 
                df_supplied = df_supplied[self.df_filtred.columns]
                summary="No Preprocess of the Supplied File \n"
            numeric_cols = [col for col in self.df_filtred.columns if pd.api.types.is_numeric_dtype(self.df_filtred[col])]
            numeric_test_cols = [col for col in numeric_cols if col in df_supplied.columns]
            df_supplied=df_supplied[numeric_test_cols]
            X_train = X
            X_test = df_supplied
        except Exception as e:
            raise ValueError(f"Error loading the supplied file: {e}")
    elif test_size is not None:
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
    else:
        X_train = X
        X_test = None 

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=50, random_state=42)
    kmeans.fit(X_train)

    if n_clusters > 1:
        sil_score = silhouette_score(X_train, kmeans.labels_)
    else:
        sil_score = float('nan')
    inertia_value = kmeans.inertia_
    #labeling new data
    new_df = self.df.copy()
    new_df["Cluster_Assigned"] = np.nan
    new_df.loc[X_train.index, "Cluster_Assigned"] = kmeans.labels_
    #labeling main df
    new_df_touched=self.never_touch.copy()
    new_df_touched["Cluster_Assigned"] = np.nan
    new_df_touched.loc[X_train.index, "Cluster_Assigned"] = kmeans.labels_
    #labeling filtered df 
    new_filtered =self.df_filtred.copy()
    new_filtered["Cluster_Assigned"]=np.nan
    new_filtered.loc[X_train.index, "Cluster_Assigned"] = kmeans.labels_
    
    if X_test is not None:
        test_labels = kmeans.predict(X_test)
        new_df.loc[X_test.index, "Cluster_Assigned"] = test_labels
        new_df_touched.loc[X_test.index, "Cluster_Assigned"] = test_labels
        new_filtered.loc[X_test.index, "Cluster_Assigned"] = test_labels
    

    
    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=X_train.columns)

    medoids = []
    medoids_indices = []
    for i in range(n_clusters):
        centroid = kmeans.cluster_centers_[i]
        distances = cdist([centroid], X_train)[0]
        closest_index = np.argmin(distances)
        medoids.append(X_train.iloc[closest_index])
        medoids_indices.append(X_train.index[closest_index])
    medoids_df = pd.DataFrame(medoids, columns=X_train.columns, index=medoids_indices)

    output_text = "==== K-Means Clustering Results ====\n"
    output_text += summary
    output_text += f"Silhouette Score: {sil_score:.3f}\n"
    output_text += f"Inertia: {inertia_value:.3f}\n\n"
    output_text += f"Total number of training samples: {X_train.shape[0]}\n"
    output_text += f"Number of features: {X_train.shape[1]}\n"
    output_text += f"Number of clusters: {n_clusters}\n\n"
    output_text_second = "==== Cluster Centers (Centroids) ====\n"
    for i, row in enumerate(centers_df.itertuples(index=False), 1):
        output_text_second += f"Centroid {i}:\n"
        for col, val in zip(centers_df.columns, row):
            output_text_second += f"  {col}: {val:.3f}\n"
        output_text_second += "\n"
    output_text_second += "==== Cluster Medoids (Closest Points to Centroids) ====\n"
    for i, (idx, row) in enumerate(medoids_df.iterrows(), 1):
        output_text_second += f"Medoid {i} (Row Number: {idx}):\n"
        for col in medoids_df.columns:
            output_text_second += f"  {col}: {row[col]:.3f}\n"
        output_text_second += "\n"
    output_text_second += "==== End of Results ====\n"
    output_final = output_text + output_text_second
    self.kmeans_model = kmeans
    self.medoids_df = medoids_df

    return output_text + "\n==== End of Results ====\n" ,output_final, kmeans, new_df, medoids_df , new_df_touched ,new_filtered
