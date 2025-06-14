import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from scripts.evaluation.save_model import apply_pre_on_supplied_simple


def apply_em(self, supplied=None, test_size=None):
    start_time = time.time()
    summary = ""
    self.is_test_file = False

    # Sélection du DataFrame à utiliser
    df_to_use = (
        self.df_filtred
        if hasattr(self, "df_filtred") and self.df_filtred is not None
        else self.df
    )
    X_raw = df_to_use.values
    n_samples, n_features = X_raw.shape
    attributes = df_to_use.columns.tolist()

    # Standardisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    def _fit_one_k(k, X_data):
        """Apprend un GMM pour une valeur donnée de k et renvoie ses scores."""
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            tol=1e-5,
            reg_covar=1e-6,
            max_iter=500,
            n_init=5,
            init_params="kmeans",
            random_state=42,
        )
        gmm.fit(X_data)
        bic = gmm.bic(X_data)
        if k > 1:
            sil = silhouette_score(
                X_data,
                gmm.predict(X_data),
                metric="euclidean",
                sample_size=min(1000, len(X_data)),  # ⇢ accélération
                random_state=42,
            )
        else:
            sil = 0.0
        return {"k": k, "model": gmm, "bic": bic, "silhouette": sil}

    def find_optimal_clusters(X_data):
        max_feasible = min(50, int(np.sqrt(n_samples * 2)))

        # Apprentissage parallèle pour toutes les valeurs de k
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_fit_one_k)(k, X_data) for k in range(1, max_feasible + 1)
        )

        bic_scores = np.array([r["bic"] for r in results])
        sil_scores = np.array([r["silhouette"] for r in results])

        max_sil = sil_scores.max()
        candidate_idxs = [i for i, s in enumerate(sil_scores) if s == max_sil]

        best_idx = min(candidate_idxs, key=lambda i: bic_scores[i])
        best = results[best_idx]

        return (
            best["model"],
            best["k"],
            best["bic"],
            best["silhouette"],
        )

    execution_mode = ""

    if supplied is not None:
        execution_mode = "Test file provided"
        test_df = pd.read_csv(supplied)

        if self.pipeline_manager.has_preprocessing_steps():
            text, test_df = apply_pre_on_supplied_simple(
                self.pipeline_manager, test_df
            )
            summary = "Preprocess of Supplied File :\n" + text
        else:
            summary = "No Preprocess of the Supplied File\n"

        common_cols = [c for c in test_df.columns if c in df_to_use.columns]
        X_test = scaler.transform(test_df[common_cols].values)

        best_model, best_k, best_bic, best_sil = find_optimal_clusters(X)
        clusters = best_model.predict(X_test)

        test_df["Cluster"] = clusters
        new_df = test_df

    elif test_size is not None:
        execution_mode = f"Train-test split ({test_size*100:.0f}% test)"
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)

        best_model, best_k, best_bic, best_sil = find_optimal_clusters(X_train)
        clusters = best_model.predict(X_test)

        new_df = self.df_filtred.copy()
        train_idx, test_idx = train_test_split(
            self.df_filtred.index, test_size=test_size, random_state=42
        )
        new_df.loc[test_idx, "Cluster"] = clusters

    else:
        execution_mode = "Training on the complete dataset"
        best_model, best_k, best_bic, best_sil = find_optimal_clusters(X)
        clusters = best_model.predict(X)

        new_df = self.df_filtred.copy()
        new_df["Cluster"] = clusters

    exec_time = time.time() - start_time

    output_text = (
        "=== GAUSSIAN MIXTURE MODEL (EM) - COMPLETE REPORT ===\n\n"
        f"Execution mode: {execution_mode}\n"
        f"Execution time: {exec_time:.2f} seconds\n\n"
        f"{summary}"
        "=== CLUSTERS SUGGESTION ===\n\n"
        f"Optimal number of clusters: {best_k}\n\n"
        "=== PERFORMANCE ===\n"
        f"BIC: {best_bic:.2f}\n"
        f"Silhouette Score: {best_sil:.3f}\n\n"
    )

    output_text_second = "=== CLUSTER DISTRIBUTION ===\n\n"
    unique, counts = np.unique(clusters, return_counts=True)
    for cl, cnt in zip(unique, counts):
        output_text_second += (
            f"Cluster {cl}: {cnt} samples ({cnt/len(clusters)*100:.1f}%)\n"
        )
    output_text_second += "\n=== FEATURES PER CLUSTER ===\n"

    for cl in range(best_k):
        output_text_second += (
            f"\nCLUSTER {cl} (n={counts[cl]})\n"
            + "Attribute".ljust(25)
            + "Mean".center(15)
            + "Std Dev".center(15)
            + "\n"
            + "-" * 55
            + "\n"
        )
        for j, attr in enumerate(attributes):
            mean = best_model.means_[cl, j]
            std = np.sqrt(best_model.covariances_[cl][j, j])
            output_text_second += f"{attr.ljust(25)}{mean:^15.3f}{std:^15.3f}\n"

    self.em_model = best_model
    self.em_k = best_k

    output_final = output_text + output_text_second
    return output_text, output_final, best_model, clusters, new_df
