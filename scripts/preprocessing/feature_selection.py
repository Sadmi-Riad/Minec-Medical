import pandas as pd
import numpy as np
from typing import Tuple, List, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from scripts.preprocessing.data_loader import get_target


def _log(msg: str, buff: List[str]):
    """Helper to append a line to a log buffer."""
    buff.append(msg)

def process_features(
    df: pd.DataFrame,
    *,
    max_cardinality_ratio: float = 0.5,
    fill_na: bool = True,
) -> Tuple[pd.DataFrame, str, str]:

    df = df.copy()  

    dropped_cols: List[str] = []
    used_cat_cols: List[str] = []
    log: List[str] = []
    suggestions: List[str] = []

    # Optionally fill NA beforehand so .nunique() is not affected
    if fill_na:
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols_all = df.select_dtypes(exclude=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df[cat_cols_all] = df[cat_cols_all].fillna("__NA__")
        
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    n_rows = len(df)

    for col in cat_cols:
        n_unique = df[col].nunique(dropna=False)
        if n_unique == 1:
            dropped_cols.append(col)
            _log(f"Dropped column '{col}' (only one unique value)", log)
        elif n_unique == n_rows :
            dropped_cols.append(col)
            _log(f"Dropped column '{col}' (all values unique)", log)
        elif n_unique > max_cardinality_ratio * n_rows:
            dropped_cols.append(col)
            _log(
                f"Dropped column '{col}' (high cardinality: {n_unique} unique values)",
                log,
            )
            suggestions.append(
                f"Review high‑cardinality column '{col}' — consider hashing or target encoding if it contains signal."
            )
        else:
            used_cat_cols.append(col)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        n_unique = df[col].nunique(dropna=False)
        if n_unique == 1:
            dropped_cols.append(col)
            _log(f"Dropped numeric column '{col}' (only one unique value)", log)
        elif n_unique == n_rows and pd.api.types.is_integer_dtype(df[col]):
            dropped_cols.append(col)
            _log(f"Dropped numeric column '{col}' (all values unique)", log)

    for col in used_cat_cols:
        n_unique = df[col].nunique(dropna=False)
        if n_unique == 2:
            df[col] = df[col].astype("category").cat.codes  # 0/1 encoding
            _log(f"Binary‑encoded column '{col}' (0/1)", log)
        else:
            freq_map = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq_map)
            _log(f"Frequency‑encoded column '{col}'", log)

    keep_cols = [c for c in df.columns if c not in dropped_cols]
    df_processed = df[keep_cols].copy()
    suggestions.append(
        f"Dropped columns ({len(dropped_cols)}): {', '.join(dropped_cols) if dropped_cols else 'None'}"
    )

    return df_processed, "\n".join(log), "\n".join(suggestions)



def auto_detect_problem_type(y: Union[pd.Series, np.ndarray]) -> Tuple[str, str]:
    y_ser = pd.Series(y)

    # If target is non‑numeric → classification straight away
    if not pd.api.types.is_numeric_dtype(y_ser):
        return "classification", "Detected classification problem (target is non‑numeric)"

    n_unique = y_ser.nunique(dropna=False)
    unique_ratio = n_unique / len(y_ser)

    # A *small* number of unique *integer* values likely means classification.
    if pd.api.types.is_integer_dtype(y_ser) and n_unique <= 20:
        return "classification", f"Detected classification problem ({n_unique} classes)"

    # Otherwise treat as regression
    return "regression", f"Detected regression problem ({n_unique} unique numeric values)"

def feature_selection(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray, None] = None,
    *,
    max_features: int | None = None,
) -> Tuple[np.ndarray, str, str]:

    full_log: List[str] = []
    suggestions: List[str] = []

    X_proc, log1, sugg1 = process_features(X)
    _log("=== STEP 1: Feature processing ===", full_log)
    full_log.append(log1)
    suggestions.append(sugg1)

    if y is not None:
        _log("\n=== STEP 2: Supervised selection ===", full_log)
        task, msg = auto_detect_problem_type(y)
        full_log.append(msg)

        Estimator = RandomForestClassifier if task == "classification" else RandomForestRegressor
        model = Estimator(n_estimators=500, random_state=42, n_jobs=-1)
        full_log.append(f"Using {model.__class__.__name__} for feature importances\n")

        model.fit(X_proc, y)
        importances = model.feature_importances_

        sorted_pairs = sorted(zip(X_proc.columns, importances), key=lambda t: t[1], reverse=True)
        median_imp = np.median(importances)
        thresh = 0.75 * median_imp

        selected = [f for f, imp in sorted_pairs if imp >= thresh]
        if not selected:
            k = max(1, len(sorted_pairs) // 3)
            selected = [f for f, _ in sorted_pairs[:k]]
            full_log.append(
                f"Threshold left no features ⇒ keeping top‑{k} by importance instead."
            )
        if max_features is not None:
            selected = selected[:max_features]

        full_log.append("Feature importance ranking (desc):")
        for idx, (f, imp) in enumerate(sorted_pairs, 1):
            full_log.append(f"  {idx:2d}. {f:<30}  {imp:.4f}")

        # Suggestions
        suggestions.append("=== Suggestions: Supervised selection ===")
        suggestions.append(f"Cut‑off importance threshold = {thresh:.4f}")
        suggestions.append(f"Selected features ({len(selected)}):")
        suggestions.extend([f"   • {f}" for f in selected])
        low_imp = [f for f, imp in sorted_pairs if f not in selected]
        if low_imp:
            suggestions.append("Consider dropping low‑importance features:")
            suggestions.extend([f"   • {f}" for f in low_imp])

        X_sel = X_proc[selected].values

    else:
        _log("\n=== STEP 2: Unsupervised selection ===", full_log)

        # 2A. Variance filter
        variances = X_proc.var(axis=0).values
        variances_series = pd.Series(variances, index=X_proc.columns)
        sorted_var_pairs = variances_series.sort_values(ascending=False)

        var_thresh = 0.5 * np.median(variances)
        var_selector = VarianceThreshold(threshold=var_thresh)
        X_var = var_selector.fit_transform(X_proc)
        kept = X_proc.columns[var_selector.get_support()].tolist()

        full_log.append(
            f"Retained {len(kept)}/{X_proc.shape[1]} features after variance threshold ({var_thresh:.4f})"
        )

        # Variance ranking output
        full_log.append("Variance ranking (desc):")
        for idx, (f, var) in enumerate(sorted_var_pairs.items(), 1):
            full_log.append(f"  {idx:2d}. {f:<30}  {var:.4f}")

        # 2B. Correlation filter among retained
        corr = pd.DataFrame(X_var, columns=kept).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

        final_keep = [c for c in kept if c not in to_drop]
        if max_features is not None:
            final_keep = final_keep[:max_features]

        full_log.append(
            f"Dropped {len(to_drop)} highly‑correlated (>0.95) features → {len(final_keep)} total kept"
        )

        suggestions.append("=== Suggestions: Unsupervised selection ===")
        suggestions.append("High‑variance features kept after threshold:")
        suggestions.extend([f"   • {f}" for f in kept])
        if to_drop:
            suggestions.append("Highly‑correlated features (>0.95) you may drop:")
            suggestions.extend([f"   • {f}" for f in to_drop])
        else:
            suggestions.append("No pairs with correlation > 0.95 detected.")

        X_sel = pd.DataFrame(X_var, columns=kept)[final_keep].values

    return X_sel, "\n".join(full_log), "\n".join(suggestions)


def apply_feature_selection(self):
    try:
        y, X = get_target(self, 2)  
        X_selected, log, sugg = feature_selection(X, y)
        return log, sugg
    except Exception as exc:  # pragma: no cover
        return f"Error during feature selection: {exc}", "No suggestions available due to error"
