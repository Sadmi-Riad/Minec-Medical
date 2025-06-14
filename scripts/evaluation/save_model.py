from __future__ import annotations

import os
from typing import List, Tuple, Optional, Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import clone
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class _PlainLogger:
    """Collects lines and provides helpers for pretty printing."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    # ------------------------------------------------------------------
    def section(self, title: str) -> None:
        bar = "=" * len(title)
        self.lines.extend([bar, title, bar])

    def line(self, msg: str = "") -> None:
        self.lines.append(msg)

    def dump(self) -> str:  # noqa: D401
        """Return the full concatenated log."""
        return "\n".join(self.lines)

    # Convenience -------------------------------------------------------
    @staticmethod
    def fmt_cols(cols: Sequence[str]) -> str:
        return "[" + ", ".join(cols) + "]"


class PipelineManager:
    """Holds preprocessing steps, model and immutable feature order."""

    def __init__(self) -> None:  # noqa: D401
        self.preprocessing_steps: List[Tuple[str, Any, List[str]]] = []
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None  # order expected by model

    # -------------------------- Mutators ----------------------------------
    def add_preprocessing_step(self, name: str, transformer: Any, features: List[str]) -> None:
        self.preprocessing_steps.append((name, transformer, features))

    def remove_last_preprocessing_step(self) -> None:
        if self.preprocessing_steps:
            self.preprocessing_steps.pop()

    def has_preprocessing_steps(self) -> bool:   
        return len(self.preprocessing_steps) > 0

    def set_model(self, model: Any) -> None:
        self.model = model

    def set_feature_names(self, feature_names: List[str]) -> None:
        """Store order **deduplicated** at first call only."""
        if self.feature_names is None:
            seen: set[str] = set()
            self.feature_names = [f for f in feature_names if not (f in seen or seen.add(f))]

    # ------------------------ Persistence ---------------------------------
    def save_pipeline(self, path: str) -> None:
        joblib.dump(
            {
                "preprocessing_steps": self.preprocessing_steps,
                "model": self.model,
                "feature_names": self.feature_names,
            },
            path,
        )

    @classmethod
    def load_pipeline(cls, path: str) -> "PipelineManager":
        data = joblib.load(path)
        mgr = cls()
        mgr.preprocessing_steps = data["preprocessing_steps"]
        mgr.model = data["model"]
        mgr.feature_names = data.get("feature_names")
        return mgr


def save_entire_pipeline(pipeline_manager: PipelineManager, parent=None) -> None:  # noqa: D401
    if pipeline_manager is None:
        QMessageBox.warning(parent, "Warning", "There is no pipeline to save.")
        return
    path, _ = QFileDialog.getSaveFileName(
        parent,
        "Save entire pipeline",
        os.path.expanduser("~"),
        "Joblib files (*.joblib *.jl);;All files (*)",
    )
    if not path:
        return
    if not (path.endswith(".joblib") or path.endswith(".jl")):
        path += ".joblib"
    try:
        joblib.dump(pipeline_manager, path)
        QMessageBox.information(parent, "Success", f"Pipeline saved to:\n{path}")
    except Exception as exc:  # pragma: no cover
        QMessageBox.critical(parent, "Error", f"Could not save pipeline:\n{exc}")


def load_entire_pipeline(parent=None) -> Optional[PipelineManager]:  # noqa: D401
    path, _ = QFileDialog.getOpenFileName(
        parent,
        "Load pipeline",
        os.path.expanduser("~"),
        "Joblib files (*.joblib *.jl);;All files (*)",
    )
    if not path:
        return None
    try:
        pm: PipelineManager = joblib.load(path)
        QMessageBox.information(parent, "Success", f"Pipeline loaded from:\n{path}")
        return pm
    except Exception as exc:  # pragma: no cover
        QMessageBox.critical(parent, "Error", f"Could not load pipeline:\n{exc}")
        return None


def _reorder_df(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    return df.loc[:, [c for c in order if c in df.columns]]

def apply_pipeline_and_predict(
    pipeline_manager: PipelineManager,
    df: pd.DataFrame,
    *,
    detailed_log: bool = True,
    logger: Optional[_PlainLogger] = None,
):
    """Apply stored preprocessing + model and return (log, predictions df)."""
    boolean =False
    log = logger or _PlainLogger()

    if pipeline_manager is None or pipeline_manager.model is None:
        log.line("Error: empty pipeline or missing model.")
        return log.dump(), None
    if df is None or df.empty:
        log.line("Error: input DataFrame is empty.")
        return log.dump(), None

    work = df.copy()

    if hasattr(pipeline_manager.model, "feature_names_in_"):
        required = list(pipeline_manager.model.feature_names_in_)
    else:
        if pipeline_manager.feature_names is None:
            pipeline_manager.set_feature_names(list(df.columns))
        required = pipeline_manager.feature_names  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Apply preprocessing steps
    # ------------------------------------------------------------------
    log.section("Preprocessing Steps")
    step_idx = 1

    for name, transformer, feats in pipeline_manager.preprocessing_steps:
        # Determine columns to process
        cols_full = list(getattr(transformer, "feature_names_in_", [])) or feats
        cols_present = [c for c in cols_full if c in work.columns]
        missing = [c for c in cols_full if c not in work.columns]
        try:
            # ------------------ KNN missing-value imputation -------------------
            if name.startswith("missing_values_knn_2"):
                # Impute ALL columns in the dataframe
                all_cols = list(work.columns)
                num_cols = [col for col in all_cols if work[col].dtype.kind in 'iufc']
                cat_cols = [
                    col for col in all_cols
                    if work[col].dtype == "object" or isinstance(work[col].dtype, pd.CategoricalDtype)
                ]
                df_impute = work[all_cols].copy()

                encoder = None
                if cat_cols:
                    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                    df_impute[cat_cols] = encoder.fit_transform(df_impute[cat_cols])

                imputer = transformer
                imputer.fit(df_impute)
                imputed_array = imputer.transform(df_impute)
                imputed_df = pd.DataFrame(imputed_array, columns=all_cols, index=work.index)

                # Remettre les valeurs dans work
                for col in num_cols:
                    work[col] = imputed_df[col].round(2)
                if cat_cols and encoder is not None:
                    imputed_df[cat_cols] = imputed_df[cat_cols].round(0)
                    work[cat_cols] = encoder.inverse_transform(imputed_df[cat_cols])

                # Vérification
                if work.isnull().any().any():
                    n_missing = work.isnull().sum().sum()        
                    log.line(f"{step_idx}) Warning: {n_missing} NaN values remain after KNN imputation!")
                else:
                    log.line(f"{step_idx}) KNN imputation applied to ALL columns ({len(all_cols)} columns). No NaN values remain.")
                if missing:
                    log.line(f"Absent cols at fit time: {missing}")
            # ------------------ Numeric missing-value imputation -------------------
            elif name.startswith("missing_values"):
                imputer = transformer
                num_cols = work.select_dtypes(include=[np.number]).columns
                cols_with_missing = [c for c in num_cols if work[c].isnull().any()]
                if not cols_with_missing:
                    log.line(f"{step_idx}) No missing values found in numeric columns → skipping imputation")
                else:
                    try:
                        imputed = imputer.transform(work[num_cols])
                    except ValueError:
                        imputer = clone(transformer)
                        imputer.fit(work[num_cols])
                        imputed = imputer.transform(work[num_cols])
                    work[num_cols] = imputed
                    msg = (
                        f"{step_idx}) Missing-value imputation ({name}) "
                        f"on columns: {', '.join(cols_with_missing)}"
                    )
                    if missing:
                        msg += f" – absent cols at fit time: {missing}"
                    log.line(msg)

            # ------------------ Normalization / Min-max --------------------
            elif name in {"normalization", "minmax_normalization"}:
                scaler = transformer
                if missing:
                    scaler = clone(transformer)
                    scaler.fit(work[cols_present])
                work[cols_present] = scaler.transform(work[cols_present]).round(3)
                msg = f"{step_idx}) Min-max normalization on {len(cols_present)} col(s): {_PlainLogger.fmt_cols(cols_present)}"
                if missing:
                    msg += f" – absent cols: {missing}"
                log.line(msg)

            # ------------------ Label encoding -----------------------------
            elif name == "label_encoding":
                for col, enc in transformer.items():
                    if col not in work.columns:
                        continue
                    mask = work[col].notnull()  # Ensure that NaN values are ignored
                    if not mask.any():
                        continue  # skip if all values missing
                    try:
                        work.loc[mask, col] = enc.transform(work.loc[mask, col])
                    except ValueError as err:
                        if "unseen" in str(err) or "previously unseen" in str(err):
                            unseen = np.setdiff1d(work.loc[mask, col].unique(), enc.classes_)
                            enc.classes_ = np.concatenate([enc.classes_, unseen])
                            work.loc[mask, col] = enc.transform(work.loc[mask, col])
                        else:
                            raise
                msg = f"{step_idx}) Label encoding on {len(cols_present)} col(s): {_PlainLogger.fmt_cols(cols_present)}"
                if missing:
                    msg += f" – absent cols: {missing}"
                log.line(msg)

            # ------------------ IQR outlier handling -----------------------
            elif name == "iqr_outlier_handling":
                handled = []
                for col in cols_present:
                    bounds = transformer[col]
                    work[col] = work[col].clip(bounds["lower"], bounds["upper"])
                    handled.append(col)
                msg = f"{step_idx}) IQR clipping on {len(handled)} col(s): {_PlainLogger.fmt_cols(handled)}"
                if missing:
                    msg += f" – config also contained absent cols: {missing}"
                log.line(msg)

            # --------------- Isolation Forest outlier handling -------------
            elif name == "isolation_forest_outlier_handling":
                treated = []
                for col in cols_present:
                    mdl = transformer[col]
                    preds = mdl.predict(work[[col]])
                    work.loc[preds == -1, col] = work[col].median()
                    treated.append(col)
                msg = f"{step_idx}) Isolation Forest outliers handled on {len(treated)} col(s): {_PlainLogger.fmt_cols(treated)}"
                if missing:
                    msg += f" – absent cols: {missing}"
                log.line(msg)
            # ------------ One-hot encoding for upload model ---------------
            elif name == "one_hot_encoding":
                processed = []
                for col, ohe in transformer.items():
                    if col not in work.columns:
                        continue
                    # transform and build dummies
                    arr = ohe.transform(work[[col]])
                    cats = ohe.categories_[0]
                    dummy_cols = [f"{col}_{cat}" for cat in cats]
                    df_dummies = pd.DataFrame(arr, columns=dummy_cols, index=work.index)
                    work = pd.concat([work.drop(columns=[col]), df_dummies], axis=1)
                    arr_full = ohe.transform(df[[col]])
                    df_dummies_full = pd.DataFrame(arr_full, columns=dummy_cols, index=df.index)
                    pre_df = pd.concat([df.drop(columns=[col]), df_dummies_full], axis=1)
                    processed.append(col)
                    boolean = True
                msg = f"{step_idx}) One-hot encoding on {len(processed)} col(s): {_PlainLogger.fmt_cols(processed)}"
                if missing:
                    msg += f" – absent cols: {missing}"
                log.line(msg)

            # ------------------ Unknown step -------------------------------
            else:
                log.line(f"{step_idx}) Unknown step '{name}' – ignored.")

        except Exception as exc:
            log.line(f"{step_idx}) Step '{name}' failed: {exc}")
        finally:
            log.line()
            step_idx += 1

    # ------------------------------------------------------------------
    # Align column order for model prediction
    work = _reorder_df(work, required)
    miss_final = [c for c in required if c not in work.columns]
    if miss_final:
        log.line("Error: missing required features " + ", ".join(miss_final))
        return log.dump(), None

    # ------------------------------------------------------------------
    # Prediction
    log.section("Prediction")
    try:
        preds = pipeline_manager.model.predict(work)
        work["Predictions"] = preds
        df = pre_df
        df["Predictions"] = preds
        log.line(f"Generated predictions for {len(preds)} rows.")
    except Exception as exc:
        log.line(f"Prediction error: {exc}")
        return log.dump(), None
    if boolean == True :
        return log.dump() if detailed_log else "Prediction complete.", work , df
    else :
        return log.dump() if detailed_log else "Prediction complete.", work

def apply_pre_on_supplied_simple(pipeline_manager: Any, df: pd.DataFrame , main_df : pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    if pipeline_manager is None:
        raise ValueError("pipeline_manager is required")
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")
    # Save original order
    original_cols: List[str] = list(main_df.columns)
    work = df.copy()
    performed: List[str] = []
    steps = getattr(pipeline_manager, "preprocessing_steps", [])

    for name, transformer, feats in steps:
        # 1) KNN missing-value imputation: handle all columns (numeric + categorical)
        if name.startswith("missing_values_knn_2"):
            all_cols = list(work.columns)
            num_cols = [col for col in all_cols if work[col].dtype.kind in 'iufc']
            cat_cols = [
                col for col in all_cols
                if work[col].dtype == "object" or isinstance(work[col].dtype, pd.CategoricalDtype)
            ]
            df_impute = work[all_cols].copy()

            # Find columns with missing values (to report them)
            missing_counts = work[all_cols].isnull().sum()
            processed_columns = [col for col in all_cols if missing_counts[col] > 0]

            encoder = None
            if cat_cols:
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                df_impute[cat_cols] = encoder.fit_transform(df_impute[cat_cols])

            imputer = transformer
            imputer.fit(df_impute)
            imputed_array = imputer.transform(df_impute)
            imputed_df = pd.DataFrame(imputed_array, columns=all_cols, index=work.index)

            # Put values back
            for col in num_cols:
                work[col] = imputed_df[col].round(2)
            if cat_cols and encoder is not None:
                imputed_df[cat_cols] = imputed_df[cat_cols].round(0)
                work[cat_cols] = encoder.inverse_transform(imputed_df[cat_cols])

            if processed_columns:
                performed.append(
                    f"KNN imputation applied to {len(processed_columns)} column(s) with missing values: "
                    + ", ".join(processed_columns)
                )
            else:
                performed.append("KNN imputation: No missing values found in any column.")

            # Optional: warning if any NaN left
            if work.isnull().any().any():
                n_missing = int(work.isnull().sum().sum())
                performed.append(f"Warning: {n_missing} NaN values remain after KNN imputation!")

        # 2) Numeric missing-value imputation
        elif name.startswith("missing_values"):
            num_cols = [c for c in work.select_dtypes(include=[np.number]).columns if work[c].isnull().any()]
            if num_cols:
                try:
                    work[num_cols] = transformer.transform(work[num_cols])
                except Exception:
                    imp = clone(transformer)
                    imp.fit(work[num_cols])
                    work[num_cols] = imp.transform(work[num_cols])
                performed.append(
                    f"Numeric missing value imputation applied to: {', '.join(num_cols)}"
                )

        # 3) Min-max / normalization
        elif name in {"normalization", "minmax_normalization"}:
            cols_present = [c for c in feats if c in work.columns and pd.api.types.is_numeric_dtype(work[c])]
            if cols_present:
                try:
                    work[cols_present] = transformer.transform(work[cols_present])
                except Exception:
                    sc = clone(transformer)
                    sc.fit(work[cols_present])
                    work[cols_present] = sc.transform(work[cols_present])
                work[cols_present] = work[cols_present].round(3)
                performed.append(f"Min-max scaling applied to: {', '.join(cols_present)}")

        # 4) Label encoding
        elif name == "label_encoding":
            changed = False
            for col, enc in transformer.items():
                if col in work.columns:
                    try:
                        work[col] = enc.transform(work[[col]]).ravel()
                    except ValueError:
                        unseen = np.setdiff1d(work[col].unique(), enc.classes_)
                        enc.classes_ = np.concatenate([enc.classes_, unseen])
                        work[col] = enc.transform(work[[col]]).ravel()
                    changed = True
            if changed:
                performed.append("Label encoding applied.")

        # 5) IQR outlier clipping
        elif name == "iqr_outlier_handling":
            cols_present = [c for c in feats if c in work.columns]
            if cols_present:
                for col in cols_present:
                    bounds = transformer[col]
                    work[col] = work[col].clip(bounds["lower"], bounds["upper"])
                performed.append(f"IQR clipping applied to: {', '.join(cols_present)}")

        # 6) Isolation Forest outlier handling
        elif name == "isolation_forest_outlier_handling":
            cols_present = [c for c in feats if c in work.columns]
            if cols_present:
                for col in cols_present:
                    mdl = transformer[col]
                    preds = mdl.predict(work[[col]])
                    work.loc[preds == -1, col] = work[col].median()
                performed.append(f"Isolation Forest applied to: {', '.join(cols_present)}")

        # 7) One-Hot Encoding
        elif name == "one_hot_encoding":
            processed = []
            for col, ohe in transformer.items():
                if col in work.columns:
                    # Apply One-Hot Encoding
                    arr = ohe.transform(work[[col]])
                    cats = ohe.categories_[0]  # Get categories for this column
                    dummy_cols = [f"{col}_{cat}" for cat in cats]  # Create new column names for dummies
                    df_dummies = pd.DataFrame(arr, columns=dummy_cols, index=work.index)  # Create DataFrame for dummies
                    work = pd.concat([work.drop(columns=[col]), df_dummies], axis=1)  # Add dummies to the work DataFrame
                    processed.append(col)
            if processed:
                performed.append(f"One-hot encoding applied to: {', '.join(processed)}")

    # Restore original column order
    work = work.loc[:, original_cols]
    # Compose summary string without duplicates
    summary = "; ".join(dict.fromkeys(performed))
    return summary, work