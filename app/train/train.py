"""Train and calibrate a baseline model for the Bank Marketing dataset.

The script performs the following steps:
1. Download the dataset from UCI via ``ucimlrepo``.
2. Clean feature values and convert the target to a binary label.
3. Split the data into train, validation, and test segments without shuffling.
4. Fit a preprocessing + logistic regression pipeline on the training split.
5. Calibrate predicted probabilities on the validation split.
6. Select the operating threshold that maximizes F1 on validation.
7. Persist the fitted model, split files, and a baseline profile for monitoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.logging import get_logger, setup_logging

ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
BASELINES_DIR = ARTIFACTS_DIR / "baselines"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_PATH = MODELS_DIR / "bank_marketing_model.joblib"
BASELINE_PATH = BASELINES_DIR / "baseline_profile.json"
TRAIN_SPLIT_PATH = PROCESSED_DIR / "train.csv"
VAL_SPLIT_PATH = PROCESSED_DIR / "val.csv"
TEST_SPLIT_PATH = PROCESSED_DIR / "test.csv"
LEAKAGE_COLUMNS = ["duration"]
THRESHOLD_GRID = np.linspace(0.10, 0.90, 81)
UCI_BANK_MARKETING_DATASET_ID = 222

logger = get_logger(__name__)


def ensure_directories() -> None:
    """Create output directories required by the training pipeline."""

    for directory in [MODELS_DIR, BASELINES_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_bank_marketing() -> pd.DataFrame:
    """Load the UCI Bank Marketing dataset and normalize the target column.

    Returns:
        A single dataframe containing cleaned features and a binary ``target``.

    Raises:
        ValueError: If the dataset exposes an unexpected target shape or if the
            target values cannot be mapped to ``0`` and ``1``.
    """

    dataset = fetch_ucirepo(id=UCI_BANK_MARKETING_DATASET_ID)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Expected exactly one target column in the source dataset.")
        y = y.iloc[:, 0]

    y = y.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Failed to map Bank Marketing target values to 0/1.")

    df = X.copy()
    df["target"] = y.astype(int)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def drop_leakage_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove columns that leak post-outcome information into training."""

    columns_to_drop = [column for column in LEAKAGE_COLUMNS if column in df.columns]
    if not columns_to_drop:
        return df, []

    return df.drop(columns=columns_to_drop), columns_to_drop


def split_time_ordered(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train, validation, and test segments in-order.

    The split is deterministic and keeps the original row order intact:
    - first 60% for training
    - next 20% for validation
    - final 20% for testing
    """

    n_rows = len(df)
    train_end = int(n_rows * 0.60)
    val_end = int(n_rows * 0.80)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a preprocessing block for mixed numeric and categorical features."""

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def build_model_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Create the baseline classification pipeline used for training."""

    base_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", base_model),
        ]
    )


def calibrate_model(
    fitted_pipeline: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> CalibratedClassifierCV:
    """Calibrate a fitted model on a disjoint validation set.

    Newer versions of scikit-learn removed ``cv="prefit"``. The supported
    replacement is to wrap the already-fitted estimator in ``FrozenEstimator``.
    That preserves the pre-trained pipeline and uses the full validation split
    exclusively for probability calibration.
    """

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(fitted_pipeline),
        method="sigmoid",
        ensemble="auto",
    )
    calibrated_model.fit(X_validation, y_validation)
    return calibrated_model


def compute_classification_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute threshold-dependent and ranking-based binary classification metrics."""

    y_pred = (y_score >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "positive_rate_true": float(np.mean(y_true)),
    }


def choose_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    """Select the decision threshold that maximizes validation F1 score."""

    best_threshold = 0.50
    best_f1 = -1.0
    best_metrics: dict[str, float] | None = None

    for threshold in THRESHOLD_GRID:
        metrics = compute_classification_metrics(y_true, y_score, threshold=float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def make_baseline_profile(train_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Build a lightweight statistical profile for downstream monitoring jobs."""

    baseline: dict[str, Any] = {
        "row_count": int(len(train_df)),
        "target_rate": float(train_df["target"].mean()),
        "features": {},
    }

    for col in feature_columns:
        series = train_df[col]

        if pd.api.types.is_numeric_dtype(series):
            baseline["features"][col] = {
                "type": "numeric",
                "mean": None if pd.isna(series.mean()) else float(series.mean()),
                "std": None if pd.isna(series.std()) else float(series.std()),
                "min": None if pd.isna(series.min()) else float(series.min()),
                "max": None if pd.isna(series.max()) else float(series.max()),
                "quantiles": {
                    "q05": None if pd.isna(series.quantile(0.05)) else float(series.quantile(0.05)),
                    "q25": None if pd.isna(series.quantile(0.25)) else float(series.quantile(0.25)),
                    "q50": None if pd.isna(series.quantile(0.50)) else float(series.quantile(0.50)),
                    "q75": None if pd.isna(series.quantile(0.75)) else float(series.quantile(0.75)),
                    "q95": None if pd.isna(series.quantile(0.95)) else float(series.quantile(0.95)),
                },
            }
            continue

        value_counts = series.astype(str).value_counts(normalize=True, dropna=False)
        baseline["features"][col] = {
            "type": "categorical",
            "top_values": {str(key): float(value) for key, value in value_counts.head(20).items()},
            "unique_count": int(series.nunique(dropna=False)),
        }

    return baseline


def save_artifacts(
    calibrated_model: CalibratedClassifierCV,
    baseline_profile: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Persist the trained model, monitoring baseline, and processed data splits."""

    joblib.dump(calibrated_model, MODEL_PATH)

    with BASELINE_PATH.open("w", encoding="utf-8") as file_obj:
        json.dump(baseline_profile, file_obj, ensure_ascii=False, indent=2)

    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    val_df.to_csv(VAL_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)


def main() -> None:
    """Run the end-to-end training and artifact generation pipeline."""

    setup_logging()
    ensure_directories()

    logger.info("Loading UCI Bank Marketing dataset.")
    df = load_bank_marketing()
    logger.info("Dataset loaded with shape=%s and columns=%s", df.shape, list(df.columns))

    df, dropped_columns = drop_leakage_columns(df)
    if dropped_columns:
        logger.info("Dropped leakage columns: %s", dropped_columns)

    train_df, val_df, test_df = split_time_ordered(df)
    logger.info(
        "Prepared deterministic splits: train=%s, validation=%s, test=%s",
        train_df.shape,
        val_df.shape,
        test_df.shape,
    )

    feature_columns = [column for column in df.columns if column != "target"]
    X_train = train_df[feature_columns]
    y_train = train_df["target"]
    X_val = val_df[feature_columns]
    y_val = val_df["target"]
    X_test = test_df[feature_columns]
    y_test = test_df["target"]

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    logger.info("Numeric features: %s", numeric_features)
    logger.info("Categorical features: %s", categorical_features)

    model_pipeline = build_model_pipeline(preprocessor)

    logger.info("Training baseline logistic regression pipeline.")
    model_pipeline.fit(X_train, y_train)

    logger.info("Calibrating predicted probabilities on the validation split.")
    calibrated_model = calibrate_model(model_pipeline, X_val, y_val)

    val_scores = calibrated_model.predict_proba(X_val)[:, 1]
    best_threshold, val_metrics = choose_threshold(y_val, val_scores)
    logger.info("Selected operating threshold %.4f based on validation F1.", best_threshold)

    test_scores = calibrated_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_classification_metrics(y_test, test_scores, best_threshold)

    baseline_profile = make_baseline_profile(train_df, feature_columns)
    baseline_profile["threshold"] = best_threshold
    baseline_profile["validation_metrics"] = val_metrics
    baseline_profile["test_metrics"] = test_metrics
    baseline_profile["split"] = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }
    baseline_profile["feature_columns"] = feature_columns
    baseline_profile["numeric_features"] = numeric_features
    baseline_profile["categorical_features"] = categorical_features
    baseline_profile["dropped_columns"] = dropped_columns

    logger.info("Persisting model and generated artifacts.")
    save_artifacts(calibrated_model, baseline_profile, train_df, val_df, test_df)

    logger.info("Training pipeline completed successfully.")
    logger.info("Validation metrics:\n%s", json.dumps(val_metrics, ensure_ascii=False, indent=2))
    logger.info("Test metrics:\n%s", json.dumps(test_metrics, ensure_ascii=False, indent=2))
    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Baseline profile saved to %s", BASELINE_PATH)


if __name__ == "__main__":
    main()
