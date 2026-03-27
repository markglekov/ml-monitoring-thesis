from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.config import settings
from app.common.logging import get_logger, setup_logging


REFERENCE_DATA_PATH = ROOT / "data" / "processed" / "train.csv"
BASELINE_PATH = settings.baseline_path
MODEL_PATH = settings.model_path

logger = get_logger(__name__)


def load_reference_data() -> pd.DataFrame:
    """Load the reference dataset used as the baseline window for drift checks."""

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found: {REFERENCE_DATA_PATH}")

    df = pd.read_csv(REFERENCE_DATA_PATH)
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    return df


def load_baseline_profile() -> dict[str, Any]:
    """Load the saved baseline profile produced during model training."""

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline profile not found: {BASELINE_PATH}")

    with BASELINE_PATH.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_model() -> Any:
    """Load the persisted model used to compare score distributions."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def to_native(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-serializable Python objects."""

    if isinstance(value, dict):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def safe_json(obj: Any) -> str:
    """Serialize monitoring payloads into JSON for PostgreSQL JSONB columns."""

    return json.dumps(to_native(obj), ensure_ascii=False, default=str)


def normalize_categorical(series: pd.Series) -> pd.Series:
    """Represent missing categorical values explicitly before comparisons."""

    return series.where(series.notna(), "__nan__").astype(str)


def load_current_window(engine: Engine, window_size: int, segment_key: str | None = None) -> pd.DataFrame:
    """Load the latest inference window from PostgreSQL and expand stored features."""

    if segment_key:
        query = text(
            """
            SELECT ts, features_json, score, pred_label, segment_key
            FROM inference_log
            WHERE segment_key = :segment_key
            ORDER BY ts DESC
            LIMIT :window_size
            """
        )
        params = {"segment_key": segment_key, "window_size": window_size}
    else:
        query = text(
            """
            SELECT ts, features_json, score, pred_label, segment_key
            FROM inference_log
            ORDER BY ts DESC
            LIMIT :window_size
            """
        )
        params = {"window_size": window_size}

    with engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

    rows = list(reversed(rows))

    records: list[dict[str, Any]] = []
    for row in rows:
        features = row["features_json"]
        if isinstance(features, str):
            features = json.loads(features)

        record = dict(features)
        record["__score"] = float(row["score"])
        record["__pred_label"] = int(row["pred_label"])
        record["__segment_key"] = row["segment_key"]
        record["__ts"] = str(row["ts"])
        records.append(record)

    return pd.DataFrame(records)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Guarantee a dataframe contains the exact requested columns in order."""

    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns]


def calculate_numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float | None:
    """Calculate PSI for numeric features using reference quantile buckets."""

    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()

    if len(ref) < 10 or len(cur) < 10:
        return None

    quantiles = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) < 2:
        return 0.0

    ref_binned = pd.cut(ref, bins=quantiles, include_lowest=True, duplicates="drop")
    cur_binned = pd.cut(cur, bins=quantiles, include_lowest=True, duplicates="drop")

    ref_pct = ref_binned.value_counts(normalize=True).sort_index()
    cur_pct = cur_binned.value_counts(normalize=True).sort_index()

    all_bins = ref_pct.index.union(cur_pct.index)
    ref_pct = ref_pct.reindex(all_bins, fill_value=0.0)
    cur_pct = cur_pct.reindex(all_bins, fill_value=0.0)

    eps = 1e-6
    ref_pct = np.clip(ref_pct.values, eps, None)
    cur_pct = np.clip(cur_pct.values, eps, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def calculate_categorical_psi(reference: pd.Series, current: pd.Series) -> float | None:
    """Calculate PSI for categorical features over aligned category frequencies."""

    ref = normalize_categorical(reference)
    cur = normalize_categorical(current)

    if len(ref) < 10 or len(cur) < 10:
        return None

    ref_pct = ref.value_counts(normalize=True, dropna=False)
    cur_pct = cur.value_counts(normalize=True, dropna=False)

    all_values = ref_pct.index.union(cur_pct.index)
    ref_pct = ref_pct.reindex(all_values, fill_value=0.0)
    cur_pct = cur_pct.reindex(all_values, fill_value=0.0)

    eps = 1e-6
    ref_pct = np.clip(ref_pct.values, eps, None)
    cur_pct = np.clip(cur_pct.values, eps, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def analyze_numeric_feature(feature_name: str, reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    """Analyze one numeric feature with KS test and PSI."""

    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()

    if len(ref) < 10 or len(cur) < 10:
        return {
            "feature_name": feature_name,
            "feature_type": "numeric",
            "ks_pvalue": None,
            "chi2_pvalue": None,
            "psi_value": None,
            "drift_detected": False,
            "details": {"status": "insufficient_data"},
        }

    ks_stat, ks_pvalue = ks_2samp(ref, cur)
    psi_value = calculate_numeric_psi(ref, cur)
    drift_detected = bool((ks_pvalue < 0.05) or ((psi_value is not None) and (psi_value >= 0.20)))

    return {
        "feature_name": feature_name,
        "feature_type": "numeric",
        "ks_pvalue": float(ks_pvalue),
        "chi2_pvalue": None,
        "psi_value": psi_value,
        "drift_detected": drift_detected,
        "details": {
            "ks_statistic": float(ks_stat),
            "reference_mean": float(ref.mean()),
            "current_mean": float(cur.mean()),
            "reference_std": float(ref.std()) if len(ref) > 1 else 0.0,
            "current_std": float(cur.std()) if len(cur) > 1 else 0.0,
            "reference_n": int(len(ref)),
            "current_n": int(len(cur)),
        },
    }


def analyze_categorical_feature(feature_name: str, reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    """Analyze one categorical feature with chi-square test and PSI."""

    ref = normalize_categorical(reference)
    cur = normalize_categorical(current)

    if len(ref) < 10 or len(cur) < 10:
        return {
            "feature_name": feature_name,
            "feature_type": "categorical",
            "ks_pvalue": None,
            "chi2_pvalue": None,
            "psi_value": None,
            "drift_detected": False,
            "details": {"status": "insufficient_data"},
        }

    ref_counts = ref.value_counts(dropna=False)
    cur_counts = cur.value_counts(dropna=False)

    all_values = ref_counts.index.union(cur_counts.index)
    ref_counts = ref_counts.reindex(all_values, fill_value=0)
    cur_counts = cur_counts.reindex(all_values, fill_value=0)

    contingency = np.vstack([ref_counts.values, cur_counts.values])

    if contingency.shape[1] <= 1:
        chi2_stat = 0.0
        chi2_pvalue = 1.0
    else:
        chi2_stat, chi2_pvalue, _, _ = chi2_contingency(contingency)

    psi_value = calculate_categorical_psi(ref, cur)
    drift_detected = bool((chi2_pvalue < 0.05) or ((psi_value is not None) and (psi_value >= 0.20)))

    top_ref = (ref_counts / ref_counts.sum()).sort_values(ascending=False).head(5).to_dict()
    top_cur = (cur_counts / cur_counts.sum()).sort_values(ascending=False).head(5).to_dict()

    return {
        "feature_name": feature_name,
        "feature_type": "categorical",
        "ks_pvalue": None,
        "chi2_pvalue": float(chi2_pvalue),
        "psi_value": psi_value,
        "drift_detected": drift_detected,
        "details": {
            "chi2_statistic": float(chi2_stat),
            "reference_top_values": {str(key): float(value) for key, value in top_ref.items()},
            "current_top_values": {str(key): float(value) for key, value in top_cur.items()},
            "reference_n": int(len(ref)),
            "current_n": int(len(cur)),
        },
    }


def insert_monitoring_run(engine: Engine, window_size: int, segment_key: str | None) -> int:
    """Insert a monitoring run placeholder row and return its identifier."""

    query = text(
        """
        INSERT INTO monitoring_runs (
            model_version,
            window_size,
            segment_key,
            status
        )
        VALUES (
            :model_version,
            :window_size,
            :segment_key,
            'running'
        )
        RETURNING id
        """
    )

    with engine.begin() as connection:
        run_id = connection.execute(
            query,
            {
                "model_version": settings.model_version,
                "window_size": window_size,
                "segment_key": segment_key,
            },
        ).scalar_one()

    return int(run_id)


def finalize_monitoring_run(
    engine: Engine,
    run_id: int,
    status: str,
    drifted_features_count: int,
    total_features_count: int,
    overall_drift: bool,
    summary: dict[str, Any],
) -> None:
    """Finalize a monitoring run row with summary fields and completion time."""

    query = text(
        """
        UPDATE monitoring_runs
        SET
            ts_finished = NOW(),
            status = :status,
            drifted_features_count = :drifted_features_count,
            total_features_count = :total_features_count,
            overall_drift = :overall_drift,
            summary_json = CAST(:summary_json AS JSONB)
        WHERE id = :run_id
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "run_id": run_id,
                "status": status,
                "drifted_features_count": drifted_features_count,
                "total_features_count": total_features_count,
                "overall_drift": overall_drift,
                "summary_json": safe_json(summary),
            },
        )


def insert_drift_metrics(engine: Engine, run_id: int, results: list[dict[str, Any]]) -> None:
    """Persist per-feature drift metrics for one monitoring run."""

    if not results:
        return

    query = text(
        """
        INSERT INTO drift_metrics (
            run_id,
            feature_name,
            feature_type,
            ks_pvalue,
            chi2_pvalue,
            psi_value,
            drift_detected,
            details_json
        )
        VALUES (
            :run_id,
            :feature_name,
            :feature_type,
            :ks_pvalue,
            :chi2_pvalue,
            :psi_value,
            :drift_detected,
            CAST(:details_json AS JSONB)
        )
        """
    )

    payloads = [
        {
            "run_id": run_id,
            "feature_name": item["feature_name"],
            "feature_type": item["feature_type"],
            "ks_pvalue": item["ks_pvalue"],
            "chi2_pvalue": item["chi2_pvalue"],
            "psi_value": item["psi_value"],
            "drift_detected": item["drift_detected"],
            "details_json": safe_json(item["details"]),
        }
        for item in results
    ]

    with engine.begin() as connection:
        connection.execute(query, payloads)


def main() -> None:
    """Execute one batch drift-monitoring run over the latest inference window."""

    setup_logging()

    parser = argparse.ArgumentParser(description="Drift monitoring job")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--segment-key", type=str, default=None)
    parser.add_argument("--min-rows", type=int, default=50)
    args = parser.parse_args()

    if args.window_size <= 0:
        parser.error("--window-size must be greater than 0")
    if args.min_rows <= 0:
        parser.error("--min-rows must be greater than 0")

    logger.info(
        "Starting drift job. model_version=%s window_size=%s segment_key=%s min_rows=%s",
        settings.model_version,
        args.window_size,
        args.segment_key,
        args.min_rows,
    )

    baseline_profile = load_baseline_profile()
    feature_columns = baseline_profile["feature_columns"]
    numeric_features = set(baseline_profile.get("numeric_features", []))
    categorical_features = set(baseline_profile.get("categorical_features", []))
    engine = create_engine(settings.database_url, future=True, pool_pre_ping=True)
    run_id = insert_monitoring_run(engine, window_size=args.window_size, segment_key=args.segment_key)

    try:
        reference_df = ensure_columns(load_reference_data(), feature_columns)
        current_df = load_current_window(engine, window_size=args.window_size, segment_key=args.segment_key)

        if current_df.empty or len(current_df) < args.min_rows:
            summary = {
                "status": "skipped_insufficient_data",
                "current_rows": int(len(current_df)),
                "required_min_rows": int(args.min_rows),
            }
            finalize_monitoring_run(
                engine=engine,
                run_id=run_id,
                status="skipped_insufficient_data",
                drifted_features_count=0,
                total_features_count=len(feature_columns),
                overall_drift=False,
                summary=summary,
            )
            logger.info("Drift job skipped due to insufficient data: %s", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return

        current_features_df = ensure_columns(current_df, feature_columns)

        model = load_model()
        reference_scores = model.predict_proba(reference_df)[:, 1]
        current_scores = pd.to_numeric(current_df["__score"], errors="coerce").fillna(0.0).values

        results: list[dict[str, Any]] = []
        for feature in feature_columns:
            ref_col = reference_df[feature]
            cur_col = current_features_df[feature]

            if feature in numeric_features:
                result = analyze_numeric_feature(feature, ref_col, cur_col)
            elif feature in categorical_features:
                result = analyze_categorical_feature(feature, ref_col, cur_col)
            elif pd.api.types.is_numeric_dtype(ref_col):
                result = analyze_numeric_feature(feature, ref_col, cur_col)
            else:
                result = analyze_categorical_feature(feature, ref_col, cur_col)

            results.append(result)

        results.append(
            analyze_numeric_feature(
                feature_name="__score",
                reference=pd.Series(reference_scores),
                current=pd.Series(current_scores),
            )
        )

        drifted = [item for item in results if item["drift_detected"]]
        insert_drift_metrics(engine, run_id, results)

        top_drift = sorted(
            results,
            key=lambda item: (item["psi_value"] if item["psi_value"] is not None else -1.0),
            reverse=True,
        )[:5]

        threshold = float(baseline_profile.get("threshold", 0.5))
        positive_rate_current = float(pd.to_numeric(current_df["__pred_label"], errors="coerce").fillna(0).mean())
        positive_rate_reference = float((reference_scores >= threshold).mean())
        drifted_features_count = len(drifted)
        total_features_count = len(results)
        overall_drift = drifted_features_count >= 2 or any(
            item["feature_name"] == "__score" and item["drift_detected"] for item in drifted
        )

        summary = {
            "reference_rows": int(len(reference_df)),
            "current_rows": int(len(current_df)),
            "drifted_features": [item["feature_name"] for item in drifted],
            "top_drift_by_psi": [
                {
                    "feature_name": item["feature_name"],
                    "psi_value": item["psi_value"],
                    "drift_detected": item["drift_detected"],
                }
                for item in top_drift
            ],
            "reference_positive_rate_pred": positive_rate_reference,
            "current_positive_rate_pred": positive_rate_current,
            "segment_key": args.segment_key,
        }

        finalize_monitoring_run(
            engine=engine,
            run_id=run_id,
            status="completed",
            drifted_features_count=drifted_features_count,
            total_features_count=total_features_count,
            overall_drift=overall_drift,
            summary=summary,
        )

        logger.info(
            "Drift job completed. run_id=%s drifted_features=%s total_features=%s overall_drift=%s",
            run_id,
            drifted_features_count,
            total_features_count,
            overall_drift,
        )
        print("Drift job completed.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    except Exception as exc:
        logger.exception("Drift job failed. run_id=%s", run_id)
        finalize_monitoring_run(
            engine=engine,
            run_id=run_id,
            status="failed",
            drifted_features_count=0,
            total_features_count=0,
            overall_drift=False,
            summary={"error": str(exc)},
        )
        raise
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
