from __future__ import annotations

import argparse
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_DATA_PATH = ROOT / "data" / "processed" / "test.csv"
MANIFEST_DIR = ROOT / "artifacts" / "reports" / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def to_native(value: Any) -> Any:
    """Convert pandas and numpy scalars into JSON-serializable Python values."""

    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_rows(source_split: str) -> pd.DataFrame:
    """Load a processed dataset that still contains the original target column."""

    if source_split == "train":
        data_path = TRAIN_DATA_PATH
    elif source_split == "test":
        data_path = TEST_DATA_PATH
    else:
        raise ValueError(f"Unsupported source split: {source_split}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError(f"Expected target column in dataset: {data_path}")
    return df


def repeat_to_size(df: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    """Sample or repeat rows until the requested stream size is reached."""

    if rows <= 0:
        raise ValueError("rows must be greater than 0")

    rng = np.random.default_rng(seed)

    if rows <= len(df):
        sampled_idx = rng.choice(len(df), size=rows, replace=False)
        return df.iloc[sampled_idx].reset_index(drop=True)

    repeats = math.ceil(rows / len(df))
    parts: list[pd.DataFrame] = []
    for _ in range(repeats):
        sampled_idx = rng.choice(len(df), size=len(df), replace=False)
        parts.append(df.iloc[sampled_idx].reset_index(drop=True))

    return pd.concat(parts, ignore_index=True).iloc[:rows].reset_index(drop=True)


def apply_mild_drift(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Apply a moderate shift to a subset of numeric and categorical features."""

    rng = np.random.default_rng(seed)
    out = df.copy()

    if "age" in out.columns:
        out["age"] = np.clip(out["age"].fillna(0) + rng.integers(4, 11, size=len(out)), 18, 95)

    if "balance" in out.columns:
        out["balance"] = out["balance"].fillna(0) * rng.uniform(1.15, 1.35, size=len(out)) + rng.normal(
            200,
            80,
            size=len(out),
        )

    if "campaign" in out.columns:
        out["campaign"] = np.maximum(out["campaign"].fillna(1) + rng.integers(1, 3, size=len(out)), 1)

    if "job" in out.columns:
        mask = rng.random(len(out)) < 0.25
        out.loc[mask, "job"] = "management"

    if "contact" in out.columns:
        mask = rng.random(len(out)) < 0.20
        out.loc[mask, "contact"] = "cellular"

    return out


def apply_severe_drift(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Apply a strong shift for clearly degraded distribution scenarios."""

    rng = np.random.default_rng(seed)
    out = df.copy()

    if "age" in out.columns:
        out["age"] = np.clip(out["age"].fillna(0) + rng.integers(10, 21, size=len(out)), 18, 95)

    if "balance" in out.columns:
        out["balance"] = out["balance"].fillna(0) * rng.uniform(1.5, 2.2, size=len(out)) + rng.normal(
            600,
            150,
            size=len(out),
        )

    if "campaign" in out.columns:
        out["campaign"] = np.maximum(out["campaign"].fillna(1) + rng.integers(2, 6, size=len(out)), 1)

    if "pdays" in out.columns:
        out["pdays"] = np.maximum(out["pdays"].fillna(0) + rng.integers(30, 120, size=len(out)), 0)

    if "previous" in out.columns:
        out["previous"] = np.maximum(out["previous"].fillna(0) + rng.integers(1, 4, size=len(out)), 0)

    if "job" in out.columns:
        mask = rng.random(len(out)) < 0.45
        out.loc[mask, "job"] = "student"

    if "marital" in out.columns:
        mask = rng.random(len(out)) < 0.30
        out.loc[mask, "marital"] = "single"

    if "housing" in out.columns:
        mask = rng.random(len(out)) < 0.35
        out.loc[mask, "housing"] = "yes"

    if "loan" in out.columns:
        mask = rng.random(len(out)) < 0.25
        out.loc[mask, "loan"] = "yes"

    if "contact" in out.columns:
        mask = rng.random(len(out)) < 0.50
        out.loc[mask, "contact"] = "cellular"

    return out


def apply_scenario(df: pd.DataFrame, scenario: str, seed: int) -> pd.DataFrame:
    """Apply the selected scenario while preserving the original target column."""

    out = df.copy()
    feature_columns = [column for column in out.columns if column != "target"]
    features_df = out[feature_columns].copy()

    if scenario == "none":
        shifted_df = features_df
    elif scenario == "mild":
        shifted_df = apply_mild_drift(features_df, seed)
    elif scenario == "severe":
        shifted_df = apply_severe_drift(features_df, seed)
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    result = shifted_df.copy()
    result["target"] = out["target"].values
    return result


def post_one(
    api_url: str,
    features: dict[str, Any],
    request_id: str,
    segment_key: str,
    timeout: float,
) -> tuple[bool, str]:
    """Send one request to the inference API with a stable request identifier."""

    payload = {
        "request_id": request_id,
        "features": {key: to_native(value) for key, value in features.items()},
        "segment_key": segment_key,
    }

    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/predict",
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return False, f"request_failed: {exc}"

    if response.status_code != 200:
        return False, f"http_{response.status_code}: {response.text}"

    return True, response.text


def build_manifest_path(segment_key: str, scenario: str) -> Path:
    """Build a default manifest path for one generated stream."""

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_segment_key = segment_key.replace("/", "_")
    return MANIFEST_DIR / f"stream_{safe_segment_key}_{scenario}_{ts}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inference stream for monitoring")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--source-split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--scenario", type=str, choices=["none", "mild", "severe"], default="none")
    parser.add_argument("--segment-key", type=str, default=None)
    parser.add_argument("--sleep-sec", type=float, default=0.03)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-path", type=str, default=None)
    args = parser.parse_args()

    if args.rows <= 0:
        parser.error("--rows must be greater than 0")

    base_df = load_rows(source_split=args.source_split)
    sample_df = repeat_to_size(base_df, rows=args.rows, seed=args.seed)
    stream_df = apply_scenario(sample_df, scenario=args.scenario, seed=args.seed)

    segment_key = args.segment_key or args.scenario
    manifest_path = Path(args.manifest_path) if args.manifest_path else build_manifest_path(segment_key, args.scenario)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    fail_count = 0
    manifest_records: list[dict[str, Any]] = []

    print(
        f"Generating stream: rows={args.rows}, source_split={args.source_split}, "
        f"scenario={args.scenario}, segment_key={segment_key}"
    )

    for idx, row in stream_df.iterrows():
        request_id = str(uuid.uuid4())
        row_dict = row.to_dict()
        target = int(row_dict.pop("target"))

        success, message = post_one(
            api_url=args.api_url,
            features=row_dict,
            request_id=request_id,
            segment_key=segment_key,
            timeout=args.timeout_sec,
        )

        if success:
            ok_count += 1
            manifest_records.append(
                {
                    "request_id": request_id,
                    "segment_key": segment_key,
                    "scenario": args.scenario,
                    "source_split": args.source_split,
                    "original_target": target,
                    "sent_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        else:
            fail_count += 1
            print(f"[FAIL] row={idx}: {message}")

        if (idx + 1) % 25 == 0 or idx == len(stream_df) - 1:
            print(f"Progress: {idx + 1}/{len(stream_df)} | ok={ok_count} | fail={fail_count}")

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    pd.DataFrame(manifest_records).to_csv(manifest_path, index=False)

    print("Done.")
    print(f"Sent={len(stream_df)}, ok={ok_count}, fail={fail_count}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
