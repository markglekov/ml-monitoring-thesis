from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_PATH = ROOT / "data" / "processed" / "test.csv"


def to_native(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-serializable Python objects."""

    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_rows() -> pd.DataFrame:
    """Load processed test rows that will be replayed as inference requests."""

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DATA_PATH}")

    df = pd.read_csv(TEST_DATA_PATH)
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    return df


def repeat_to_size(df: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    """Sample or repeat the dataset until the requested stream length is reached."""

    if rows <= 0:
        raise ValueError("rows must be greater than 0")

    rng = np.random.default_rng(seed)

    if rows <= len(df):
        sampled_idx = rng.choice(len(df), size=rows, replace=False)
        return df.iloc[sampled_idx].reset_index(drop=True)

    repeats = math.ceil(rows / len(df))
    parts = []
    for _ in range(repeats):
        sampled_idx = rng.choice(len(df), size=len(df), replace=False)
        parts.append(df.iloc[sampled_idx].reset_index(drop=True))

    return pd.concat(parts, ignore_index=True).iloc[:rows].reset_index(drop=True)


def apply_mild_drift(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Apply a moderate feature shift that should be visible but not extreme."""

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
    """Apply a strong feature shift for obvious drift scenarios."""

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
    """Apply the requested drift scenario to the sampled dataframe."""

    if scenario == "none":
        return df.copy()
    if scenario == "mild":
        return apply_mild_drift(df, seed)
    if scenario == "severe":
        return apply_severe_drift(df, seed)
    raise ValueError(f"Unsupported scenario: {scenario}")


def post_one(api_url: str, features: dict[str, Any], segment_key: str, timeout: float) -> tuple[bool, str]:
    """Send a single inference request to the API."""

    payload = {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inference stream for monitoring")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--scenario", type=str, choices=["none", "mild", "severe"], default="none")
    parser.add_argument("--segment-key", type=str, default=None)
    parser.add_argument("--sleep-sec", type=float, default=0.03)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rows <= 0:
        parser.error("--rows must be greater than 0")

    base_df = load_rows()
    sample_df = repeat_to_size(base_df, rows=args.rows, seed=args.seed)
    stream_df = apply_scenario(sample_df, scenario=args.scenario, seed=args.seed)

    segment_key = args.segment_key or args.scenario

    ok_count = 0
    fail_count = 0

    print(f"Generating stream: rows={args.rows}, scenario={args.scenario}, segment_key={segment_key}")

    for idx, row in stream_df.iterrows():
        success, message = post_one(
            api_url=args.api_url,
            features=row.to_dict(),
            segment_key=segment_key,
            timeout=args.timeout_sec,
        )

        if success:
            ok_count += 1
        else:
            fail_count += 1
            print(f"[FAIL] row={idx}: {message}")

        if (idx + 1) % 25 == 0 or idx == len(stream_df) - 1:
            print(f"Progress: {idx + 1}/{len(stream_df)} | ok={ok_count} | fail={fail_count}")

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    print("Done.")
    print(f"Sent={len(stream_df)}, ok={ok_count}, fail={fail_count}")


if __name__ == "__main__":
    main()
