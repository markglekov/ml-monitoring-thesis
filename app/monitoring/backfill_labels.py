from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.config import settings
from app.common.logging import get_logger, setup_logging


MANIFEST_DIR = ROOT / "artifacts" / "reports" / "manifests"

logger = get_logger(__name__)


def pick_latest_manifest() -> Path:
    """Pick the newest generated stream manifest from the manifest directory."""

    files = sorted(MANIFEST_DIR.glob("stream_*.csv"))
    if not files:
        raise FileNotFoundError(f"No manifests found in: {MANIFEST_DIR}")
    return files[-1]


def scenario_flip_prob(scenario: str) -> float:
    """Return a default label-noise probability for one drift scenario."""

    if scenario == "none":
        return 0.00
    if scenario == "mild":
        return 0.10
    if scenario == "severe":
        return 0.25
    return 0.00


def derive_label(
    original_target: int,
    scenario: str,
    policy: str,
    rng: np.random.Generator,
    flip_prob: float | None,
) -> int:
    """Derive a delayed label with optional controlled degradation."""

    y_true = int(original_target)

    if policy == "perfect":
        return y_true

    if policy == "scenario_default":
        probability = scenario_flip_prob(scenario)
    elif policy == "custom_flip":
        if flip_prob is None:
            raise ValueError("flip_prob must be provided for custom_flip policy")
        probability = float(flip_prob)
    else:
        raise ValueError(f"Unsupported label policy: {policy}")

    if rng.random() < probability:
        return 1 - y_true
    return y_true


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before inserting any labels."""

    if args.delay_hours < 0:
        raise ValueError("delay_hours must be greater than or equal to 0")

    if args.label_policy == "custom_flip":
        if args.flip_prob is None:
            raise ValueError("flip_prob must be provided when label_policy=custom_flip")
        if not 0.0 <= float(args.flip_prob) <= 1.0:
            raise ValueError("flip_prob must be between 0 and 1")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")


def post_labels_batch(api_url: str, labels: list[dict[str, Any]], timeout: float) -> dict[str, Any]:
    """Send one delayed-label batch to the ingestion API."""

    response = requests.post(
        f"{api_url.rstrip('/')}/labels/batch",
        json={"labels": labels},
        timeout=timeout,
    )
    response.raise_for_status()
    return dict(response.json())


def main() -> None:
    """Backfill delayed labels into ground_truth from a generated manifest."""

    setup_logging()

    parser = argparse.ArgumentParser(description="Backfill delayed ground-truth labels")
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument(
        "--label-policy",
        type=str,
        choices=["perfect", "scenario_default", "custom_flip"],
        default="perfect",
    )
    parser.add_argument("--flip-prob", type=float, default=None)
    parser.add_argument("--delay-hours", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-url", type=str, default=f"http://localhost:{settings.api_port}")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    args = parser.parse_args()

    validate_args(args)

    manifest_path = Path(args.manifest_path) if args.manifest_path else pick_latest_manifest()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        raise ValueError("Manifest is empty")

    required_columns = {"request_id", "segment_key", "scenario", "original_target"}
    missing_columns = required_columns - set(manifest_df.columns)
    if missing_columns:
        raise ValueError(f"Manifest missing columns: {sorted(missing_columns)}")

    rng = np.random.default_rng(args.seed)
    label_ts = datetime.now(timezone.utc) - timedelta(hours=float(args.delay_hours))

    logger.info(
        "Backfilling labels from manifest=%s policy=%s delay_hours=%s rows=%s",
        manifest_path,
        args.label_policy,
        args.delay_hours,
        len(manifest_df),
    )

    payloads: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []

    for _, row in manifest_df.iterrows():
        y_true = derive_label(
            original_target=int(row["original_target"]),
            scenario=str(row["scenario"]),
            policy=args.label_policy,
            rng=rng,
            flip_prob=args.flip_prob,
        )

        payload = {
            "request_id": str(row["request_id"]),
            "y_true": y_true,
            "label_ts": label_ts.isoformat(),
        }
        payloads.append(payload)

        if len(preview_rows) < 5:
            preview_rows.append(
                {
                    "request_id": str(row["request_id"]),
                    "segment_key": str(row["segment_key"]),
                    "scenario": str(row["scenario"]),
                    "y_true": y_true,
                }
            )

    inserted_count = 0
    for offset in range(0, len(payloads), args.batch_size):
        batch = payloads[offset : offset + args.batch_size]
        batch_response = post_labels_batch(
            api_url=args.api_url,
            labels=batch,
            timeout=float(args.timeout_sec),
        )
        inserted_count += int(batch_response.get("upserted_count", 0))

    print("Labels backfilled.")
    print(f"Manifest: {manifest_path}")
    print(f"Rows processed: {len(payloads)}")
    print(f"Rows upserted via API: {inserted_count}")
    print(f"Label policy: {args.label_policy}")
    print(f"Delay hours: {args.delay_hours}")
    print(f"API URL: {args.api_url}")
    print("Preview:")
    for item in preview_rows:
        print(item)


if __name__ == "__main__":
    main()
