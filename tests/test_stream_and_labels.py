from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from app.monitoring import backfill_labels
from app.simulator import generate_stream


def test_apply_scenario_preserves_target_and_changes_features_for_drift() -> (
    None
):
    source_df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "balance": [100.0, 150.0, 200.0],
            "campaign": [1, 2, 3],
            "job": ["admin", "technician", "services"],
            "contact": ["telephone", "unknown", "telephone"],
            "target": [0, 1, 0],
        }
    )

    drifted_df = generate_stream.apply_scenario(
        source_df, scenario="mild", seed=42
    )

    assert drifted_df["target"].tolist() == [0, 1, 0]
    assert not drifted_df.drop(columns=["target"]).equals(
        source_df.drop(columns=["target"])
    )


def test_repeat_to_size_raises_for_non_positive_rows() -> None:
    source_df = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(ValueError) as exc_info:
        generate_stream.repeat_to_size(source_df, rows=0, seed=42)

    assert "rows must be greater than 0" in str(exc_info.value)


def test_derive_label_supports_perfect_and_custom_flip_policies() -> None:
    always_flip_rng = np.random.default_rng(1)
    never_flip_rng = np.random.default_rng(42)

    perfect_label = backfill_labels.derive_label(
        original_target=1,
        scenario="severe",
        policy="perfect",
        rng=never_flip_rng,
        flip_prob=None,
    )
    flipped_label = backfill_labels.derive_label(
        original_target=1,
        scenario="none",
        policy="custom_flip",
        rng=always_flip_rng,
        flip_prob=1.0,
    )

    assert perfect_label == 1
    assert flipped_label == 0


def test_validate_args_rejects_invalid_custom_flip_arguments() -> None:
    args = argparse.Namespace(
        delay_hours=24.0,
        label_policy="custom_flip",
        flip_prob=None,
        batch_size=100,
    )

    with pytest.raises(ValueError) as exc_info:
        backfill_labels.validate_args(args)

    assert "flip_prob must be provided" in str(exc_info.value)
