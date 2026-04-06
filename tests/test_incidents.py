from __future__ import annotations

import json

import numpy as np

from app.monitoring.incidents import _safe_json


def test_safe_json_replaces_non_finite_values_with_null() -> None:
    payload = {
        "severity": "warning",
        "top_drift_by_psi": [
            {
                "feature_name": "age",
                "psi_value": float("nan"),
                "effect_size": np.float64("inf"),
                "pvalue_adj": 0.01,
            }
        ],
        "score_drift": {"effect_size": float("-inf")},
    }

    serialized = _safe_json(payload)
    decoded = json.loads(serialized)

    first_item = decoded["top_drift_by_psi"][0]
    assert first_item["psi_value"] is None
    assert first_item["effect_size"] is None
    assert decoded["score_drift"]["effect_size"] is None
