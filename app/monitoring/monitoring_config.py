from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

from app.common.config import settings

DEFAULT_CONFIG: dict[str, Any] = {
    "scheduler": {
        "poll_interval_sec": 5.0,
        "drift_interval_sec": 300.0,
        "quality_interval_sec": 300.0,
    },
    "drift": {
        "window_size": 300,
        "min_rows": 50,
        "univariate": {
            "pvalue_warning": 0.05,
            "pvalue_critical": 0.01,
            "psi_warning": 0.20,
            "psi_critical": 0.35,
        },
        "domain_classifier": {
            "auc_warning": 0.65,
            "auc_critical": 0.75,
            "pvalue_max": 0.05,
            "permutations": 31,
        },
        "advanced_detectors": {
            "mmd": {"warning": 0.05, "critical": 0.10, "permutations": 31},
            "wasserstein": {"warning": 0.10, "critical": 0.20},
            "cusum": {"warning": 5.0, "critical": 8.0},
            "ewma": {"warning": 3.0, "critical": 5.0},
            "adwin": {"delta": 0.20, "warning": 0.10, "critical": 0.20},
            "ddm": {"warning": 0.15, "critical": 0.40},
            "eddm": {"warning": 0.20, "critical": 0.40},
        },
    },
    "quality": {
        "window_size": 300,
        "min_rows": 50,
        "baseline_source": "test",
        "critical_multiplier": 2.0,
        "proxy_threshold_band": 0.10,
        "degradation_rules": {
            "roc_auc": {
                "mode": "min_delta",
                "threshold": 0.03,
                "detector_name": "labeled",
            },
            "pr_auc": {
                "mode": "min_delta",
                "threshold": 0.04,
                "detector_name": "labeled",
            },
            "precision": {
                "mode": "min_delta",
                "threshold": 0.04,
                "detector_name": "labeled",
            },
            "recall": {
                "mode": "min_delta",
                "threshold": 0.03,
                "detector_name": "labeled",
            },
            "f1": {
                "mode": "min_delta",
                "threshold": 0.04,
                "detector_name": "labeled",
            },
            "brier_score": {
                "mode": "max_delta",
                "threshold": 0.02,
                "detector_name": "labeled",
            },
            "ece": {
                "mode": "max_delta",
                "threshold": 0.03,
                "detector_name": "calibration",
            },
            "positive_rate_pred": {
                "mode": "abs_delta",
                "threshold": 0.08,
                "detector_name": "proxy",
            },
            "positive_rate_true": {
                "mode": "abs_delta",
                "threshold": 0.05,
                "detector_name": "labeled",
            },
            "score_mean": {
                "mode": "abs_delta",
                "threshold": 0.08,
                "detector_name": "proxy",
            },
            "score_std": {
                "mode": "abs_delta",
                "threshold": 0.06,
                "detector_name": "proxy",
            },
            "score_entropy": {
                "mode": "max_delta",
                "threshold": 0.05,
                "detector_name": "proxy",
            },
            "near_threshold_rate": {
                "mode": "abs_delta",
                "threshold": 0.10,
                "detector_name": "proxy",
            },
            "score_psi": {
                "mode": "max_absolute",
                "threshold": 0.20,
                "detector_name": "proxy",
            },
        },
    },
    "reaction": {
        "engine_enabled": True,
        "mode": "dry_run",
        "threshold_step": 0.05,
        "threshold_cap": 0.95,
        "manual_review_probability": 1.0,
        "action_policy": {
            "drift": {"critical": "manual_review", "warning": "observe"},
            "quality": {
                "critical": "tighten_threshold",
                "warning": "collect_labels",
            },
            "proxy": {
                "critical": "collect_labels",
                "warning": "collect_labels",
            },
        },
    },
}


def _deep_merge(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class MonitoringConfig:
    """Typed accessors for operator-facing monitoring thresholds."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def get(self, path: tuple[str, ...], default: Any) -> Any:
        value: Any = self.payload
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def get_int(self, path: tuple[str, ...], default: int) -> int:
        return int(self.get(path, default))

    def get_float(self, path: tuple[str, ...], default: float) -> float:
        return float(self.get(path, default))

    def get_str(self, path: tuple[str, ...], default: str) -> str:
        return str(self.get(path, default))

    def get_bool(self, path: tuple[str, ...], default: bool) -> bool:
        value = self.get(path, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def get_mapping(
        self, path: tuple[str, ...], default: dict[str, Any]
    ) -> dict[str, Any]:
        value = self.get(path, default)
        if not isinstance(value, dict):
            return deepcopy(default)
        return deepcopy(value)


def load_monitoring_config() -> MonitoringConfig:
    """Load monitoring defaults from YAML with in-code fallback values."""

    if not settings.monitoring_config_path.exists():
        return MonitoringConfig(deepcopy(DEFAULT_CONFIG))

    with settings.monitoring_config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        return MonitoringConfig(deepcopy(DEFAULT_CONFIG))

    return MonitoringConfig(_deep_merge(DEFAULT_CONFIG, loaded))


monitoring_config = load_monitoring_config()
