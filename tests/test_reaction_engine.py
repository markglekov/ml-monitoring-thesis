from __future__ import annotations

import pytest

from app.monitoring import reaction_engine


def test_should_route_to_manual_review_is_deterministic() -> None:
    request_id = "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"

    first = reaction_engine.should_route_to_manual_review(request_id, 0.5)
    second = reaction_engine.should_route_to_manual_review(request_id, 0.5)

    assert first is second
    assert (
        reaction_engine.should_route_to_manual_review(request_id, 0.0) is False
    )
    assert (
        reaction_engine.should_route_to_manual_review(request_id, 1.0) is True
    )


def test_build_action_configs_tightens_threshold_for_segment() -> None:
    old_config, new_config = reaction_engine._build_action_configs(
        incident={
            "segment_key": "segment-a",
        },
        current_policy={
            "threshold": 0.5,
            "manual_review_probability": 0.0,
        },
        baseline_threshold=0.5,
        action_type="tighten_threshold",
        threshold_step=0.05,
    )

    assert old_config["segment_key"] == "segment-a"
    assert old_config["threshold"] == 0.5
    assert new_config["threshold"] == 0.55
    assert new_config["manual_review_probability"] == 0.0


def test_build_action_configs_rejects_threshold_above_cap() -> None:
    with pytest.raises(ValueError):
        reaction_engine._build_action_configs(
            incident={"segment_key": None},
            current_policy={
                "threshold": 0.95,
                "manual_review_probability": 0.0,
            },
            baseline_threshold=0.5,
            action_type="tighten_threshold",
            threshold_step=0.05,
        )
