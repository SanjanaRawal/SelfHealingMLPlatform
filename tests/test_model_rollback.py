
import json
from pathlib import Path

import pytest

from self_healing_ml.orchestration.model_rollback import (
    rollback_champion,
)


@pytest.fixture
def rollback_environment(tmp_path):
    """Create a temporary registry with a successful promotion."""

    model_dir = tmp_path / "models"
    model_dir.mkdir()

    registry_path = model_dir / "registry.json"

    champion_name = "tuned_gradient_boosting.joblib"
    previous_champion_name = "tuned_logistic_regression.joblib"

    # Create dummy model artifacts.
    (model_dir / champion_name).touch()
    (model_dir / previous_champion_name).touch()

    registry = {
        "active_champion": champion_name,
        "models": [
            {
                "model_name": champion_name,
                "artifact_path": str(model_dir / champion_name),
                "metrics": {"f1_score": 0.62},
                "status": "champion",
            },
            {
                "model_name": previous_champion_name,
                "artifact_path": str(model_dir / previous_champion_name),
                "metrics": {"f1_score": 0.61},
                "status": "previous_champion",
            },
        ],
        "history": [
            {
                "timestamp": "2026-10-30T10:00:00+00:00",
                "action": "promote_challenger",
                "previous_champion": previous_champion_name,
                "new_champion": champion_name,
                "decision": "accept_challenger",
            }
        ],
    }

    registry_path.write_text(
        json.dumps(registry, indent=4),
        encoding="utf-8",
    )

    return {
        "registry_path": registry_path,
        "model_dir": model_dir,
        "registry": registry,
        "champion_name": champion_name,
        "previous_champion_name": previous_champion_name,
    }


def test_successful_rollback(rollback_environment):
    """Verify that the previous champion is successfully restored."""

    result = rollback_champion(
        registry_path=rollback_environment["registry_path"],
        model_dir=rollback_environment["model_dir"],
    )

    assert result["rolled_back"] is True
    assert result["active_champion"] == (
        rollback_environment["previous_champion_name"]
    )


def test_rollback_updates_registry_statuses(rollback_environment):
    """Verify that model statuses are updated after rollback."""

    rollback_champion(
        registry_path=rollback_environment["registry_path"],
        model_dir=rollback_environment["model_dir"],
    )

    registry = json.loads(
        rollback_environment["registry_path"].read_text(
            encoding="utf-8"
        )
    )

    statuses = {
        model["model_name"]: model["status"]
        for model in registry["models"]
    }

    assert statuses[
        rollback_environment["previous_champion_name"]
    ] == "champion"

    assert statuses[
        rollback_environment["champion_name"]
    ] == "previous_champion"


def test_rollback_records_history(rollback_environment):
    """Verify that rollback events are recorded in registry history."""

    rollback_champion(
        registry_path=rollback_environment["registry_path"],
        model_dir=rollback_environment["model_dir"],
    )

    registry = json.loads(
        rollback_environment["registry_path"].read_text(
            encoding="utf-8"
        )
    )

    latest_event = registry["history"][-1]

    assert latest_event["action"] == "rollback_champion"
    assert latest_event["previous_champion"] == (
        rollback_environment["champion_name"]
    )
    assert latest_event["new_champion"] == (
        rollback_environment["previous_champion_name"]
    )


def test_missing_registry_raises_error(tmp_path):
    """Verify that a missing registry raises FileNotFoundError."""

    with pytest.raises(FileNotFoundError):
        rollback_champion(
            registry_path=tmp_path / "registry.json",
            model_dir=tmp_path,
        )


def test_empty_active_champion_raises_error(rollback_environment):
    """Verify that rollback is rejected without an active champion."""

    registry_path = rollback_environment["registry_path"]

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["active_champion"] = None

    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="no active champion"):
        rollback_champion(
            registry_path=registry_path,
            model_dir=rollback_environment["model_dir"],
        )


def test_empty_history_raises_error(rollback_environment):
    """Verify that rollback is rejected when history is empty."""

    registry_path = rollback_environment["registry_path"]

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["history"] = []

    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="history is empty"):
        rollback_champion(
            registry_path=registry_path,
            model_dir=rollback_environment["model_dir"],
        )


def test_invalid_latest_event_raises_error(rollback_environment):
    """Verify that rollback requires a latest promotion event."""

    registry_path = rollback_environment["registry_path"]

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["history"][-1]["action"] = "rollback_champion"

    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="latest registry event"):
        rollback_champion(
            registry_path=registry_path,
            model_dir=rollback_environment["model_dir"],
        )


def test_invalid_previous_champion_raises_error(rollback_environment):
    """Verify that invalid artifact names are rejected."""

    registry_path = rollback_environment["registry_path"]

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["history"][-1]["previous_champion"] = "../outside.joblib"

    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="previous champion"):
        rollback_champion(
            registry_path=registry_path,
            model_dir=rollback_environment["model_dir"],
        )


def test_unregistered_previous_champion_raises_error(
    rollback_environment,
):
    """Verify that an unregistered previous champion cannot be restored."""

    registry_path = rollback_environment["registry_path"]

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["models"] = [
        model
        for model in registry["models"]
        if model["model_name"]
        != rollback_environment["previous_champion_name"]
    ]

    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="not registered"):
        rollback_champion(
            registry_path=registry_path,
            model_dir=rollback_environment["model_dir"],
        )


def test_missing_previous_artifact_raises_error(rollback_environment):
    """Verify that a missing previous model artifact prevents rollback."""

    previous_model_path = (
        rollback_environment["model_dir"]
        / rollback_environment["previous_champion_name"]
    )

    previous_model_path.unlink()

    registry_path = rollback_environment["registry_path"]
    original_registry = registry_path.read_text(encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Previous champion artifact"):
        rollback_champion(
            registry_path=registry_path,
            model_dir=rollback_environment["model_dir"],
        )

    # Registry must remain unchanged when rollback fails.
    assert registry_path.read_text(encoding="utf-8") == original_registry


def test_repeated_rollback_is_rejected(rollback_environment):
    """Verify that the same promotion cannot be rolled back twice."""

    registry_path = rollback_environment["registry_path"]
    model_dir = rollback_environment["model_dir"]

    rollback_champion(
        registry_path=registry_path,
        model_dir=model_dir,
    )

    with pytest.raises(ValueError, match="latest registry event"):
        rollback_champion(
            registry_path=registry_path,
            model_dir=model_dir,
        )
