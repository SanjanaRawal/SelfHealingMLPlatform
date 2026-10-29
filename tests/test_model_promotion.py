import json
import pytest
from src.self_healing_ml.orchestration.model_promotion import promote_challenger


@pytest.fixture
def promotion_setup(tmp_path):
    """Create temporary model artifacts and an evaluation report."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    champion_name = "tuned_logistic_regression.joblib"
    challenger_name = "tuned_gradient_boosting.joblib"

    (model_dir / champion_name).write_bytes(b"test champion artifact")
    (model_dir / challenger_name).write_bytes(b"test challenger artifact")

    report_path = tmp_path / "reports" / "champion_challenger_report.json"
    report_path.parent.mkdir(parents=True)

    report = {
        "champion_model": champion_name,
        "challenger_model": challenger_name,
        "champion_metrics": {
            "accuracy": 0.74,
            "precision": 0.51,
            "recall": 0.79,
            "f1_score": 0.61,
            "roc_auc": 0.84,
        },
        "challenger_metrics": {
            "accuracy": 0.80,
            "precision": 0.66,
            "recall": 0.52,
            "f1_score": 0.58,
            "roc_auc": 0.85,
        },
        "decision": "accept_challenger",
    }

    report_path.write_text(json.dumps(report), encoding="utf-8")
    registry_path = tmp_path / "registry" / "registry.json"

    return {
        "model_dir": model_dir,
        "report_path": report_path,
        "registry_path": registry_path,
        "report": report,
    }


def test_rejected_challenger_does_not_create_registry(promotion_setup):
    setup = promotion_setup
    setup["report"]["decision"] = "reject_challenger"
    setup["report_path"].write_text(
        json.dumps(setup["report"]),
        encoding="utf-8",
    )

    result = promote_challenger(
        report_path=setup["report_path"],
        registry_path=setup["registry_path"],
        model_dir=setup["model_dir"],
    )

    assert result["promoted"] is False
    assert result["reason"] == "challenger_rejected"
    assert not setup["registry_path"].exists()


def test_accepted_challenger_is_promoted(promotion_setup):
    setup = promotion_setup

    result = promote_challenger(
        report_path=setup["report_path"],
        registry_path=setup["registry_path"],
        model_dir=setup["model_dir"],
    )

    assert result["promoted"] is True
    assert result["active_champion"] == ("tuned_gradient_boosting.joblib")
    assert setup["registry_path"].exists()


def test_promotion_updates_registry_and_history(promotion_setup):
    setup = promotion_setup

    promote_challenger(
        report_path=setup["report_path"],
        registry_path=setup["registry_path"],
        model_dir=setup["model_dir"],
    )

    registry = json.loads(setup["registry_path"].read_text(encoding="utf-8"))

    assert registry["active_champion"] == ("tuned_gradient_boosting.joblib")
    assert len(registry["models"]) == 2
    assert len(registry["history"]) == 1
    assert registry["history"][0]["action"] == "promote_challenger"
    assert registry["history"][0]["decision"] == "accept_challenger"


def test_promotion_fails_when_model_artifact_is_missing(promotion_setup):
    setup = promotion_setup
    (setup["model_dir"] / "tuned_gradient_boosting.joblib").unlink()

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        promote_challenger(
            report_path=setup["report_path"],
            registry_path=setup["registry_path"],
            model_dir=setup["model_dir"],
        )

    assert not setup["registry_path"].exists()


def test_promotion_rejects_invalid_report_structure(promotion_setup):
    setup = promotion_setup
    setup["report_path"].write_text(
        json.dumps({"decision": "accept_challenger"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid structure"):
        promote_challenger(
            report_path=setup["report_path"],
            registry_path=setup["registry_path"],
            model_dir=setup["model_dir"],
        )


def test_promotion_rejects_invalid_decision(promotion_setup):
    setup = promotion_setup
    setup["report"]["decision"] = "maybe"
    setup["report_path"].write_text(
        json.dumps(setup["report"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid decision"):
        promote_challenger(
            report_path=setup["report_path"],
            registry_path=setup["registry_path"],
            model_dir=setup["model_dir"],
        )
