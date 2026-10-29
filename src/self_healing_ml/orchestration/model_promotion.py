import json
from datetime import datetime, timezone
from pathlib import Path
from self_healing_ml.orchestration.model_registry import load_registry, save_registry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_PATH = PROJECT_ROOT / "reports" / "champion_challenger_report.json"
REGISTRY_PATH = MODEL_DIR / "registry.json"


def _utc_timestamp():
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _read_evaluation_report(report_path):
    """Load and validate a champion-challenger evaluation report."""
    report_path = Path(report_path)

    if not report_path.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {report_path}")

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    required_keys = {
        "champion_model",
        "challenger_model",
        "champion_metrics",
        "challenger_metrics",
        "decision",
    }

    if not isinstance(report, dict) or not required_keys.issubset(report):
        raise ValueError("Evaluation report has an invalid structure.")

    if report["decision"] not in {
        "accept_challenger",
        "reject_challenger",
    }:
        raise ValueError("Evaluation report contains an invalid decision.")

    for key in ("champion_model", "challenger_model"):
        name = report[key]

        if (
            not isinstance(name, str)
            or not name.strip()
            or Path(name).name != name
            or name in {".", ".."}
        ):
            raise ValueError(f"Invalid model artifact name: {name!r}")

    if report["champion_model"] == report["challenger_model"]:
        raise ValueError("Champion and challenger must be different models.")

    return report


def _upsert_model(registry, model_name, metrics, status):
    """Add model metadata or update an existing entry."""
    for model in registry["models"]:
        if model.get("model_name") == model_name:
            model.update({
                "artifact_path": str(MODEL_DIR / model_name),
                "metrics": metrics,
                "status": status,
            })
            return

    registry["models"].append({
        "model_name": model_name,
        "artifact_path": str(MODEL_DIR / model_name),
        "metrics": metrics,
        "status": status,
        "registered_at": _utc_timestamp(),
    })


def promote_challenger(report_path=REPORT_PATH,
    registry_path=REGISTRY_PATH, model_dir=MODEL_DIR):
    """
    Promote an accepted challenger and record the decision in the registry.
    A rejected challenger must not modify the registry.
    """
    report = _read_evaluation_report(report_path)

    if report["decision"] != "accept_challenger":
        print("Promotion skipped: challenger was rejected.")
        return {
            "promoted": False,
            "reason": "challenger_rejected",
            "active_champion": None,
        }

    model_dir = Path(model_dir)
    champion_name = report["champion_model"]
    challenger_name = report["challenger_model"]

    champion_path = model_dir / champion_name
    challenger_path = model_dir / challenger_name

    for model_path in (champion_path, challenger_path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

    registry = load_registry(registry_path)
    previous_champion = registry["active_champion"]

    # Register or refresh the evaluated models.
    _upsert_model(registry, champion_name, report["champion_metrics"], "previous_champion")
    _upsert_model(registry,challenger_name,report["challenger_metrics"],"champion")

    # Keep exactly one model marked as the active champion.
    for model in registry["models"]:
        if model.get("model_name") == challenger_name:
            model["status"] = "champion"
        elif model.get("model_name") == champion_name:
            model["status"] = "previous_champion"
        elif model.get("status") == "champion":
            model["status"] = "previous_champion"

    registry["active_champion"] = challenger_name

    registry["history"].append({
        "timestamp": _utc_timestamp(),
        "action": "promote_challenger",
        "previous_champion": previous_champion or champion_name,
        "new_champion": challenger_name,
        "decision": report["decision"],
        "champion_metrics": report["champion_metrics"],
        "challenger_metrics": report["challenger_metrics"],
    })

    save_registry(registry, registry_path)

    result = {
        "promoted": True,
        "reason": "acceptance_criteria_met",
        "previous_champion": previous_champion or champion_name,
        "active_champion": challenger_name,
    }

    print(json.dumps(result, indent=4))
    print(f"Registry updated: {registry_path}")

    return result


if __name__ == "__main__":
    promote_challenger()