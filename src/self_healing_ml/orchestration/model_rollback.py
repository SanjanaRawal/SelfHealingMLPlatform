import json
from datetime import datetime, timezone
from pathlib import Path
from self_healing_ml.orchestration.model_registry import load_registry, save_registry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODEL_DIR / "registry.json"


def _utc_timestamp():
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def rollback_champion(
    registry_path=REGISTRY_PATH,
    model_dir=MODEL_DIR,
):
    """
    Restore the previous champion model using the latest promotion event.

    Rollback is allowed only when:
    - A registry with an active champion exists.
    - The latest registry event is a successful challenger promotion.
    - The promotion event identifies a previous champion.
    - The previous champion is registered.
    - The previous champion artifact exists.

    The registry is updated only after all validations succeed.
    """
    registry_path = Path(registry_path)
    model_dir = Path(model_dir)

    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Model registry not found: {registry_path}"
        )

    registry = load_registry(registry_path)

    active_champion = registry["active_champion"]

    if not isinstance(active_champion, str) or not active_champion.strip():
        raise ValueError(
            "Rollback is not possible because no active champion exists."
        )

    if not registry["history"]:
        raise ValueError(
            "Rollback is not possible because promotion history is empty."
        )

    # Only the latest promotion can be rolled back.
    latest_event = registry["history"][-1]

    if (
        not isinstance(latest_event, dict)
        or latest_event.get("action") != "promote_challenger"
        or latest_event.get("new_champion") != active_champion
    ):
        raise ValueError(
            "Rollback is not possible because the latest registry "
            "event is not a promotion of the active champion."
        )

    previous_champion = latest_event.get("previous_champion")

    if (
        not isinstance(previous_champion, str)
        or not previous_champion.strip()
        or previous_champion in {".", ".."}
        or Path(previous_champion).name != previous_champion
    ):
        raise ValueError(
            "The previous champion recorded in promotion history is invalid."
        )

    if previous_champion == active_champion:
        raise ValueError(
            "Rollback is not possible because both champion references "
            "are identical."
        )

    # Verify that the previous champion is registered.
    registered_model = next(
        (
            model
            for model in registry["models"]
            if model.get("model_name") == previous_champion
        ),
        None,
    )

    if registered_model is None:
        raise ValueError(
            f"Previous champion is not registered: {previous_champion}"
        )

    # Verify that the previous champion artifact exists.
    previous_model_path = model_dir / previous_champion

    if not previous_model_path.is_file():
        raise FileNotFoundError(
            f"Previous champion artifact not found: {previous_model_path}"
        )

    # Update model statuses.
    for model in registry["models"]:
        if model.get("model_name") == previous_champion:
            model["status"] = "champion"
        elif model.get("status") == "champion":
            model["status"] = "previous_champion"

    # Update the active champion.
    registry["active_champion"] = previous_champion

    # Record the rollback event.
    rollback_event = {
        "timestamp": _utc_timestamp(),
        "action": "rollback_champion",
        "previous_champion": active_champion,
        "new_champion": previous_champion,
        "reason": "manual_rollback",
    }

    registry["history"].append(rollback_event)

    # Persist the updated registry.
    save_registry(registry, registry_path)

    result = {
        "rolled_back": True,
        "reason": "previous_champion_restored",
        "previous_champion": active_champion,
        "active_champion": previous_champion,
    }

    print(json.dumps(result, indent=4))
    print(f"Registry updated: {registry_path}")

    return result


if __name__ == "__main__":
    try:
        rollback_champion()
    except FileNotFoundError as error:
        print(f"Rollback skipped: {error}")
    except ValueError as error:
        print(f"Rollback skipped: {error}")
