import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODEL_DIR / "registry.json"


def _utc_timestamp():
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_registry(registry_path=REGISTRY_PATH):
    """Load the registry, or return an empty initial registry."""
    registry_path = Path(registry_path)

    if not registry_path.exists():
        return {
            "active_champion": None,
            "models": [],
            "history": [],
        }

    with registry_path.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    required_keys = {"active_champion", "models", "history"}
    if not isinstance(registry, dict) or not required_keys.issubset(registry):
        raise ValueError("Registry file has an invalid structure.")

    if not isinstance(registry["models"], list):
        raise ValueError("Registry 'models' must be a list.")

    if not isinstance(registry["history"], list):
        raise ValueError("Registry 'history' must be a list.")

    return registry


def save_registry(registry, registry_path=REGISTRY_PATH):
    """Validate and save registry data as JSON."""
    required_keys = {"active_champion", "models", "history"}

    if not isinstance(registry, dict) or not required_keys.issubset(registry):
        raise ValueError("Registry has an invalid structure.")

    if not isinstance(registry["models"], list):
        raise ValueError("Registry 'models' must be a list.")

    if not isinstance(registry["history"], list):
        raise ValueError("Registry 'history' must be a list.")

    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = registry_path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)

    temporary_path.replace(registry_path)
