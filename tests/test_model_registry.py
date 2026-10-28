import json
import pytest
from src.self_healing_ml.orchestration.model_registry import load_registry, save_registry


def test_load_registry_when_file_does_not_exist(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = load_registry(registry_path)

    assert registry == {
        "active_champion": None,
        "models": [],
        "history": [],
    }


def test_save_and_load_registry(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = {
        "active_champion": "tuned_logistic_regression",
        "models": [
            {
                "name": "tuned_logistic_regression",
                "version": "1",
                "path": "models/tuned_logistic_regression.joblib",
            }
        ],
        "history": [],
    }

    save_registry(registry, registry_path)
    loaded_registry = load_registry(registry_path)

    assert loaded_registry == registry


def test_save_registry_creates_parent_directory(tmp_path):
    registry_path = tmp_path / "nested" / "registry.json"
    registry = {
        "active_champion": None,
        "models": [],
        "history": [],
    }
    save_registry(registry, registry_path)

    assert registry_path.exists()
    assert load_registry(registry_path) == registry


def test_load_registry_rejects_invalid_structure(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"active_champion": None}),  encoding="utf-8" )

    with pytest.raises(ValueError, match="invalid structure"):
        load_registry(registry_path)


def test_load_registry_rejects_non_list_models(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({
            "active_champion": None,
            "models": {},
            "history": [],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a list"):
        load_registry(registry_path)


def test_load_registry_rejects_non_list_history(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({
            "active_champion": None,
            "models": [],
            "history": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a list"):
        load_registry(registry_path)


def test_save_registry_rejects_invalid_structure(tmp_path):
    registry_path = tmp_path / "registry.json"

    with pytest.raises(ValueError, match="invalid structure"):
        save_registry({"active_champion": None}, registry_path)


def test_save_registry_rejects_non_list_models(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = {
        "active_champion": None,
        "models": {},
        "history": [],
    }

    with pytest.raises(ValueError, match="must be a list"):
        save_registry(registry, registry_path)


def test_save_registry_rejects_non_list_history(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = {
        "active_champion": None,
        "models": [],
        "history": {},
    }

    with pytest.raises(ValueError, match="must be a list"):
        save_registry(registry, registry_path)