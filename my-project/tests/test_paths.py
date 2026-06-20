"""Tests for portable paths and non-mutating data.yaml materialization."""
import os
import sys
import types
import importlib

import pytest

# task.py references ultralytics only in type hints; stub it so the module
# imports without the heavy dependency (CI installs the real package).
if "ultralytics" not in sys.modules:
    stub = types.ModuleType("ultralytics")
    stub.YOLO = object
    sys.modules["ultralytics"] = stub


def _fresh_task(monkeypatch, data_root=None):
    """Import (or reimport) my_project.task with an optional FL_AV_DATA_ROOT."""
    if data_root is None:
        monkeypatch.delenv("FL_AV_DATA_ROOT", raising=False)
    else:
        monkeypatch.setenv("FL_AV_DATA_ROOT", data_root)
    sys.modules.pop("my_project.task", None)
    return importlib.import_module("my_project.task")


def test_base_data_path_defaults_to_package_parent(monkeypatch):
    task = _fresh_task(monkeypatch)
    # Default base path must contain the batch dirs and never be a Windows path.
    assert "C:\\" not in task.BASE_DATA_PATH and "C:/" not in task.BASE_DATA_PATH
    assert os.path.isdir(os.path.join(task.BASE_DATA_PATH, "batch"))


def test_base_data_path_env_override(monkeypatch, tmp_path):
    task = _fresh_task(monkeypatch, data_root=str(tmp_path))
    assert task.BASE_DATA_PATH == str(tmp_path)
    assert str(task.get_batch_path(1)).startswith(str(tmp_path))


def test_materialize_writes_runtime_not_source(monkeypatch):
    task = _fresh_task(monkeypatch)
    src = task.get_data_yaml_path(1)
    before = src.read_text()  # tracked file content

    runtime = task.materialize_data_yaml(1)

    # A separate runtime file was produced, leaving the tracked file byte-identical.
    assert runtime.name == "data.runtime.yaml"
    assert runtime != src
    assert runtime.exists()
    assert src.read_text() == before, "tracked data.yaml must not be mutated"

    import yaml
    data = yaml.safe_load(runtime.read_text())
    # path is now an absolute, machine-correct directory for batch_1.
    assert os.path.isabs(data["path"])
    assert data["path"].endswith(os.path.join("batch", "batch_1"))
    assert "C:\\" not in data["path"]

    runtime.unlink()  # cleanup gitignored artifact


def test_no_windows_path_in_task_source():
    """Regression guard: the hardcoded Windows base path must be gone."""
    here = os.path.dirname(__file__)
    src = os.path.join(here, "..", "my_project", "task.py")
    text = open(src).read()
    assert "C:/Users" not in text and "C:\\Users" not in text
