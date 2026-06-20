"""ARCHIVED legacy path helpers (no longer wired into the FL flow).

Preserved for reference per the project's "archive, don't delete" policy. These
were replaced because:

- ``update_data_yaml_paths`` rewrote the *tracked* ``batch/*/data.yaml`` in place
  on every run, dirtying the git working tree and baking machine-specific
  absolute paths into version control. Superseded by
  ``my_project.task.materialize_data_yaml`` which writes a gitignored
  ``data.runtime.yaml`` instead and never touches the committed file.
- ``get_normalized_path`` did manual OS slash conversion, made obsolete by using
  ``os.path``/``pathlib`` and the env-driven ``BASE_DATA_PATH``.

Nothing imports this module; it is documentation, not live code.
"""
import os
import yaml


def get_normalized_path(path, is_windows=None):
    """Legacy: convert path separators for the current OS."""
    if is_windows is None:
        import platform
        is_windows = platform.system() == "Windows"
    if is_windows:
        return str(path).replace("/", "\\")
    return str(path).replace("\\", "/")


def update_data_yaml_paths(yaml_path, batch_path):
    """Legacy: rewrite the tracked data.yaml 'path' in place (caused tree churn)."""
    if not os.path.exists(yaml_path):
        return False
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        data["path"] = str(batch_path)
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        return True
    except Exception:
        return False
