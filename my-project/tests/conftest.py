"""Shared pytest setup.

Puts the project root (``my-project/``) on sys.path so ``my_project`` and
``utils`` import whether or not the package was installed, and provides a stub
for ``ultralytics`` (only referenced in type hints by the modules under test) so
the fast unit tests don't require the heavy dependency.
"""
import os
import sys
import types

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Stub ultralytics if absent (real package is used when installed, e.g. full CI).
if "ultralytics" not in sys.modules:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        stub = types.ModuleType("ultralytics")
        stub.YOLO = object
        sys.modules["ultralytics"] = stub
