"""Pipeline and fleet observability for the federated-YOLOv8 project.

This package *invokes* `my-project` and *reads* its outputs. It never imports its
internals and never edits its source — see tests/test_isolation.py, which enforces it.

Observability is assembled, not built: MLflow owns metrics and history, the Ray
Dashboard owns actor and GPU internals. The only custom UI is what neither can do —
launching runs, and narrating a fleet of simulated vehicles.
"""
__all__ = ["paths", "gpu", "vehicles", "logparse", "stages", "runner", "report"]
