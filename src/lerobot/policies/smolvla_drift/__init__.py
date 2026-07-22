"""SmolVLA-Drift — a LeRobot out-of-tree plugin policy (Path A).

A faithful vendored copy of SmolVLA (fork zuoxingdong/smolvla-drift @ 15778da0) plus the
one-step "Drifting" (DBPO) objective and one-NFE inference. Provenance: README.md.
"""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package "
        "(e.g. `pip install 'lerobot[smolvla]>=0.6.0'`)."
    ) from exc

from .configuration_smolvla_drift import SmolVLADriftConfig
from .modeling_smolvla_drift import SmolVLADriftPolicy
from .processor_smolvla_drift import make_smolvla_drift_pre_post_processors

__all__ = [
    "SmolVLADriftConfig",
    "SmolVLADriftPolicy",
    "make_smolvla_drift_pre_post_processors",
]
