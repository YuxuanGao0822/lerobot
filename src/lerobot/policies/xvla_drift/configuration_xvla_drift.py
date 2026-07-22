import math
from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig


@PreTrainedConfig.register_subclass("xvla_drift")
@dataclass
class XVLADriftConfig(XVLAConfig):
    """X-VLA with its own direct clean-action Drifting objective."""

    use_drifting_loss: bool = True
    drifting_gen_per_label: int = 8
    drifting_temperatures: tuple[float, ...] = (0.02, 0.05, 0.2)
    drifting_per_timestep_loss: bool = False
    drifting_perdim_loss: bool = True
    test_time_samples: int = 1
    test_time_clusters: int = 2
    test_time_unimodal_tau: float = 0.3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.drifting_gen_per_label < 2:
            raise ValueError("`drifting_gen_per_label` must be >= 2.")
        temperatures = tuple(float(value) for value in self.drifting_temperatures)
        if not temperatures or any(not math.isfinite(value) or value <= 0 for value in temperatures):
            raise ValueError("`drifting_temperatures` must contain finite positive values.")
        self.drifting_temperatures = temperatures
        if self.drifting_perdim_loss and self.drifting_per_timestep_loss:
            raise ValueError("Per-dimension and per-timestep drifting losses are mutually exclusive.")
        if self.test_time_samples < 1:
            raise ValueError("`test_time_samples` must be >= 1.")
        if self.test_time_samples > 1 and self.test_time_clusters < 2:
            raise ValueError("`test_time_clusters` must be >= 2 when KeyStone is enabled.")
        if self.test_time_unimodal_tau <= 0:
            raise ValueError("`test_time_unimodal_tau` must be > 0.")
