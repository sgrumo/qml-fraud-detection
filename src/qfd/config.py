"""Experiment configuration loaded from YAML files."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast, get_args

import yaml

from qfd.types import Params


class Stage(StrEnum):
    """Project stage an experiment belongs to."""

    CLASSICAL = "classical"
    SIMULATION = "simulation"
    HARDWARE = "hardware"


type Reduction = Literal["first", "pca", "mutual_info", "qubo"]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the data comes from and how it is split in time."""

    source: Literal["ulb", "synthetic"] = "ulb"
    path: Path | None = None
    val_fraction: float = 0.1
    test_fraction: float = 0.3


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Feature reduction to the qubit budget and scaling to rotation angles."""

    n_features: int = 4
    reduction: Reduction = "pca"
    qubo_sampler: str = "simulated"

    def __post_init__(self) -> None:
        """Validate the reduction method and feature budget."""
        if self.reduction not in get_args(Reduction.__value__):
            raise ValueError(f"Unknown reduction '{self.reduction}'")
        if self.n_features < 1:
            raise ValueError("n_features must be positive")


@dataclass(frozen=True, slots=True)
class SubsampleConfig:
    """Training and test subsampling under quantum compute budgets.

    ``n_train=None`` uses the full training period. ``test_negative_rate`` is
    the fraction of genuine test transactions kept; metrics are reweighted to
    the natural prevalence.
    """

    n_train: int | None = 500
    fraud_rate: float = 0.05
    test_negative_rate: float = 1.0


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Quantum execution backend.

    ``backend`` is ``statevector`` (exact), any PennyLane device name
    (``default.qubit``, ``lightning.qubit``, ``default.mixed``, ``cirq.qsim``,
    ``braket.local.qubit``), ``braket`` (AWS hardware or managed simulators) or
    ``google_engine``.
    """

    backend: str = "statevector"
    shots: int | None = None
    depolarizing: float | None = None
    options: Params = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """A model name from the registry and its hyperparameters."""

    name: str
    params: Params = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Full description of one experiment."""

    name: str
    stage: Stage
    models: tuple[ModelConfig, ...]
    data: DataConfig = DataConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    subsample: SubsampleConfig = SubsampleConfig()
    device: DeviceConfig = DeviceConfig()
    repetitions: int = 10
    seed: int = 0
    max_hardware_circuits: int = 10_000
    output_dir: Path = Path("results")


def _section[T](raw: dict[str, object], key: str, cls: type[T]) -> T:
    """Build a config dataclass from a YAML sub-mapping."""
    values = raw.get(key) or {}
    if not isinstance(values, dict):
        raise TypeError(f"Section '{key}' must be a mapping")
    return cls(**values)


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration.

    Args:
        path: YAML configuration file.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If required fields are missing or invalid.
        TypeError: If a section has the wrong structure.
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    raw = cast(dict[str, object], raw)
    models_raw = raw.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise ValueError("'models' must be a non-empty list")
    data = _section(raw, "data", DataConfig)
    if data.path is not None:
        data = DataConfig(
            source=data.source,
            path=Path(data.path),
            val_fraction=data.val_fraction,
            test_fraction=data.test_fraction,
        )
    return ExperimentConfig(
        name=str(raw["name"]),
        stage=Stage(str(raw["stage"])),
        models=tuple(ModelConfig(**m) for m in models_raw),
        data=data,
        preprocessing=_section(raw, "preprocessing", PreprocessingConfig),
        subsample=_section(raw, "subsample", SubsampleConfig),
        device=_section(raw, "device", DeviceConfig),
        repetitions=int(cast(int, raw.get("repetitions", 10))),
        seed=int(cast(int, raw.get("seed", 0))),
        max_hardware_circuits=int(cast(int, raw.get("max_hardware_circuits", 10_000))),
        output_dir=Path(str(raw.get("output_dir", "results"))),
    )
