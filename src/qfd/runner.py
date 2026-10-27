"""Run an experiment: split, preprocess, subsample, fit, score and log results."""

import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import orjson

from qfd.config import ExperimentConfig, ModelConfig, Stage
from qfd.data import (
    Dataset,
    build_preprocessor,
    enriched_subsample,
    load_dataset,
    negative_subsample,
    one_class_subsample,
    temporal_split,
)
from qfd.evaluation import evaluate
from qfd.hardware import (
    HardwareBudgetError,
    check_hardware_budget,
    estimate_kernel_circuits,
)
from qfd.hardware.dwave import REMOTE_SAMPLERS
from qfd.registry import HARDWARE_BACKENDS, QUANTUM_MODELS, build_detector
from qfd.types import Detector, FloatArray, IntArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Metrics of one model on one repetition."""

    model: str
    repetition: int
    n_train: int
    n_test: int
    fit_seconds: float
    score_seconds: float
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class PreparedData:
    """Preprocessed training and test data of one experiment."""

    X_train: FloatArray
    y_train: IntArray
    X_test: FloatArray
    y_test: IntArray


def prepare_data(config: ExperimentConfig, dataset: Dataset) -> PreparedData:
    """Split in time and fit preprocessing on the training period only.

    Args:
        config: Experiment configuration.
        dataset: Loaded dataset.

    Returns:
        Transformed training and test sets (validation is held out).
    """
    split = temporal_split(
        dataset.time, config.data.val_fraction, config.data.test_fraction
    )
    pre = build_preprocessor(config.preprocessing, config.seed)
    X_train = pre.fit_transform(dataset.X[split.train], dataset.y[split.train])
    return PreparedData(
        X_train=np.asarray(X_train, dtype=np.float64),
        y_train=dataset.y[split.train],
        X_test=np.asarray(pre.transform(dataset.X[split.test]), dtype=np.float64),
        y_test=dataset.y[split.test],
    )


def _train_indices(
    detector: Detector, config: ExperimentConfig, y: IntArray, rng: np.random.Generator
) -> IntArray:
    """Training subsample for one detector (docs/06 section 5.2)."""
    n = config.subsample.n_train
    if n is None:
        return np.arange(len(y), dtype=np.int64)
    if detector.is_one_class:
        return one_class_subsample(y, n, rng)
    return enriched_subsample(y, n, config.subsample.fraud_rate, rng)


def _guard_hardware(
    model: ModelConfig,
    config: ExperimentConfig,
    sizes: tuple[int, int],
    *,
    allowed: bool,
) -> None:
    """Apply the hardware budget check to quantum models on gate-based hardware."""
    on_hardware = config.device.backend in HARDWARE_BACKENDS
    if config.stage is Stage.HARDWARE and on_hardware and model.name in QUANTUM_MODELS:
        check_hardware_budget(
            estimate_kernel_circuits(*sizes),
            config.max_hardware_circuits,
            allowed=allowed,
        )


def _guard_annealer(config: ExperimentConfig, *, allowed: bool) -> None:
    """Require confirmation before QUBO feature selection on D-Wave Leap."""
    pre = config.preprocessing
    if pre.reduction == "qubo" and pre.qubo_sampler in REMOTE_SAMPLERS and not allowed:
        raise HardwareBudgetError(
            "QUBO feature selection submits problems to D-Wave Leap. "
            "Re-run with --allow-hardware to confirm."
        )


def run_experiment(
    config: ExperimentConfig, *, allow_hardware: bool = False
) -> list[RunResult]:
    """Run every model on every repetition and append results to a JSONL file.

    Args:
        config: Experiment configuration.
        allow_hardware: Confirmation that paid hardware may be used.

    Returns:
        All run results.
    """
    _guard_annealer(config, allowed=allow_hardware)
    data = prepare_data(config, load_dataset(config.data, config.seed))
    n_qubits = config.preprocessing.n_features
    out_dir = (
        config.output_dir / config.name / datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    for rep in range(config.repetitions):
        rng = np.random.default_rng(config.seed + rep)
        test_idx, weights = negative_subsample(
            data.y_test, config.subsample.test_negative_rate, rng
        )
        for model in config.models:
            max_train = config.subsample.n_train or len(data.y_train)
            _guard_hardware(
                model, config, (max_train, len(test_idx)), allowed=allow_hardware
            )
            detector = build_detector(
                model, config.stage, config.device, n_qubits, config.seed + rep
            )
            train_idx = _train_indices(detector, config, data.y_train, rng)
            result = _fit_and_score(
                detector, model.name, rep, data, (train_idx, test_idx, weights)
            )
            results.append(result)
            _append_jsonl(out_dir / "results.jsonl", result)
            logger.info(
                "%s rep %d: AUPRC %.4f", model.name, rep, result.metrics["auprc"]
            )
    return results


def _fit_and_score(
    detector: Detector,
    name: str,
    rep: int,
    data: PreparedData,
    indices: tuple[IntArray, IntArray, FloatArray],
) -> RunResult:
    """Fit one detector, score the test set and compute metrics."""
    train_idx, test_idx, weights = indices
    start = time.perf_counter()
    detector.fit(data.X_train[train_idx], data.y_train[train_idx])
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    scores = detector.fraud_score(data.X_test[test_idx])
    score_seconds = time.perf_counter() - start
    return RunResult(
        model=name,
        repetition=rep,
        n_train=len(train_idx),
        n_test=len(test_idx),
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        metrics=evaluate(data.y_test[test_idx], scores, weights),
    )


def _append_jsonl(path: Path, result: RunResult) -> None:
    """Append one result as a JSON line."""
    with path.open("ab") as fh:
        fh.write(orjson.dumps(asdict(result)) + b"\n")
