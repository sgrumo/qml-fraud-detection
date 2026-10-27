from dataclasses import replace
from pathlib import Path

import orjson
import pytest

from qfd.cli import main
from qfd.config import DeviceConfig, Stage, load_config
from qfd.hardware import HardwareBudgetError
from qfd.runner import run_experiment


def test_classical_smoke_run_writes_results(tmp_path: Path) -> None:
    config = replace(
        load_config(Path("configs/classical/smoke.yaml")), output_dir=tmp_path
    )

    results = run_experiment(config)

    lines = next(tmp_path.rglob("results.jsonl")).read_bytes().splitlines()
    assert len(results) == len(lines) == 4
    assert {orjson.loads(line)["model"] for line in lines} == {
        "logistic_regression",
        "isolation_forest",
    }


@pytest.mark.slow
def test_simulation_smoke_run(tmp_path: Path) -> None:
    config = replace(
        load_config(Path("configs/simulation/smoke.yaml")), output_dir=tmp_path
    )

    results = run_experiment(config)

    assert [r.model for r in results] == ["qsvc", "qocsvm", "vqc"]
    assert all(0.0 <= r.metrics["auprc"] <= 1.0 for r in results)


def test_quantum_models_rejected_in_classical_stage(tmp_path: Path) -> None:
    config = load_config(Path("configs/simulation/smoke.yaml"))
    config = replace(config, stage=Stage.CLASSICAL, output_dir=tmp_path)

    with pytest.raises(ValueError, match="classical stage"):
        run_experiment(config)


def test_hardware_run_requires_confirmation(tmp_path: Path) -> None:
    config = load_config(Path("configs/simulation/smoke.yaml"))
    config = replace(
        config,
        stage=Stage.HARDWARE,
        device=DeviceConfig(backend="braket", shots=10, options={"device_arn": "arn"}),
        output_dir=tmp_path,
    )

    with pytest.raises(HardwareBudgetError, match="--allow-hardware"):
        run_experiment(config)


def test_cli_runs_config(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    text = Path("configs/classical/smoke.yaml").read_text()
    config_path.write_text(text + f"output_dir: {tmp_path / 'out'}\n")

    main(["--log-level", "WARNING", "run", str(config_path)])

    assert next((tmp_path / "out").rglob("results.jsonl")).exists()
