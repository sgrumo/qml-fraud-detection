from pathlib import Path

import pytest

from qfd.config import Stage, load_config

CONFIGS = sorted(Path("configs").rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=str)
def test_all_shipped_configs_parse(path: Path) -> None:
    config = load_config(path)

    assert config.models
    assert config.stage.value == path.parent.name


def test_smoke_config_values() -> None:
    config = load_config(Path("configs/simulation/smoke.yaml"))

    assert config.stage is Stage.SIMULATION
    assert config.preprocessing.n_features == 3
    assert config.models[0].params["feature_map"] == "iqp"


def test_missing_models_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nstage: classical\nmodels: []\n")

    with pytest.raises(ValueError, match="models"):
        load_config(path)
