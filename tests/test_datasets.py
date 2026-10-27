from pathlib import Path

import numpy as np
import pytest

from qfd.data.datasets import ULB_FEATURES, load_ulb

HEADER = ",".join(f'"{c}"' for c in ("Time", *ULB_FEATURES, "Class"))


def _row(time: str, amount: float, label: int) -> str:
    return ",".join([time, *["0.5"] * 28, str(amount), f'"{label}"'])


def test_load_ulb_parses_kaggle_format(tmp_path: Path) -> None:
    path = tmp_path / "creditcard.csv"
    rows = [_row("1e+05", 9.0, 1), _row("0", 1.0, 0), _row("0", 1.0, 0)]
    path.write_text("\n".join([HEADER, *rows]) + "\n")

    data = load_ulb(path)

    np.testing.assert_array_equal(data.time, [0.0, 1e5])
    np.testing.assert_array_equal(data.y, [0, 1])
    assert data.X.shape == (2, 29)
    assert data.X[1, -1] == pytest.approx(np.log1p(9.0))


def test_load_ulb_missing_file_explains_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="qfd download ulb"):
        load_ulb(tmp_path / "missing.csv")
