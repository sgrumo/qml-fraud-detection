from pathlib import Path

import kagglehub
import pytest
from kagglehub.exceptions import UnauthenticatedError

from qfd.cli import main
from qfd.data.download import DataDownloadError, download_ulb, sha256sum

CSV = b"Time,V1,Amount,Class\n0,0.1,1.0,0\n"


@pytest.fixture
def fake_kaggle(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Replace the Kaggle download with one that writes a tiny CSV."""
    calls: list[dict[str, object]] = []

    def fake_download(handle: str, **kwargs: object) -> str:
        calls.append({"handle": handle, **kwargs})
        out = Path(str(kwargs["output_dir"])) / "creditcard.csv"
        out.write_bytes(CSV)
        return str(out)

    monkeypatch.setenv("KAGGLE_API_TOKEN", "test-token")
    monkeypatch.setattr(kagglehub, "dataset_download", fake_download)
    return calls


def test_downloads_file_and_writes_checksum(
    tmp_path: Path, fake_kaggle: list[dict[str, object]]
) -> None:
    path = download_ulb(tmp_path)

    assert path == tmp_path / "creditcard.csv"
    assert fake_kaggle[0]["handle"] == "mlg-ulb/creditcardfraud"
    checksum = (tmp_path / "creditcard.csv.sha256").read_text()
    assert checksum.startswith(sha256sum(path))


def test_existing_file_is_not_downloaded_again(
    tmp_path: Path, fake_kaggle: list[dict[str, object]]
) -> None:
    (tmp_path / "creditcard.csv").write_bytes(CSV)

    download_ulb(tmp_path)

    assert fake_kaggle == []


def test_force_downloads_again(
    tmp_path: Path, fake_kaggle: list[dict[str, object]]
) -> None:
    (tmp_path / "creditcard.csv").write_bytes(b"old")

    download_ulb(tmp_path, force=True)

    assert fake_kaggle[0]["force_download"] is True
    assert (tmp_path / "creditcard.csv").read_bytes() == CSV


def test_missing_token_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    with pytest.raises(DataDownloadError, match="KAGGLE_API_TOKEN"):
        download_ulb(tmp_path)


def test_rejected_token_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(handle: str, **kwargs: object) -> str:
        raise UnauthenticatedError()

    monkeypatch.setenv("KAGGLE_API_TOKEN", "bad-token")
    monkeypatch.setattr(kagglehub, "dataset_download", reject)

    with pytest.raises(DataDownloadError, match="check KAGGLE_API_TOKEN"):
        download_ulb(tmp_path)


def test_cli_download(tmp_path: Path, fake_kaggle: list[dict[str, object]]) -> None:
    main(["--log-level", "WARNING", "download", "ulb", "--dest", str(tmp_path)])

    assert (tmp_path / "creditcard.csv").is_file()
