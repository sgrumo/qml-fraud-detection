"""Download public datasets from Kaggle with an API token.

Authentication uses the ``KAGGLE_API_TOKEN`` environment variable (put it in
``.env``; create the token at https://www.kaggle.com/settings/api). The token
is read by ``kagglehub`` and never logged.
"""

import hashlib
import logging
import os
from pathlib import Path

import kagglehub
from kagglehub.exceptions import (
    KaggleApiHTTPError,
    NotFoundError,
    UnauthenticatedError,
)

logger = logging.getLogger(__name__)

ULB_HANDLE = "mlg-ulb/creditcardfraud"
ULB_FILE = "creditcard.csv"


class DataDownloadError(RuntimeError):
    """A dataset could not be downloaded."""


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file.

    Args:
        path: File to hash.
        chunk_size: Bytes read per iteration.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_ulb(dest_dir: Path = Path("data/raw"), *, force: bool = False) -> Path:
    """Download the ULB credit-card fraud dataset into ``dest_dir``.

    The file's SHA-256 is written next to it (``creditcard.csv.sha256``) so
    experiments can record exactly which data they used.

    Args:
        dest_dir: Destination directory.
        force: Download again even if the file already exists.

    Returns:
        Path to ``creditcard.csv``.

    Raises:
        DataDownloadError: If the token is missing or Kaggle rejects the request.
    """
    target = dest_dir / ULB_FILE
    if target.is_file() and not force:
        logger.info("%s already exists; use --force to download again", target)
        return target
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise DataDownloadError(
            "Set KAGGLE_API_TOKEN in .env "
            "(create a token at https://www.kaggle.com/settings/api)"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        kagglehub.dataset_download(
            ULB_HANDLE, path=ULB_FILE, output_dir=str(dest_dir), force_download=force
        )
    except (UnauthenticatedError, KaggleApiHTTPError, NotFoundError) as exc:
        raise DataDownloadError(
            f"Kaggle download of {ULB_HANDLE} failed; check KAGGLE_API_TOKEN"
        ) from exc
    if not target.is_file():
        raise DataDownloadError(f"Download finished but {target} is missing")
    digest = sha256sum(target)
    target.with_name(f"{ULB_FILE}.sha256").write_text(f"{digest}  {ULB_FILE}\n")
    logger.info("Downloaded %s (sha256 %s)", target, digest)
    return target
