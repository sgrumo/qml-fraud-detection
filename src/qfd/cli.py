"""Command-line entry point.

``qfd download ulb`` fetches the dataset; ``qfd run <config>`` runs an
experiment from ``configs/<stage>/``.
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from qfd.config import load_config
from qfd.data.download import download_ulb
from qfd.runner import run_experiment


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="qfd", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an experiment from a YAML config")
    run.add_argument("config", type=Path)
    run.add_argument(
        "--allow-hardware",
        action="store_true",
        help="Confirm that jobs may be submitted to paid quantum hardware",
    )

    download = sub.add_parser("download", help="Download a dataset from Kaggle")
    download.add_argument("dataset", choices=["ulb"])
    download.add_argument("--dest", type=Path, default=Path("data/raw"))
    download.add_argument("--force", action="store_true", help="Download again")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the requested command.

    Args:
        argv: Arguments; defaults to ``sys.argv``.
    """
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    load_dotenv()
    if args.command == "download":
        download_ulb(args.dest, force=args.force)
    else:
        run_experiment(load_config(args.config), allow_hardware=args.allow_hardware)


if __name__ == "__main__":
    main()
