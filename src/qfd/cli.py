"""Command-line entry point: ``qfd run configs/<stage>/<experiment>.yaml``."""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from qfd.config import load_config
from qfd.runner import run_experiment


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="qfd", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run an experiment from a YAML config")
    run.add_argument("config", type=Path)
    run.add_argument(
        "--allow-hardware",
        action="store_true",
        help="Confirm that jobs may be submitted to paid quantum hardware",
    )
    run.add_argument("--log-level", default="INFO")
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
    run_experiment(load_config(args.config), allow_hardware=args.allow_hardware)


if __name__ == "__main__":
    main()
