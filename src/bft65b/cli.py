"""Phase 0 CLI; data pipeline commands are intentionally not available yet."""

import argparse
import json
from pathlib import Path
from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bft65b", description="BFT65B project foundation")
    parser.add_argument("--config", type=Path, help="JSON configuration override")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show-config", help="Validate and print configuration")
    commands.add_parser("status", help="Show implemented stage status")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        print("Phase 0 foundation only. Acquisition and modelling stages are not implemented.")
        return 0
    try:
        config = load_config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        build_parser().error(str(exc))
    print(json.dumps(config.to_dict(), indent=2))
    return 0
