#!/usr/bin/env python3
"""Validate, inspect, or execute one resolved Composite-KRR run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_config import KRRConfig, json_schema, run, template


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="validate one JSON/TOML run")
    validate.add_argument("config", type=Path)
    validate.add_argument(
        "--no-check-paths",
        action="store_true",
        help="validate values without requiring input files to exist",
    )

    execute = subcommands.add_parser("run", help="execute one validated run")
    execute.add_argument("config", type=Path)

    schema = subcommands.add_parser("schema", help="print the JSON discovery schema")
    schema.add_argument("--compact", action="store_true")

    example = subcommands.add_parser("template", help="print a minimal JSON template")
    example.add_argument("--compact", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    if args.command == "schema":
        print(json.dumps(json_schema(), indent=None if args.compact else 2))
        return
    if args.command == "template":
        print(json.dumps(template(), indent=None if args.compact else 2))
        return

    configuration = KRRConfig.load(args.config)
    if args.command == "validate":
        configuration.validate(check_paths=not args.no_check_paths)
        print(
            f"VALID config_sha256={configuration.resolved().sha256()} "
            f"output={configuration.resolved().values['OUTPUT_DIR']}"
        )
        return
    execution = run(configuration)
    print(
        f"COMPLETE returncode={execution.returncode} "
        f"config_sha256={execution.config_sha256} "
        f"output={execution.output_directory}"
    )


if __name__ == "__main__":
    main()

