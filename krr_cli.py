"""Configuration tools for agents and a compatibility single-run command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from run_config import KRRConfig, json_schema, parse_assignment, run, template, write_immutable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("schema", "template"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path)
    for name in ("validate", "run", "resolve", "migrate"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        command.add_argument("--set", action="append", default=[], metavar="FIELD=JSON")
        if name != "run":
            command.add_argument("--no-check-paths", action="store_true")
        if name in {"resolve", "migrate"}:
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--portable-workspace", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in {"schema", "template"}:
            value = json_schema() if args.command == "schema" else template()
            text = json.dumps(value, indent=2, allow_nan=False) + "\n"
            if args.output:
                write_immutable(args.output, text)
            else:
                print(text, end="")
            return 0
        overrides = {}
        for assignment in args.set:
            key, value = parse_assignment(assignment)
            if key in overrides:
                raise ValueError(f"Duplicate override: {key}")
            overrides[key] = value
        configuration = KRRConfig.load(args.config).with_overrides(overrides)
        if args.command == "run":
            execution = run(configuration)
            print(f"COMPLETE config_sha256={execution.config_sha256} output={execution.output_dir}")
            return execution.returncode
        configuration.validate(check_paths=not args.no_check_paths)
        resolved = configuration.resolved()
        if args.command in {"resolve", "migrate"}:
            resolved = configuration.portable(args.portable_workspace) if args.portable_workspace else resolved
            write_immutable(args.output, resolved.canonical_json())
        print(f"VALID config_sha256={resolved.sha256()} output={resolved.spec.output.directory}")
        return 0
    except (ValueError, OSError) as error:
        parser.exit(2, f"Configuration error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
