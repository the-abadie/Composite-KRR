"""Run one saved configuration: python main.py --config path/to/run.json."""
from __future__ import annotations

import argparse
from pathlib import Path

from run_config import KRRConfig, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"),
                        help="JSON/TOML configuration (default: config.json next to main.py)")
    args = parser.parse_args(argv)
    try:
        execution = run(KRRConfig.load(args.config))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Configuration/run error: {error}\n")
    print(f"COMPLETE config_sha256={execution.config_sha256} output={execution.output_dir}")
    return execution.returncode


if __name__ == "__main__":
    raise SystemExit(main())
