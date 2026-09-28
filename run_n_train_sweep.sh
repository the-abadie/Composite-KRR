#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/config.json}"
CONFIG_OUTPUT_DIR="${CONFIG_OUTPUT_DIR:-$ROOT/sweep_configs}"
RUN_AFTER_UPDATE="${RUN_AFTER_UPDATE:-1}"
KRR_PYTHON="${KRR_PYTHON:-$ROOT/.venv/bin/python}"
N_TRAINS=(100 250 500)

for n_train in "${N_TRAINS[@]}"; do
    config="$CONFIG_OUTPUT_DIR/n${n_train}.json"
    "$KRR_PYTHON" "$ROOT/krr_cli.py" resolve "$CONFIG_FILE" \
        --set "split.n_train=$n_train" \
        --set "run_name=\"n${n_train}\"" \
        --set "output.directory=\"output/n${n_train}\"" \
        --set 'output.overwrite=false' --no-check-paths --output "$config"
    if [[ "$RUN_AFTER_UPDATE" == "1" ]]; then
        "$KRR_PYTHON" "$ROOT/main.py" --config "$config"
    fi
done
