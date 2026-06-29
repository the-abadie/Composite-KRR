    #!/usr/bin/env bash
    if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
    fi

    set -euo pipefail

    CONFIG_FILE="${CONFIG_FILE:-config.py}"
    RUN_AFTER_UPDATE="${RUN_AFTER_UPDATE:-1}"

    # Edit this list for the N_TRAIN values you want to sweep.
    N_TRAINS=(
    100
    250
    500
    )

    # Environment variables for the run command. Example:
    # RUN_ENV=(CUDA_VISIBLE_DEVICES=1,2,3,7)
    RUN_ENV=()

    # Command to run after each config update. Example:
    # RUN_COMMAND=(.venv-cu124/bin/python main.py)
    RUN_COMMAND=(uv run python main.py)

    update_n_train() {
    local n_train="$1"
    local matches tmp

    matches="$(grep -Ec '^N_TRAIN[[:space:]]*=' "$CONFIG_FILE" || true)"
    if [[ "$matches" -ne 1 ]]; then
        echo "Expected exactly one N_TRAIN assignment in $CONFIG_FILE; found $matches." >&2
        exit 1
    fi

    tmp="$(mktemp)"
    awk -v n_train="$n_train" '
        /^N_TRAIN[[:space:]]*=/ {
        print "N_TRAIN = " n_train
        next
        }
        { print }
    ' "$CONFIG_FILE" > "$tmp"
    mv "$tmp" "$CONFIG_FILE"
    }

    if [[ "${#N_TRAINS[@]}" -eq 0 ]]; then
    echo "N_TRAINS is empty; add at least one value." >&2
    exit 1
    fi

    for n_train in "${N_TRAINS[@]}"; do
    if ! [[ "$n_train" =~ ^[0-9]+$ ]] || [[ "$n_train" -le 0 ]]; then
        echo "Invalid N_TRAIN value: $n_train" >&2
        exit 1
    fi

    echo "Setting N_TRAIN = $n_train in $CONFIG_FILE"
    update_n_train "$n_train"

    if [[ "$RUN_AFTER_UPDATE" == "1" && "${#RUN_COMMAND[@]}" -gt 0 ]]; then
        if [[ "${#RUN_ENV[@]}" -gt 0 ]]; then
        echo "Running: ${RUN_ENV[*]} ${RUN_COMMAND[*]}"
        env "${RUN_ENV[@]}" "${RUN_COMMAND[@]}"
        else
        echo "Running: ${RUN_COMMAND[*]}"
        "${RUN_COMMAND[@]}"
        fi
    else
        echo "Skipping run command because RUN_AFTER_UPDATE=$RUN_AFTER_UPDATE"
    fi
    done
