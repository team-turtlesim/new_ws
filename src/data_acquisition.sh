#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../bagfile}"
MIN_FREE_GIB="${MIN_FREE_GIB:-5}"
CHUNK_SIZE_BYTES=$((1024 * 1024 * 1024))
BAR_WIDTH="${BAR_WIDTH:-20}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-2}"

mkdir -p "$OUTPUT_DIR"

RECORDER_PID=""

human_bytes() {
    if command -v numfmt >/dev/null 2>&1; then
        numfmt --to=iec-i --suffix=B "$1"
    else
        echo "${1}B"
    fi
}

get_storage_stats() {
    df -B1 --output=size,avail "$OUTPUT_DIR" | tail -n 1 | awk '{print $1, $2}'
}

render_usage_bar() {
    local total_bytes="$1"
    local available_bytes="$2"
    local used_bytes used_percent filled empty
    local bar_filled="" bar_empty=""

    used_bytes=$((total_bytes - available_bytes))
    used_percent=$((used_bytes * 100 / total_bytes))
    filled=$((used_percent * BAR_WIDTH / 100))
    empty=$((BAR_WIDTH - filled))

    printf -v bar_filled '%*s' "$filled" ''
    printf -v bar_empty '%*s' "$empty" ''
    bar_filled="${bar_filled// /■}"
    bar_empty="${bar_empty// /-}"

    printf '[%s%s] : %3d%% used\n' "$bar_filled" "$bar_empty" "$used_percent"
}

print_remaining_storage() {
    local total_bytes="$1"
    local available_bytes="$2"

    echo "Remaining storage: $(human_bytes "$available_bytes") / $(human_bytes "$total_bytes") free"
    render_usage_bar "$total_bytes" "$available_bytes"
}

stop_recorder() {
    if [[ -n "$RECORDER_PID" ]] && kill -0 "$RECORDER_PID" >/dev/null 2>&1; then
        kill -INT "$RECORDER_PID" >/dev/null 2>&1 || true
        wait "$RECORDER_PID" || true
    fi
}

trap 'stop_recorder; exit 130' INT TERM

read -r TOTAL_BYTES AVAILABLE_BYTES < <(get_storage_stats)
THRESHOLD_BYTES=$((MIN_FREE_GIB * 1024 * 1024 * 1024))

print_remaining_storage "$TOTAL_BYTES" "$AVAILABLE_BYTES"

if (( AVAILABLE_BYTES <= THRESHOLD_BYTES )); then
    echo "full"
    exit 0
fi

echo "Starting ros2 bag record with 1GiB split size in ${OUTPUT_DIR}"
BAG_PREFIX="$OUTPUT_DIR/bag_$(date +%Y%m%d_%H%M%S)"

ros2 bag record -a -o "$BAG_PREFIX" -b "$CHUNK_SIZE_BYTES" &
RECORDER_PID=$!

while kill -0 "$RECORDER_PID" >/dev/null 2>&1; do
    read -r TOTAL_BYTES AVAILABLE_BYTES < <(get_storage_stats)
    print_remaining_storage "$TOTAL_BYTES" "$AVAILABLE_BYTES"

    if (( AVAILABLE_BYTES <= THRESHOLD_BYTES )); then
        echo "Storage threshold reached. Stopping ros2 bag record."
        kill -INT "$RECORDER_PID" >/dev/null 2>&1 || true
        wait "$RECORDER_PID" || true
        RECORDER_PID=""
        break
    fi

    sleep "$CHECK_INTERVAL_SEC"
done

if [[ -n "$RECORDER_PID" ]]; then
    wait "$RECORDER_PID"
    RECORD_STATUS=$?
    RECORDER_PID=""

    if (( RECORD_STATUS != 0 )) && (( RECORD_STATUS != 130 )); then
        echo "ros2 bag record exited unexpectedly with status ${RECORD_STATUS}"
    fi
fi

read -r TOTAL_BYTES AVAILABLE_BYTES < <(get_storage_stats)
echo "Recording loop finished."
print_remaining_storage "$TOTAL_BYTES" "$AVAILABLE_BYTES"
