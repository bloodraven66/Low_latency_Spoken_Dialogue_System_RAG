#!/usr/bin/env bash
# Submit the speculative-RAG backend to a blade node and open an SSH tunnel.
# Usage: ./dockerless/start_backend_tunnel.sh [node]   (default: blade001)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_SCRIPT="${SCRIPT_DIR}/start_backend_speculative_rag.sh"
PORT=8020
NODE="${1:-blade001}"
JOB_NAME="bknd_rag"
LOG_PATH="${SCRIPT_DIR}/logs/${JOB_NAME}.log"
ROOT_RUN="/mnt/matylda4/udupa/common/root_run.sh"

mkdir -p "${SCRIPT_DIR}/logs"
[[ -f "$LOG_PATH" ]] && mv "$LOG_PATH" "${LOG_PATH}.bak"

# Submit directly via qsub — blade nodes don't have the 'gpu' resource defined
SUBMIT_OUT=$(qsub -N "$JOB_NAME" \
    -q long.q \
    -l "h=${NODE},ram_free=4G,matylda4=1" \
    -j yes \
    -o "$LOG_PATH" \
    "$ROOT_RUN" "$BACKEND_SCRIPT")
echo "$SUBMIT_OUT"

JOB_ID=$(echo "$SUBMIT_OUT" | grep -oP 'Your job \K[0-9]+')
if [[ -z "$JOB_ID" ]]; then
    echo "ERROR: could not parse job ID from submission output"
    exit 1
fi
echo "Job ID: $JOB_ID  |  Log: $LOG_PATH"

# Wait for running state
echo "Waiting for job $JOB_ID to start on $NODE..."
while true; do
    ROW=$(qstat -u udupa 2>/dev/null | awk -v id="$JOB_ID" '$1 == id {print}')
    if [[ -z "$ROW" ]]; then
        echo "ERROR: job $JOB_ID no longer in queue — it may have failed. Check: $LOG_PATH"
        exit 1
    fi
    STATE=$(echo "$ROW" | awk '{print $5}')
    if [[ "$STATE" == "r" ]]; then
        echo "Job is running."
        break
    fi
    echo "  state=$STATE, retrying in 5s..."
    sleep 5
done

echo "Opening tunnel: localhost:${PORT} -> ${NODE}:${PORT}  (Ctrl-C to close)"
ssh -L "${PORT}:localhost:${PORT}" "$NODE" -N
