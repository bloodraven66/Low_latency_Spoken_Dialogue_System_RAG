#!/bin/bash
set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

port=8020

uv run uvicorn unmute.main_websocket_rag:app --host 127.0.0.1 --port $port --ws-per-message-deflate=false
