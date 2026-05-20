#!/bin/bash
set -ex
cd "$(dirname "$0")/.."

##from svatava
## sshg -L 8089:localhost:8089 -L 8090:localhost:8090 -L 8091:localhost:8091 supergpu4 -N &
## sshg -L 8082:localhost:8082  supergpu9 -N &


## kill -9 $(lsof -t -i:8090)

##from local
# ssh -L 3000:localhost:3000 \
#     -L 8000:localhost:8000 \
#     -L 8089:localhost:8089 \
#     -L 8090:localhost:8090 \
#     -L 8091:localhost:8091 \
#     svatava

# port=8010
# port=8012
# port=8014
# port=8016
port=8020

# uv run uvicorn unmute.main_websocket_ep:app --host 127.0.0.1 --port $port --ws-per-message-deflate=false

uv run uvicorn unmute.main_websocket:app --host 127.0.0.1 --port $port --ws-per-message-deflate=false

# uv run uvicorn unmute.main_websocket_forecast:app --host 127.0.0.1 --port $port --ws-per-message-deflate=false

