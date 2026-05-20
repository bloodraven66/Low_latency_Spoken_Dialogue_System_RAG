#!/bin/bash
set -eo pipefail

root_path=/mnt/matylda4/udupa/exps/full_duplex/unmute
benchmark_dir="/mnt/matylda4/udupa/exps/RAG/FIT_RAG_Benchmark_with_audio"

m=gemma3_12b
model_name=unmute_anticipate_rag_${m}

rag_url="http://127.0.0.1:8095"
rag_top_k=1

skip_if_present=true

output_base="${root_path}/results_tmp/fit_rag_benchmark/${model_name}"
export PYTHONPATH="${root_path}"

cd "${root_path}"

# Build flat list: category|json_type|item_id|audio_path
item_list=$(python3.12 - <<'PYEOF'
import json
from pathlib import Path

benchmark_dir = "/mnt/matylda4/udupa/exps/RAG/FIT_RAG_Benchmark_with_audio"
for json_file in sorted(Path(benchmark_dir).rglob("*.json")):
    category = json_file.parent.name
    json_type = json_file.stem
    with open(json_file) as f:
        d = json.load(f)
    for item in d["items"]:
        print(f"{category}|{json_type}|{item['id']}|{item['audio_path']}")
PYEOF
)

total=$(echo "${item_list}" | wc -l)
current=0
start_time=$(date +%s)

echo "Output base: ${output_base}"
echo "Total items: ${total}"

# Prime RAG server — retry until healthy or give up after 30s
echo "Checking RAG server at ${rag_url} ..."
rag_ready=false
for i in $(seq 1 10); do
    response=$(curl -sf "${rag_url}/api/health" 2>/dev/null || true)
    if echo "${response}" | grep -q '"status":"ok"'; then
        echo "RAG server is up."
        # Warmup query to load index into memory
        curl -sf -X POST "${rag_url}/api/rag/retrieve" \
            -H "Content-Type: application/json" \
            -d '{"query": "warmup", "top_k": 1}' > /dev/null 2>&1 || true
        echo "RAG warmup done."
        rag_ready=true
        break
    fi
    echo "  RAG not ready (attempt ${i}/10), retrying in 3s..."
    sleep 3
done

if [ "${rag_ready}" = "false" ]; then
    echo "ERROR: RAG server not reachable after 30s. Aborting."
    exit 1
fi

# Warm up the endpointer (anticipator) GPU — the first CUDA kernel call takes 1-3s,
# which would cause item 001 to miss all speculation on short (<4s) utterances.
# We connect once before the item loop, send silence, and wait for predictions.
echo "Warming up endpointer GPU..."
python3.12 - <<'ENDPOINTER_WARMUP'
import asyncio
import numpy as np
import msgpack
import websockets
import os
import sys

sys.path.insert(0, os.environ.get("PYTHONPATH", "."))
from unmute.kyutai_constants import ENDPOINTER_SERVER, ENDPOINTER_PATH, HEADERS

async def warmup():
    uri = ENDPOINTER_SERVER + ENDPOINTER_PATH
    try:
        async with websockets.connect(uri, additional_headers=HEADERS, open_timeout=10) as ws:
            data = await asyncio.wait_for(ws.recv(), timeout=10.0)
            msg = msgpack.unpackb(data, raw=False)
            if msg.get("type") != "Ready":
                print(f"Warmup: unexpected message {msg.get('type')}", flush=True)
                return

            # Send ~1.5s of 960-sample silence frames
            silence = np.zeros(960, dtype=np.float32)
            payload = msgpack.packb(
                {"type": "Audio", "pcm": silence.tolist()},
                use_bin_type=True, use_single_float=True,
            )
            for _ in range(40):
                await ws.send(payload)

            # Wait for at least 5 predictions (GPU warm once we see them)
            n = 0
            try:
                while n < 5:
                    resp = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    m = msgpack.unpackb(resp, raw=False)
                    if m.get("type") == "Prediction":
                        n += 1
            except asyncio.TimeoutError:
                pass
            print(f"Endpointer GPU warmed up ({n} predictions received).", flush=True)
    except Exception as e:
        print(f"Endpointer warmup failed ({e!r}), continuing anyway.", flush=True)

asyncio.run(warmup())
ENDPOINTER_WARMUP

while IFS='|' read -r category json_type item_id audio_path; do
    current=$((current + 1))

    save_dir="${output_base}/${item_id}"
    mkdir -p "${save_dir}"

    input_save="${save_dir}/input.wav"
    output_save="${save_dir}/output.wav"

    if [ "${skip_if_present}" = "true" ] && [ -f "${output_save}" ]; then
        echo "[${current}/${total}] Skipping ${item_id} (output exists)"
        continue
    fi

    if [ ! -f "${audio_path}" ]; then
        echo "[${current}/${total}] Audio not found, skipping: ${audio_path}"
        continue
    fi

    now=$(date +%s)
    elapsed=$(( now - start_time ))
    elapsed_fmt=$(printf "%02d:%02d:%02d" $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60)))
    if [ "${current}" -gt 1 ]; then
        secs_per_item=$(( elapsed / (current - 1) ))
        remaining=$(( secs_per_item * (total - current + 1) ))
        eta_fmt=$(printf "%02d:%02d:%02d" $((remaining/3600)) $(( (remaining%3600)/60 )) $((remaining%60)))
    else
        eta_fmt="--:--:--"
    fi

    echo ""
    echo "[${current}/${total}] elapsed=${elapsed_fmt} ETA=${eta_fmt}  Processing: ${item_id}"
    echo "  Input:  ${audio_path}"
    echo "  Output: ${output_save}"

    cp "${audio_path}" "${input_save}"

    python3.12 unmute/scripts/evaluate_recording_speculativerag.py \
        "${audio_path}" "${output_save}" \
        --rag-url "${rag_url}" \
        --top-k "${rag_top_k}"

    # if [ ${current} -ge 5 ]; then
    #     echo "Processed 3 items, exiting early for testing."
    #     break
    # fi

done <<< "${item_list}"

echo ""
echo "Done. Processed ${current}/${total} items."
echo "Results in: ${output_base}"
