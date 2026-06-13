#!/bin/bash
# gvd_lite full pipeline:
#   phase 1: eval ALL questions on every video that already has a VKG
#   phase 2: build the missing VKGs (build_remaining.py, one resident engine)
#   phase 3: eval the newly built videos (resume-safe, same --out)
# Logs running accuracy to W&B project gvd-lite-lvbench.
cd /workspace
set -a; source /workspace/.env; set +a
export WANDB_RUN_ID=lite-full-v1

OUT=results_gvd_lite_full.jsonl
PROJ=gvd-lite-lvbench

drain() {
    pkill -f "VLLM::[E]ngineCore" 2>/dev/null
    for i in $(seq 1 30); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "$used" -lt 2000 ] && return 0
        sleep 5
    done
}

eval_pass() {
    local tries=0
    until python3 -m gvd_lite.eval_lvbench \
            --csv data/LVBench_full.csv \
            --vkg-dir /workspace/vkgs \
            --video-dir /workspace/videos \
            --out "$OUT" \
            --model /workspace/models/Qwen3.5-4B \
            --tp 1 \
            --gpu-memory-utilization 0.65 \
            --max-model-len 65536 \
            --wandb-project "$PROJ"; do
        tries=$((tries + 1))
        echo "[pipeline] eval exited nonzero (attempt $tries), draining + resuming in 20s"
        drain; sleep 20
        if [ "$tries" -ge 30 ]; then
            echo "[pipeline] too many eval crashes, giving up this pass"
            return 1
        fi
    done
}

echo "[pipeline] phase 1: eval on videos with existing VKGs"
eval_pass
drain

echo "[pipeline] phase 2: build missing VKGs"
python3 build_remaining.py || echo "[pipeline] build_remaining exited nonzero, continuing"
drain

echo "[pipeline] phase 3: eval on newly built videos"
eval_pass

echo "[pipeline] ALL DONE"
