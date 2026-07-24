#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONHASHSEED=0

MODEL=/path/to/Qwen/Qwen3-8B
DATASET=/path/to/swe_bench_train_model.parquet
DEPLOYMENT=examples/kvc_aware_router/agent_config_simulated.yaml
MAX_SAMPLES=64
CONCURRENCY=16
RES_LEN=8000
LOG_BASE=/tmp

TARGET="=> Mean RM Score"

sed -i "s|    max_turns: .*|    max_turns: 300|" $DEPLOYMENT

concurrencys=(8 16 32)
contexts=(16384 32768 64000 128000)

for CONCURRENCY in "${concurrencys[@]}"; do
    for CONTEXT in "${contexts[@]}"; do
        sed -i "s|  concurrency: .*|  concurrency: ${CONCURRENCY}|" $DEPLOYMENT

        LOG_FILE="infer-sticky-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}.log"
        sed -i "s|  log_dir: .*|  log_dir: ${LOG_BASE}/router-trajs/sticky-${CONCURRENCY}-${CONTEXT}|" $DEPLOYMENT
        while ! grep -q "$TARGET" "$LOG_FILE" 2>/dev/null; do
            pkill -9 python
            ps -aux | grep run_infer.sh | grep -v grep | awk -F ' ' '{print$2}' | xargs -I {} kill -9 {}
            ray stop
            fuser -k /dev/davinci*
            npu-smi info
            echo "Running sticky concurrency=${CONCURRENCY} context=${CONTEXT}"
            bash examples/kvc_aware_router/run_infer.sh $MODEL $DATASET $DEPLOYMENT --device ascend \
                    --n-gpus-per-node 16 --tp 1 --response-length $RES_LEN --max-model-len $CONTEXT \
                    --num-workers 16 --max-samples $MAX_SAMPLES --n 8 --shuffle \
                    --slow-cut least-inflight --overload-mode None --kv-events > $LOG_FILE 2>&1
        done

        lts=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
        for lt in "${lts[@]}"; do
            LOG_FILE="infer-kvcaware-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}.log"
            sed -i "s|  log_dir: .*|  log_dir: ${LOG_BASE}/router-trajs/kvcaware-${CONCURRENCY}-${CONTEXT}|" $DEPLOYMENT
            while ! grep -q "$TARGET" "$LOG_FILE" 2>/dev/null; do
                pkill -9 python
                ps -aux | grep run_infer.sh | grep -v grep | awk -F ' ' '{print$2}' | xargs -I {} kill -9 {}
                ray stop
                fuser -k /dev/davinci*
                npu-smi info
                echo "Running kvcaware-lt${lt} concurrency=${CONCURRENCY} context=${CONTEXT}"
                bash examples/kvc_aware_router/run_infer.sh $MODEL $DATASET $DEPLOYMENT --device ascend \
                        --n-gpus-per-node 16 --tp 1 --response-length $RES_LEN --max-model-len $CONTEXT \
                        --num-workers 16 --max-samples $MAX_SAMPLES --n 8 --shuffle \
                        --slow-cut capacity-token-aware --overload-mode kv_cache_usage_perc --load-threshold $lt --kv-events > $LOG_FILE 2>&1
            done
        done
    done
done
