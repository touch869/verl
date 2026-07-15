#!/bin/bash
# Overnight ABCD matrix on Ascend (defaults = 910C 16-card; also runs on 910B3 8-card).
# Runs A(sticky/no-mc) -> B(kvcare/no-mc) -> C(sticky/mc) -> D(kvcare/mc).
# Each group restarts the container (clean ray/vllm) and archives logs.
# drop-safe: one group failing does not stop the next.
#
# Per-group collectors (so A/C and B/D emit the SAME telemetry for apples-to-apples):
#   A/C (no router): verl sticky routing + standalone_collector.py (observes /metrics +
#     kv-events, ZERO routing) + --kv-events publisher. Same vllm-evidence + kv-events
#     tally output as B/D.
#   B/D (router): KVCAwareBalancer (its own collector emits the same evidence/tally).
#
# C/D (mc): mooncake single-node topology -- metadata_server="P2PHANDSHAKE" (transfer-engine
#   peers handshake directly; NO http_metadata_server) + one mooncake_master daemon, from
#   `pip install mooncake-transfer-engine` (aarch64 wheel is ascend-enabled; no source build).
#   C/D needs 910C (A3 die, ASCEND_ENABLE_USE_FABRIC_MEM fabric-mem path); 910B3 RDMA cannot
#   register NPU device memory, so on 910B3 run only the no-mc mainline via RUN_GROUPS="A B".
#
# Everything is env-overridable. 910C uses defaults; 910B3 example (mainline A/B, skip C/D):
#   WORKDIR=/home/zzq/hgq MODEL=/home/zzq/hgq/models/Qwen3-8B \
#   REPLICAS=4 TP=2 N_GPUS=8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_GROUPS="A B" \
#   [SMOKE=1] bash examples/llm_router/run_matrix_overnight.sh
#
# SMOKE=1: tiny samples + short turns + short cap -> validates the flow in ~minutes/group.
#   On 910C runs all 4 groups (incl. mooncake C/D + collector A/C); on 910B3 use RUN_GROUPS="A B".
set -uo pipefail

# ---- machine-specific paths (env-overridable; defaults = 910C) ----
WORKDIR="${WORKDIR:-/mnt/data/h00500767}"               # 910B3: /home/zzq/hgq
REPO="${REPO:-$WORKDIR/uni-agent}"
MODEL="${MODEL:-$WORKDIR/data/models/Qwen/Qwen3-8B}"   # 910B3: $WORKDIR/models/Qwen3-8B
DATA="${DATA:-$REPO/examples/llm_router/swe_bench_verified_modal.parquet}"
CONTAINER="${CONTAINER:-hgq-verl-v080}"
MC_MASTER_PORT="${MC_MASTER_PORT:-9422}"                # mooncake master RPC port (C/D); must match mooncake_config.json master_server_address
MOONCAKE_CFG="${MOONCAKE_CFG:-$REPO/mooncake_config.json}"
# ASCEND_ENABLE_USE_FABRIC_MEM for C/D mooncake: 910C (A3 die) needs 1 (NPU unified-memory
# direct xfer, bypasses RDMA). 910B3 (non-A3) CANNOT do mooncake C/D either way — the RDMA
# path fails to register NPU device memory (EINVAL), and fabric mem is A3-only hardware.
# So validate C/D on 910C; on 910B3 use RUN_GROUPS="A B" to skip C/D.
MC_FABRIC_MEM="${MC_FABRIC_MEM:-1}"

# ---- experiment knobs (env-overridable; defaults = 910C) ----
REPLICAS="${REPLICAS:-16}"          # --num-workers (910B3: 4)
TP="${TP:-1}"                       # --tensor-parallel-size (910B3: 2)
N_GPUS="${N_GPUS:-16}"              # --n-gpus-per-node (910B3: 8)
MAX_SAMPLES="${MAX_SAMPLES:-200}"   # unique prompts; total traj = MAX_SAMPLES x N
N="${N:-8}"                         # trajectories per prompt
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_TURNS="${MAX_TURNS:-100}"
MML="${MML:-40960}"
PROMPT_LEN="${PROMPT_LEN:-4096}"
RESP_LEN="${RESP_LEN:-8192}"
CAP_MIN="${CAP_MIN:-300}"           # per-group wall-clock cap (minutes)

# ---- SMOKE mode: tiny samples for fast end-to-end flow validation ----
if [ "${SMOKE:-0}" = "1" ]; then
  MAX_SAMPLES=2; N=2; MAX_TURNS=3; CAP_MIN=20
  echo "=== SMOKE MODE: max-samples=$MAX_SAMPLES n=$N max-turns=$MAX_TURNS cap=${CAP_MIN}min/group ==="
fi

# ---- derived ----
SIM=$REPO/examples/kvc_aware_router/agent_config_simulated.yaml
ROUTER="--router-strategy kvcaware"
SCOLLECTOR=$REPO/examples/kvc_aware_router/standalone_collector.py
OUTDIR=$WORKDIR/results/matrix_$(date +%Y%m%d_%H%M)${SMOKE:+_smoke}
LOGS=$WORKDIR/logs
mkdir -p "$OUTDIR" "$LOGS"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export HF_HOME="${HF_HOME:-$WORKDIR/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$WORKDIR/.hf_cache/datasets}"
export MOONCAKE_CPU_STAGING=1
export MC_TCP_ENABLE_CONNECTION_POOL=1

# --- preflight: skip if NPU cards are occupied (shared machine: do not collide) ---
_check_cards_free() {
  local MAX_USED_MB
  MAX_USED_MB=$(npu-smi info 2>/dev/null | grep -oE "[0-9]+ ?/ ?65536" | sed 's#/.*##' | sort -rn | head -1)
  if [ -z "$MAX_USED_MB" ] || [ "$MAX_USED_MB" -gt 10000 ]; then
    echo "$(date +%H:%M:%S) SKIP: NPU cards occupied (max HBM=${MAX_USED_MB:-?}MB > baseline). Aborting."
    exit 0
  fi
  echo "$(date +%H:%M:%S) preflight OK: cards free (max HBM=${MAX_USED_MB}MB)"
}
_check_cards_free

run_group() {
  local G=$1 ROUTER_FLAG=$2 MC_FLAG=$3 EXTRA=$4
  rm -f "$LOGS/${G}.log" "$LOGS/${G}_scrape.log"
  echo "$(date +%H:%M:%S) === GROUP $G start (router=${ROUTER_FLAG:-none} mc=$MC_FLAG) ==="
  docker restart "$CONTAINER"; sleep 5

  # mooncake for C/D: single-node ascend-direct (fabric-mem). metadata_server="P2PHANDSHAKE"
  # in MOONCAKE_CFG => peers handshake directly, NO http_metadata_server; the store still needs
  # a master, so start mooncake_master only. NOTE: source-build master uses --port (not pip's
  # --rpc_port). Flags/lease-TTL follow the official vLLM-Ascend KV Pool guide (lease TTL must
  # exceed ASCEND_TRANSFER_TIMEOUT). See findings.md §12 + SETUP_GUIDE section 7.
  if [ "$MC_FLAG" = "on" ]; then
    docker exec -d "$CONTAINER" bash -lc "mooncake_master --port $MC_MASTER_PORT --default_kv_lease_ttl 11000 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 > $LOGS/mc_master_$G.log 2>&1"
    sleep 5
  fi

  # --- gpu_mem auto-adapt: TP1 (1 replica/card) targets 0.8; TP2 (2 replicas/card) 0.55 ---
  local MIN_USED_MB FREE_GB GMU
  MIN_USED_MB=$(docker exec "$CONTAINER" bash -lc "npu-smi info 2>/dev/null | grep -oE '[0-9]+ ?/ ?65536' | sed 's#/.*##' | sort -rn | head -1" 2>/dev/null)
  if [ -n "$MIN_USED_MB" ]; then
    FREE_GB=$(( (65536 - MIN_USED_MB) / 1024 ))
    if [ "$TP" = "1" ]; then
      if   [ "$FREE_GB" -lt 45 ]; then GMU=0.6
      elif [ "$FREE_GB" -lt 55 ]; then GMU=0.7
      else GMU=0.8
      fi
    else  # TP2
      if   [ "$FREE_GB" -lt 40 ]; then GMU=0.35
      elif [ "$FREE_GB" -lt 50 ]; then GMU=0.45
      else GMU=0.55
      fi
    fi
  else
    GMU=0.8
  fi
  echo "$(date +%H:%M:%S) GROUP $G: TP=$TP min_free~${FREE_GB:-?}GB -> gpu_mem_util=$GMU"
  docker exec "$CONTAINER" bash -lc "sed -i \"s/gpu_memory_utilization = 0\.[0-9]*/gpu_memory_utilization = $GMU/\" $REPO/examples/kvc_aware_router/parallel_infer.py"

  # A/C (no router): enable the kv-events publisher so the standalone collector can subscribe.
  local KV_FLAG=""; [ -z "$ROUTER_FLAG" ] && KV_FLAG="--kv-events"

  # C/D (mc, fabric-mem): env vars per the official vLLM-Ascend KV Pool guide.
  # ACL_OP_INIT_MODE/HCCL_RDMA_TIMEOUT/ASCEND_CONNECT|TRANSFER_TIMEOUT are required for the
  # ascend-direct fabric-mem path; lease TTL (above) must exceed ASCEND_TRANSFER_TIMEOUT.
  local MC_ENV=""; [ "$MC_FLAG" = "on" ] && MC_ENV="-e ASCEND_ENABLE_USE_FABRIC_MEM=$MC_FABRIC_MEM -e ACL_OP_INIT_MODE=1 -e HCCL_RDMA_TIMEOUT=17 -e ASCEND_CONNECT_TIMEOUT=10000 -e ASCEND_TRANSFER_TIMEOUT=10000"

  docker exec -d -e HF_HOME="$HF_HOME" -e HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    -e ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" -e PYTHONHASHSEED=0 $MC_ENV \
    "$CONTAINER" bash -lc "cd $REPO && setsid bash examples/llm_router/run_infer.sh \
    $MODEL $DATA $SIM \
    $ROUTER_FLAG $KV_FLAG $EXTRA \
    --num-workers $REPLICAS --n-gpus-per-node $N_GPUS --tensor-parallel-size $TP \
    --max-num-seqs $MAX_NUM_SEQS --max-samples $MAX_SAMPLES --n $N --max-turns $MAX_TURNS \
    --prompt-length $PROMPT_LEN --response-length $RESP_LEN --max-model-len $MML \
    > $LOGS/${G}.log 2>&1"

  # A/C (no router): standalone collector observes /metrics + kv-events (zero routing).
  # Same vllm-evidence + kv-events tally output as B/D. Launched after vllm is up.
  if [ -z "$ROUTER_FLAG" ] && [ -f "$SCOLLECTOR" ]; then
    sleep 25
    docker exec -d -e HF_HOME="$HF_HOME" -e ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
      "$CONTAINER" bash -lc "cd $REPO && PYTHONPATH=$REPO python $SCOLLECTOR --interval 1 --num-replicas $REPLICAS > $LOGS/${G}_scrape.log 2>&1"
    echo "$(date +%H:%M:%S) GROUP $G: standalone collector launched (A/C observer -> ${G}_scrape.log)"
  fi

  # wait for completion (poll Mean RM or crash), cap CAP_MIN min/group
  local i max=$((CAP_MIN * 60 / 10))
  for ((i=1; i<=max; i++)); do
    if grep -qE "Mean RM Score|EngineDeadError|SYSTEM_ERROR|ValueError|RuntimeError|WorkerProc failed|OSError|Read-only|not found|No such file" "$LOGS/${G}.log" 2>/dev/null; then break; fi
    sleep 10
  done
  grep -E "Mean RM Score|EngineDeadError" "$LOGS/${G}.log" | tail -2
  # fold scrape log (A/C telemetry) into the main group log so it archives together
  [ -f "$LOGS/${G}_scrape.log" ] && docker exec "$CONTAINER" bash -c "cat $LOGS/${G}_scrape.log >> $LOGS/${G}.log" 2>/dev/null
  cp "$LOGS/${G}.log" "$OUTDIR/${G}.log"
  echo "$(date +%H:%M:%S) === GROUP $G done ==="
}

# A: sticky no-mc | B: kvcare no-mc | C: sticky mc | D: kvcare mc
# RUN_GROUPS selects which to run (default "A B C D"). e.g. RUN_GROUPS="A B" runs the no-mc
# mainline only (skips mooncake C/D); RUN_GROUPS="A" isolates the standalone-collector path.
RUN_GROUPS="${RUN_GROUPS:-A B C D}"
for G in $RUN_GROUPS; do
  case "$G" in
    A) run_group A "" "" "" ;;
    B) run_group B "$ROUTER" "" "" ;;
    C) run_group C "" "on" "--enable-mooncake --mooncake-config-path $MOONCAKE_CFG" ;;
    D) run_group D "$ROUTER" "on" "--enable-mooncake --mooncake-config-path $MOONCAKE_CFG" ;;
    *) echo "$(date +%H:%M:%S) skip unknown group: $G" ;;
  esac
done

echo "$(date +%H:%M:%S) === MATRIX DONE - results in $OUTDIR ==="
# release cards
docker restart "$CONTAINER"
