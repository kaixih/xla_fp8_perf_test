OUTPUT=$(mktemp -d)
PAXML_DIR=$(dirname `python -c 'import paxml; print(*paxml.__path__)'`)
pushd ${PAXML_DIR} > /dev/null

MODEL_NAME=$1
BACKEND=$2
MATH_MODE=$3
SDPA=$4
GPUS=$5

if [[ "$BACKEND" == "XLA" ]]; then
  export ENABLE_TE=0
elif [[ "$BACKEND" == "TE" ]]; then
  export ENABLE_TE=1
else
  echo TRAINING SCRIPT FAILED: Unsupported BACKEND: $BACKEND
  exit 1
fi
if [[ "$MATH_MODE" == *"fp8"* ]]; then
  USE_FP8=true
  export ENABLE_FP8=1
else
  USE_FP8=false
  export ENABLE_FP8=0
fi
if [[ "$SDPA" == *"FA"* ]]; then
  USE_FLASH_ATTENTION=true
else
  USE_FLASH_ATTENTION=false
fi

# Use fake datasets
export VOCAB_PATH="/home/dataset/c4_en_301_5Mexp2_spm.model"

XLA_DUMP_DIR=$TMPDIR/xla_dump
XLA_COMMON="--xla_gpu_enable_triton_gemm=false \
            --xla_dump_hlo_as_text --xla_dump_to=$XLA_DUMP_DIR \
           "

if [[ "$MODEL_NAME" = "Synthetic5B" ]]; then
  MODEL_PATH="paxml.contrib.gpu.scripts_gpu.configs"
elif [[ "$MODEL_NAME" = "Llama2_7B" ]]; then
  MODEL_PATH="paxml.tasks.lm.params.nvidia"
fi

export XLA_FLAGS="$XLA_COMMON"
SECONDS=0
TMPFILE="$TMPDIR/$(mktemp tmp.XXXXXX)"
python -u -m paxml.main \
    --fdl_config=$MODEL_PATH.$MODEL_NAME \
    --fdl.USE_FP8=$USE_FP8 \
    --fdl.USE_CUDNN_FLASH_ATTENTION=$USE_FLASH_ATTENTION \
    --fdl.USE_REPEATED_LAYER=1 \
    '--fdl.ICI_MESH_SHAPE=[1,8,1]' \
    '--fdl.DCN_MESH_SHAPE=[1,1,1]' \
    '--fdl.CHECKPOINT_POLICY="save_nothing"' \
    --job_log_dir=${OUTPUT} \
    --enable_checkpoint_saving=False \
    --fdl.MAX_STEPS=100 \
    --fdl.SUMMARY_INTERVAL_STEPS=10 \
    --alsologtostderr >> "$TMPFILE" 2>&1

FAILURE=$?
if [[ $FAILURE -eq 0 ]]; then
  PERF=$(cat "$TMPFILE" | \
         grep 'Setting task status: step = 100,.*steps/sec' | \
         awk '{
           for(i = 1; i <= NF; i++) {
             found = match($i, /steps\/sec/)
             if (found) {
               throughput=$((i+1))
               throughput=substr(throughput, 1, length(throughput) - 1)
               printf "%.3f\n", throughput
             }
           }
         }')
  [[ -z "$PERF" ]] && FAILURE=1
fi
WALLTIME=$SECONDS

if [[ $FAILURE -ne 0 ]]; then
  cat "$TMPFILE"
  echo TRAINING SCRIPT FAILED
  rm -f "$TMPFILE"
  exit 1
fi

printf "%-18s %8s %4s %4s %4d %9.3f %8d\n" $MODEL_NAME $BACKEND $MATH_MODE $SDPA $GPUS $PERF $WALLTIME
rm -rf "$TMP_DIR"


