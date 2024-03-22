OUTPUT=$(mktemp -d)
PAXML_DIR=$(dirname `python -c 'import paxml; print(*paxml.__path__)'`)
pushd ${PAXML_DIR} > /dev/null

MODEL_NAME=$1
BACKEND=$2
MATH_MODE=$3
SDPA=$4
XLA_EXTRAS=$5
GPUS=$6

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
USE_CUBLASLT="false"
USE_CUDNN_LN="false"
USE_CUDNN_FMHA="false"
USE_TRITON_GEMM="false"
for F in $(echo $XLA_EXTRAS | sed 's/,/ /g'); do
  case "$F" in
    cublaslt) USE_CUBLASLT="true" ;;
    cudnn_ln) USE_CUDNN_LN="true" ;;
    cudnn_fmha) USE_CUDNN_FMHA="true" ;;
    triton_gemm) USE_TRITON_GEMM="true" ;;
    none) ;;
    *) echo Invalid XLA_EXTRAS $F; exit 1 ;;
  esac
done

export VOCAB_PATH=/datasets/google_c4_spm/c4_en_301_5Mexp2_spm.model

# TODO(kaixih): This is a known hang issue for latency-hiding for TE-FP8.
# Verify the fix and remove this when 11-21.
X=true
if [[ "$BACKEND" == "TE" && "$MATH_MODE" == "fp8" ]]; then
  X=false
fi

XLA_DUMP_DIR=/xla_dump
XLA_COMMON="--xla_gpu_enable_latency_hiding_scheduler=$X \
            --xla_gpu_enable_async_collectives=true \
            --xla_gpu_enable_highest_priority_async_stream=true \
            --xla_gpu_all_reduce_combine_threshold_bytes=51200 \
            --xla_gpu_enable_cudnn_layer_norm=$USE_CUDNN_LN \
            --xla_gpu_enable_cudnn_fmha=$USE_CUDNN_FMHA \
            --xla_gpu_fused_attention_use_cudnn_rng=true \
            --xla_gpu_enable_cublaslt=$USE_CUBLASLT \
            --xla_gpu_enable_triton_gemm=$USE_TRITON_GEMM \
            --xla_gpu_simplify_all_fp_conversions=true \
           "

DEBUG=1
if [[ "$DEBUG" == "1" ]]; then
  XLA_COMMON+="--xla_dump_hlo_pass_re=.* \
               --xla_dump_hlo_as_text --xla_dump_to=$XLA_DUMP_DIR \
              "
  echo XLA DUMP TO PATH $XLA_DUMP_DIR
fi

if [[ "$BACKEND" == "XLA" && "$MATH_MODE" == "fp8" ]]; then
  CKPT_OPTION='--fdl.CHECKPOINT_POLICY="save_nothing"'
fi

export XLA_FLAGS="$XLA_COMMON"
SECONDS=0
TMPFILE="$TMPDIR/$(mktemp tmp.XXXXXX)"
python -m paxml.main \
    --fdl_config=paxml.contrib.gpu.scripts_gpu.configs.Pile5B \
    --fdl.USE_FP8=$USE_FP8 \
    --fdl.USE_CUDNN_FLASH_ATTENTION=$USE_FLASH_ATTENTION \
    '--fdl.ICI_MESH_SHAPE=[1,8,1]' \
    '--fdl.DCN_MESH_SHAPE=[1,1,1]' \
    $CKPT_OPTION \
    --job_log_dir=${OUTPUT} \
    --tfds_data_dir=/datasets/the-pile-tfds_fraction/ \
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
if [[ $FAILURE -eq 0 ]]; then
  mapfile -t LOSS_CURVE < <(cat "$TMPFILE" | grep 'training' | grep -o "\[PAX STATUS.*")
  LOSS=$(cat "$TMPFILE" | \
         grep 'step_i: 100,.*training loss:' | \
         awk '{
           for(i = 1; i <= NF; i++) {
             found = match($i, /loss:/)
             if (found) {
               loss=$((i+1))
               printf "%.3f\n", loss
             }
           }
         }')
  [[ -z "$LOSS" ]] && FAILURE=1
fi
WALLTIME=$SECONDS

if [[ $FAILURE -ne 0 ]]; then
  cat "$TMPFILE"
  echo TRAINING SCRIPT FAILED
  if [[ "$DEBUG" == "0" ]]; then
    rm -f "$TMPFILE"
  fi
  exit 1
fi
echo $TMPFILE

printf "%-18s %8s %4s %4s %20s %4d %9.3f %4.3f %8d\n" $MODEL_NAME $BACKEND $MATH_MODE $SDPA $XLA_EXTRAS $GPUS $PERF $LOSS $WALLTIME
for line in "${LOSS_CURVE[@]}"; do
  echo $line
done
if [[ "$DEBUG" == "0" ]]; then
  rm -rf "$TMPFILE"
fi


