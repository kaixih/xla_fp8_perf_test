OUTPUT=$(mktemp -d)
PAXML_DIR=$(dirname `python -c 'import paxml; print(*paxml.__path__)'`)
cp ci_configs.py ${PAXML_DIR}
pushd ${PAXML_DIR} > /dev/null

MODEL_NAME=$1
OPT_MODE=$2
MATH_MODE=$3
XLA_EXTRAS=$4
GPUS=$5

if [[ "$OPT_MODE" == "XLA" ]]; then
  export ENABLE_TE=0
elif [[ "$OPT_MODE" == "TE" ]]; then
  export ENABLE_TE=1
else
  echo TRAINING SCRIPT FAILED: Unsupported OPT_MODE: $OPT_MODE
  exit 1
fi
if [[ "$MATH_MODE" == "fp8" ]]; then
  USE_FP8=true
  export ENABLE_FP8=1
elif [[ "$MATH_MODE" == "bf16" ]]; then
  USE_FP8=false
  export ENABLE_FP8=0
else
  echo TRAINING SCRIPT FAILED: Unsupported MATH_MODE: $MATH_MODE
  exit 1
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

# Use fake datasets
export VOCAB_PATH="/home/dataset/c4_en_301_5Mexp2_spm.model"

# TODO(kaixih): This is a known hang issue for latency-hiding for TE-FP8.
# Verify the fix and remove this when 11-21.
X=true
if [[ "$OPT_MODE" == "TE" && "$MATH_MODE" == "fp8" ]]; then
  X=false
fi

FP8_COMMON="--xla_gpu_enable_reduction_epilogue_fusion=false \
           "
XLA_COMMON="--xla_gpu_enable_latency_hiding_scheduler=$X \
            --xla_gpu_enable_async_collectives=true \
            --xla_gpu_enable_highest_priority_async_stream=true \
            --xla_gpu_all_reduce_combine_threshold_bytes=51200 \
            --xla_gpu_enable_cudnn_layer_norm=$USE_CUDNN_LN \
            --xla_gpu_enable_cudnn_fmha=$USE_CUDNN_FMHA \
            --xla_gpu_enable_cublaslt=$USE_CUBLASLT \
            --xla_gpu_enable_triton_gemm=$USE_TRITON_GEMM \
            --xla_gpu_simplify_all_fp_conversions=true \
           "
# TODO(kaixih): This is a known issue that shows extremely long compilation time
# for XLA-FP8. So, disabling the reduction epilog fusion for now.
if [[ "$OPT_MODE" == "XLA" && "$MATH_MODE" == "fp8" ]]; then
  XLA_COMMON+=$FP8_COMMON
fi

export XLA_FLAGS="$XLA_COMMON"
SECONDS=0
TMPFILE="$TMPDIR/$(mktemp tmp.XXXXXX)"
python -m paxml.main \
    --fdl_config=ci_configs.$MODEL_NAME \
    --fdl.PACKED_INPUT=False \
    --fdl.USE_FP8=$USE_FP8 \
    --job_log_dir=${OUTPUT} \
    --enable_checkpoint_saving=False \
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

printf "%-18s %8s %4s %35s %4d %9.3f %8d\n" $MODEL_NAME $OPT_MODE $MATH_MODE $XLA_EXTRAS $GPUS $PERF $WALLTIME
rm -rf "$TMP_DIR"


