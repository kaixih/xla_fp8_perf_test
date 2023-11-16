OUTPUT=$(mktemp -d)
PAXML_DIR=$(dirname `python -c 'import paxml; print(*paxml.__path__)'`)
cp ci_configs.py ${PAXML_DIR}
pushd ${PAXML_DIR} > /dev/null

# Use fake datasets
export VOCAB_PATH="/home/dataset/c4_en_301_5Mexp2_spm.model"

FP8_COMMON="--xla_gpu_enable_reduction_epilogue_fusion=false \
            --xla_gpu_enable_cublaslt=true \
           "
XLA_COMMON="--xla_gpu_enable_latency_hiding_scheduler=true \
            --xla_gpu_enable_async_collectives=true \
            --xla_gpu_enable_highest_priority_async_stream=true \
            --xla_gpu_all_reduce_combine_threshold_bytes=51200 \
            --xla_gpu_enable_cudnn_layer_norm=true \
            --xla_gpu_enable_cudnn_fmha=false \
           "
MODEL_NAME=$1
OPT_MODE=$2
MATH_MODE=$3
TRITON_GEMM=$4
G=$5

if [[ "$OPT_MODE" == "XLA" ]]; then
  export ENABLE_TE=0
elif [[ "$OPT_MODE" == "TE" ]]; then
  export ENABLE_TE=1
else
  echo TRAINING SCRIPT FAILED: Unsupported OPT_MODE: $OPT_MODE
  exit 1
fi
if [[ "$MATH_MODE" == "fp8" ]]; then
  XLA_COMMON+=$FP8_COMMON
  USE_FP8=true
  export ENABLE_FP8=1
elif [[ "$MATH_MODE" == "bf16" ]]; then
  USE_FP8=false
  export ENABLE_FP8=0
else
  echo TRAINING SCRIPT FAILED: Unsupported MATH_MODE: $MATH_MODE
  exit 1
fi
if [[ "$TRITON_GEMM" == "no" ]]; then
  USE_TRITON_GEMM=false
elif [[ "$TRITON_GEMM" == "yes" ]]; then
  USE_TRITON_GEMM=true
else
  echo TRAINING SCRIPT FAILED: Unsupported TRITON_GEMM: $TRITON_GEMM
  exit 1
fi

export XLA_FLAGS="$XLA_COMMON --xla_gpu_enable_triton_gemm=$USE_TRITON_GEMM"
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

printf "%-30s %7s %4s %11s %5s %9s %8s\n" $MODEL_NAME $OPT_MODE $MATH_MODE $TRITON_GEMM $G $PERF $WALLTIME
rm -rf "$TMP_DIR"

#echo -n 'ref(triton) '
#export XLA_FLAGS="$XLA_COMM"
#python -m paxml.main \
#    --fdl_config=ci_configs.GPT5BSynthetic \
#    --fdl.PACKED_INPUT=False \
#    --fdl.USE_FP8=False \
#    --job_log_dir=${OUTPUT} \
#    --enable_checkpoint_saving=False \
#    --alsologtostderr |& grep 'Setting task status: step = 100,.*steps/sec'
#
#echo -n 'ref(no_triton) '
#export XLA_FLAGS="$XLA_COMM --xla_gpu_enable_triton_gemm=false"
#python -m paxml.main \
#    --fdl_config=ci_configs.GPT5BSynthetic \
#    --fdl.PACKED_INPUT=False \
#    --fdl.USE_FP8=False \
#    --job_log_dir=${OUTPUT} \
#    --enable_checkpoint_saving=False \
#    --alsologtostderr |& grep 'Setting task status: step = 100,.*steps/sec'
#
#
#echo -n 'fp8(triton) '
#export XLA_FLAGS="$XLA_COMM $FP8_COMM"
#python -m paxml.main \
#    --fdl_config=ci_configs.GPT5BSynthetic \
#    --fdl.PACKED_INPUT=False \
#    --fdl.USE_FP8=True \
#    --job_log_dir=${OUTPUT} \
#    --enable_checkpoint_saving=False \
#    --alsologtostderr |& grep 'Setting task status: step = 100,.*steps/sec'


#export ENABLE_TE=1
#export ENABLE_FP8=1
#echo -n 'te-fp8 '
#export XLA_FLAGS="$XLA_COMM --xla_gpu_enable_triton_gemm=false"
#python -m paxml.main \
#    --fdl_config=ci_configs.GPT5BSynthetic \
#    --fdl.PACKED_INPUT=False \
#    --job_log_dir=${OUTPUT} \
#    --enable_checkpoint_saving=False \
#    --alsologtostderr


 

