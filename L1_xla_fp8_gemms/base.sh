OUTPUT=$(mktemp -d)
PAXML_DIR=$(dirname `python -c 'import paxml; print(*paxml.__path__)'`)
cp ci_configs.py ${PAXML_DIR}
pushd ${PAXML_DIR} > /dev/null

export ENABLE_TE=0
USE_FP8=true
XLA_DUMP_DIR=$TMPDIR/xla_dump

# Use fake datasets
export VOCAB_PATH="/home/dataset/c4_en_301_5Mexp2_spm.model"

FP8_COMMON="--xla_gpu_enable_reduction_epilogue_fusion=false \
           "
XLA_COMMON="--xla_gpu_enable_latency_hiding_scheduler=true \
            --xla_gpu_enable_async_collectives=true \
            --xla_gpu_enable_highest_priority_async_stream=true \
            --xla_gpu_all_reduce_combine_threshold_bytes=51200 \
            --xla_gpu_enable_cudnn_layer_norm=true \
            --xla_gpu_enable_cudnn_fmha=false \
            --xla_gpu_enable_cublaslt=true \
            --xla_gpu_enable_triton_gemm=false \
            --xla_dump_hlo_as_text --xla_dump_to=$XLA_DUMP_DIR \
           "
XLA_COMMON+=$FP8_COMMON
export XLA_FLAGS="$XLA_COMMON"
TMPFILE="$TMPDIR/$(mktemp tmp.XXXXXX)"

NUM_LAYERS=$1
USE_REPEATED_LAYER=$2
FWD_FP8_GEMMS=$3
BWD_FP8_GEMMS=$4

python -m paxml.main \
    --fdl_config=ci_configs.GPT5BSynthetic \
    --fdl.PACKED_INPUT=False \
    --fdl.USE_FP8=$USE_FP8 \
    --fdl.NUM_LAYERS=$NUM_LAYERS \
    --fdl.USE_REPEATED_LAYER=$USE_REPEATED_LAYER \
    --job_log_dir=${OUTPUT} \
    --enable_checkpoint_saving=False \
    --alsologtostderr >> "$TMPFILE" 2>&1

FAILURE=$?
if [[ $FAILURE -ne 0 ]]; then
  cat "$TMPFILE"
  echo TRAINING SCRIPT FAILED
  rm -f "$TMPFILE"
  rm -rf "$XLA_DUMP_DIR"
  exit 1
fi

cat > match_fp8_fprop.ll <<EOF
; CHECK-COUNT-$FWD_FP8_GEMMS: custom_call_target="__cublas\$lt\$matmul\$f8"{{.*}}"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]}
; CHECK-NOT: custom_call_target="__cublas\$lt\$matmul\$f8"{{.*}}"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]}
EOF

cat > match_fp8_bprop.ll <<EOF
; CHECK-COUNT-$BWD_FP8_GEMMS: custom_call_target="__cublas\$lt\$matmul\$f8"{{.*}}"precision_config":{"operand_precision":["HIGHEST","HIGHEST"]}
; CHECK-NOT: custom_call_target="__cublas\$lt\$matmul\$f8"{{.*}}"precision_config":{"operand_precision":["HIGHEST","HIGHEST"]}
EOF

TARGET_HLO_FILE=$XLA_DUMP_DIR/module_0025.pjit__wrapped_step_fn.sm_9.0_gpu_after_optimizations.txt
FileCheck-14 --input-file $TARGET_HLO_FILE match_fp8_fprop.ll &> /dev/null
FWD_FAILURE=$?
FileCheck-14 --input-file $TARGET_HLO_FILE match_fp8_bprop.ll &> /dev/null
BWD_FAILURE=$?

if [[ $FWD_FAILURE -eq 0 ]]; then
  echo FWD CHECKING ... Pass
else
  echo FWD CHECKING ... FAIL: Got "<" $FWD_FP8_GEMMS FP8 GEMMS
fi

if [[ $BWD_FAILURE -eq 0 ]]; then
  echo BWD CHECKING ... Pass
else
  echo BWD CHECKING ... FAIL: Got "<" $BWD_FP8_GEMMS FP8 GEMMS
fi

if [[ $FWD_FAILURE -ne 0 || $BWD_FAILURE -ne 0 ]]; then
  exit 1
fi


