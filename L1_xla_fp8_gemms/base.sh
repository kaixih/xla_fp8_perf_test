OUTPUT=$(mktemp -d)
PAXML_DIR=$(dirname `python -c 'import paxml; print(*paxml.__path__)'`)
cp ci_configs.py ${PAXML_DIR}
cp gen_match_rule.sh ${PAXML_DIR}
pushd ${PAXML_DIR} > /dev/null

export ENABLE_TE=0
USE_FP8=true
XLA_DUMP_DIR=$TMPDIR/xla_dump

GEN_XLA_DUMP=$1
NUM_LAYERS=$2
USE_REPEATED_LAYER=$3
PROP_DIRECTION=$4
EXPECTED_FP8_GEMMS=$5
TARGET_LAYER=$6

if [ -n "$TARGET_LAYER" ]; then
  case "$TARGET_LAYER" in
    "QKV")
      EXPECTED_OP_NAME=self_attention/combined_qkv/einsum
      ;;
    "POST")
      EXPECTED_OP_NAME=self_attention/post/einsum
      ;;
    "FF1")
      EXPECTED_OP_NAME=ffn_layer1/linear/einsum
      ;;
    "FF2")
      EXPECTED_OP_NAME=ffn_layer2/linear/einsum
      ;;
    *)
      echo FAILED: Unsupported TARGET_LAYER: $TARGET_LAYER
      exit 1
      ;;
  esac
fi


# Use fake datasets
export VOCAB_PATH="/home/dataset/c4_en_301_5Mexp2_spm.model"

FP8_COMMON="--xla_gpu_enable_reduction_epilogue_fusion=false \
           "
XLA_COMMON="--xla_gpu_enable_latency_hiding_scheduler=true \
            --xla_gpu_enable_async_collectives=true \
            --xla_gpu_enable_highest_priority_async_stream=true \
            --xla_gpu_all_reduce_combine_threshold_bytes=51200 \
            --xla_gpu_enable_cudnn_layer_norm=true \
            --xla_gpu_enable_cudnn_fmha=true \
            --xla_gpu_fused_attention_use_cudnn_rng=true \
            --xla_gpu_enable_cublaslt=true \
            --xla_gpu_enable_triton_gemm=false \
            --xla_dump_hlo_as_text --xla_dump_to=$XLA_DUMP_DIR \
           "
XLA_COMMON+=$FP8_COMMON
export XLA_FLAGS="$XLA_COMMON"
TMPFILE="$TMPDIR/$(mktemp tmp.XXXXXX)"

if [[ "$GEN_XLA_DUMP" == "y" ]]; then
  rm -rf $XLA_DUMP_DIR
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
  rm -f "$TMPFILE"
fi

TARGET_HLO_FILE=$XLA_DUMP_DIR/module_0025.pjit__wrapped_step_fn.sm_9.0_gpu_after_optimizations.txt
TMPFC="$TMPDIR/$(mktemp tmp.XXXXXX)"

function fetch_matches() {
  awk '{
      is_err = 0
      is_cc = 0
      m = "NA"
      for(i = 1; i <= NF; i++) {
        if (match($i, /error:/)) {
          is_err = 1
        }
        if (match($i, /CHECK-COUNT:/)) {
          is_cc = 1
        }
        if (match($i, /\([0-9]+/)) {
          m = substr($i, 2)
        }
      }
      if (is_err && is_cc) {
        print m
      }
  }'
}

bash gen_match_rule.sh $EXPECTED_FP8_GEMMS $PROP_DIRECTION $EXPECTED_OP_NAME
FileCheck-14 --input-file $TARGET_HLO_FILE match_rules_gen.ll &> $TMPFC
FAILURE=$?

if [[ $FAILURE -eq 0 ]]; then
  echo $PROP_DIRECTION $TARGET_LAYER CHECKING ... Pass
else
  GOT=0
  if [[ "$EXPECTED_FP8_GEMMS" > 1 ]]; then
    FETCH=$(cat $TMPFC | fetch_matches)
    GOT=$((FETCH-1))
  fi
  echo $PROP_DIRECTION $TARGET_LAYER CHECKING ... FAIL: EXPECTED $EXPECTED_FP8_GEMMS FP8 GEMMS, BUT GOT $GOT
fi

rm -f "$TMPFC"

exit $FAILURE
