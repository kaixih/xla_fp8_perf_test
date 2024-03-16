printf "%-18s %8s %4s %20s %4s %9s %4s %8s\n" NETWORK OPT_MODE MATH XLA_EXTRAS GPUs STEPS/SEC LOSS WALLSECS

if [ "$1" = "ref" ]; then
  echo Running reference with no cudnn attention
  bash base.sh GPT5B XLA fp8 cublaslt,cudnn_ln 8
elif [ "$1" = "ref_cudnn" ]; then
  echo Running reference with cudnn attention pattern matching
  bash base.sh GPT5B XLA fp8 cublaslt,cudnn_ln,cudnn_fmha 8
else
  echo Running cudnn attention
  bash base.sh GPT5B XLA fp8+fa cublaslt,cudnn_ln 8
fi
#bash base.sh GPT5B XLA bf16 triton_gemm,cudnn_ln,cudnn_fmha 8
#bash base.sh GPT5B TE fp8 none 8
#bash base.sh GPT5B TE bf16 none 8

