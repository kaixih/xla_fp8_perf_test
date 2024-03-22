printf "%-18s %8s %4s %4s %20s %4s %9s %4s %8s\n" NETWORK BACKEND MATH SDPA XLA_EXTRAS GPUs STEPS/SEC LOSS WALLSECS

if [ "$1" = "default" ]; then
  echo Running with no cudnn attention
  bash base.sh GPT5B XLA fp8 NA cublaslt,cudnn_ln 8
elif [ "$1" = "pattern_match" ]; then
  echo Running pattern-match cudnn attention
  bash base.sh GPT5B XLA fp8 NA cublaslt,cudnn_ln,cudnn_fmha 8
else
  echo Running direct cudnn attention
  bash base.sh GPT5B XLA fp8 FA cublaslt,cudnn_ln 8
fi
#bash base.sh GPT5B XLA bf16 triton_gemm,cudnn_ln,cudnn_fmha 8
#bash base.sh GPT5B TE fp8 none 8
#bash base.sh GPT5B TE bf16 none 8

