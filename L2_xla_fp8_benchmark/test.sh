printf "%-18s %8s %4s %35s %4s %9s %8s\n" NETWORK OPT_MODE MATH XLA_EXTRAS GPUs STEPS/SEC WALLSECS

#bash base.sh Synthetic5B XLA fp8 cublaslt,cudnn_ln 8
bash base.sh Synthetic5B XLA fp8+fa cublaslt,cudnn_ln 8
##bash base.sh Synthetic5B XLA bf16 triton_gemm,cudnn_ln 8
#bash base.sh Synthetic5B XLA bf16+fa triton_gemm,cudnn_ln 8
#bash base.sh GPT5BSynthetic TE fp8 none 8
#bash base.sh GPT5BSynthetic TE bf16 none 8

