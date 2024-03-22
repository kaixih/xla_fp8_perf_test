printf "%-18s %8s %4s %4s %30s %4s %9s %8s\n" NETWORK BACKEND MATH SDPA XLA_EXTRAS GPUs STEPS/SEC WALLSECS

bash base.sh Synthetic5B XLA fp8 NA cublaslt,cudnn_ln 8
bash base.sh Synthetic5B XLA fp8 FA cublaslt,cudnn_ln 8
#bash base.sh Synthetic5B XLA bf16 NA triton_gemm,cudnn_ln 8 # OOM
bash base.sh Synthetic5B XLA bf16 FA triton_gemm,cudnn_ln 8
#bash base.sh GPT5BSynthetic TE fp8 none 8
#bash base.sh GPT5BSynthetic TE bf16 none 8

