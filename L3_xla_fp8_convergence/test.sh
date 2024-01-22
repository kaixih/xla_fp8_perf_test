printf "%-18s %8s %4s %35s %4s %9s %4s %8s\n" NETWORK OPT_MODE MATH XLA_EXTRAS GPUs STEPS/SEC LOSS WALLSECS

bash base.sh GPT5B XLA fp8 cublaslt,cudnn_ln,cudnn_fmha 8
bash base.sh GPT5B XLA bf16 triton_gemm,cudnn_ln,cudnn_fmha 8
#bash base.sh GPT5B TE fp8 none 8
#bash base.sh GPT5B TE bf16 none 8

