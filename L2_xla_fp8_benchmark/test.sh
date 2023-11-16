
printf "%-30s %7s %4s %11s %5s %9s %8s\n" NETWORK OPT_MODE MATH TRITON_GEMM GPUs STEPS/SEC WALLSECS

bash base.sh GPT5BSynthetic XLA fp8 no 8
bash base.sh GPT5BSynthetic XLA fp8 yes 8
bash base.sh GPT5BSynthetic XLA bf16 no 8
bash base.sh GPT5BSynthetic XLA bf16 yes 8

#bash base.sh GPT175BSynthetic XLA fp8 no 8
#bash base.sh GPT175BSynthetic XLA bf16 yes 8

