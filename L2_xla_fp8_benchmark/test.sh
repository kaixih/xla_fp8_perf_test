printf "%-18s %8s %4s %4s %4s %9s %8s\n" NETWORK BACKEND MATH SDPA GPUs STEPS/SEC WALLSECS

bash base.sh Synthetic5B XLA fp8 NA 8
bash base.sh Synthetic5B XLA fp8 FA 8
bash base.sh Synthetic5B XLA bf16 NA 8
bash base.sh Synthetic5B XLA bf16 FA 8

#bash base.sh Llama2_7B XLA fp8 NA 8
#bash base.sh Llama2_7B XLA fp8 FA 8

