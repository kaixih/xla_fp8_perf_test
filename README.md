# XLA-FP8 Tests

```
$ cd xla_fp8_perf_test/L2_xla_fp8_benchmark
# base test.sh
NETWORK            OPT_MODE MATH                          XLA_EXTRAS GPUs STEPS/SEC WALLSECS
Synthetic5B             XLA  fp8        cublaslt,cudnn_ln,cudnn_fmha    8     1.012      180
Synthetic5B             XLA bf16     triton_gemm,cudnn_ln,cudnn_fmha    8     0.639      240
```

```
$ cd xla_fp8_perf_test/L1_xla_fp8_gemms
$ bash test.sh
Checking model with no repeated layer:
FWD CHECKING ... Pass
BWD CHECKING ... Pass
Checking model with repeated layer:
FWD CHECKING ... Pass
BWD CHECKING ... Pass
```



