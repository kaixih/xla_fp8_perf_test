# XLA-FP8 Tests

```
$ cd xla_fp8_perf_test/L2_xla_fp8_benchmark
# base test.sh
NETWORK            OPT_MODE MATH                          XLA_EXTRAS GPUs STEPS/SEC WALLSECS
GPT5BSynthetic          XLA  fp8                   cublaslt,cudnn_ln    8     0.709      207
GPT5BSynthetic          XLA bf16     triton_gemm,cudnn_ln,cudnn_fmha    8     0.570      248
GPT5BSynthetic           TE  fp8                                none    8     0.813      181
GPT5BSynthetic           TE bf16                                none    8     0.660      200
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



