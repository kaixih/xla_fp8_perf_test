# XLA-FP8 Tests

```
$ cd xla_fp8_perf_test/L2_xla_fp8_benchmark
# base test.sh full
NETWORK            OPT_MODE MATH                          XLA_EXTRAS GPUs STEPS/SEC WALLSECS
GPT5BSynthetic          XLA  fp8                   cublaslt,cudnn_ln    8     0.694      210
GPT5BSynthetic          XLA bf16     triton_gemm,cudnn_ln,cudnn_fmha    8     0.568      249
GPT5BSynthetic          XLA  fp8                                none    8     0.663      210
GPT5BSynthetic          XLA  fp8                            cublaslt    8     0.685      206
GPT5BSynthetic          XLA  fp8                         triton_gemm    8     0.639      228
GPT5BSynthetic          XLA  fp8        cublaslt,cudnn_ln,cudnn_fmha    8     0.663      218
GPT5BSynthetic          XLA bf16                                none    8     0.557      236
GPT5BSynthetic          XLA bf16                         triton_gemm    8     0.547      250
GPT5BSynthetic          XLA bf16                triton_gemm,cudnn_ln    8     0.552      255
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



