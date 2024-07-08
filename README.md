# XLA-FP8 Tests

## Functionality test
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


## Performance test
```
$ cd xla_fp8_perf_test/L2_xla_fp8_benchmark
# base test.sh
NETWORK             BACKEND MATH SDPA                     XLA_EXTRAS GPUs STEPS/SEC WALLSECS
Synthetic5B             XLA  fp8   NA              cublaslt,cudnn_ln    8     0.788      202
Synthetic5B             XLA  fp8   FA              cublaslt,cudnn_ln    8     1.126      162
Synthetic5B             XLA bf16   NA           triton_gemm,cudnn_ln    8     0.647      227
Synthetic5B             XLA bf16   FA           triton_gemm,cudnn_ln    8     0.858      197

```

## Convergence test
```
root@d6a0f3181d5a:/home/repo/xla_fp8_perf_test/L3_xla_fp8_convergence# bash test.sh
NETWORK             BACKEND MATH SDPA           XLA_EXTRAS GPUs STEPS/SEC LOSS WALLSECS
Running direct cudnn attention
XLA DUMP TO PATH /xla_dump
/tmp.uwBN5s
GPT5B                   XLA  fp8   FA    cublaslt,cudnn_ln    8     1.028 3.847     1619
[PAX STATUS]: Starting training loop.
[PAX STATUS] step_i: 100, training loss: 6.5196137
[PAX STATUS] step_i: 200, training loss: 5.92186
[PAX STATUS] step_i: 300, training loss: 5.3081756
[PAX STATUS] step_i: 400, training loss: 5.589448
[PAX STATUS] step_i: 500, training loss: 5.206719
[PAX STATUS] step_i: 600, training loss: 5.1764674
[PAX STATUS] step_i: 700, training loss: 4.906096
[PAX STATUS] step_i: 800, training loss: 4.590818
[PAX STATUS] step_i: 900, training loss: 4.5946627
[PAX STATUS] step_i: 1000, training loss: 4.365218
[PAX STATUS] step_i: 1100, training loss: 4.404575
[PAX STATUS] step_i: 1200, training loss: 4.2801576
[PAX STATUS] step_i: 1300, training loss: 4.2293515
[PAX STATUS] step_i: 1400, training loss: 4.029117
[PAX STATUS] step_i: 1500, training loss: 3.8467343

