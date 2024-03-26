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
Synthetic5B             XLA  fp8   NA              cublaslt,cudnn_ln    8     0.781      220
Synthetic5B             XLA  fp8   FA              cublaslt,cudnn_ln    8     1.060      185
Synthetic5B             XLA bf16   FA           triton_gemm,cudnn_ln    8     0.806      216
```

## Convergence test
```
root@d6a0f3181d5a:/home/repo/xla_fp8_perf_test/L3_xla_fp8_convergence# bash test.sh
NETWORK             BACKEND MATH SDPA           XLA_EXTRAS GPUs STEPS/SEC LOSS WALLSECS
Running direct cudnn attention
XLA DUMP TO PATH /xla_dump
/tmp.GTJTLl
GPT5B                   XLA  fp8   FA    cublaslt,cudnn_ln    8     0.974 3.877     1719
[PAX STATUS]: Starting training loop.
[PAX STATUS] step_i: 100, training loss: 6.3584294
[PAX STATUS] step_i: 200, training loss: 5.824545
[PAX STATUS] step_i: 300, training loss: 5.382527
[PAX STATUS] step_i: 400, training loss: 5.3988776
[PAX STATUS] step_i: 500, training loss: 5.3378267
[PAX STATUS] step_i: 600, training loss: 4.955474
[PAX STATUS] step_i: 700, training loss: 5.0833197
[PAX STATUS] step_i: 800, training loss: 4.8449764
[PAX STATUS] step_i: 900, training loss: 4.6752
[PAX STATUS] step_i: 1000, training loss: 4.821402
[PAX STATUS] step_i: 1100, training loss: 4.4439073
[PAX STATUS] step_i: 1200, training loss: 4.197976
[PAX STATUS] step_i: 1300, training loss: 3.9788842
[PAX STATUS] step_i: 1400, training loss: 4.03221
[PAX STATUS] step_i: 1500, training loss: 3.8765123
```



