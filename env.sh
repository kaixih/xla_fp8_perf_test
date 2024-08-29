# TODO(kaixih): nvbug/4833198 and nvbug/4830674
export XLA_COMMON=" \
    --xla_gpu_enable_triton_gemm=false \
    --xla_gpu_enable_pipelined_all_reduce=false \
    --xla_gpu_enable_pipelined_all_gather=false \
    --xla_gpu_enable_pipelined_reduce_scatter=false \
"
 
