# 训练一步的时间分解（torch.profiler）与 MFU

- 配置：{'micro_batch': 32, 'accum': 16, 'seq_len': 1024, 'precision': 'bf16', 'compile': True}，每步 524,288 token
- 编译版：每步 11.164 s，46,964 tok/s，峰值显存 62.73 GB；不编译时显存不足（OOM）：torch.compile 的算子融合也省下了中间激活的显存
- 每 token 计算量 ≈ 6·(层内参数 450.9M + 输出投影 32.8M) + 12·L·d·T = 3.41 GFLOPs（注意力 QKᵀ/AV 占 15%）
- 实际算力 159.9 TFLOPS，**MFU = 51.3%**（A800 bf16 稠密峰值 312 TFLOPS）
- profiler 抓的那一步：墙钟 11.701 s，GPU kernel 合计 11.041 s，**GPU 忙碌 94.4%**（其余是 CPU 调度、数据搬运、同步造成的空隙；profiler 本身也有少量开销）

## 按类别

| 类别 | GPU 时间 (s) | 占比 |
|---|---|---|
| 矩阵乘（GEMM） | 7.11 | 64.4% |
| Triton 融合 kernel | 2.222 | 20.1% |
| 注意力（FlashAttention 等） | 1.603 | 14.5% |
| 其它 | 0.087 | 0.8% |
| 优化器 | 0.013 | 0.1% |
| 拷贝 / 清零 | 0.006 | 0.1% |

## 耗时最多的 kernel

| kernel | 调用次数 | GPU 时间 (s) | 占比 |
|---|---|---|---|
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` | 2560 | 1.5453 | 14.0% |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_nt` | 1280 | 1.1753 | 10.6% |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nn` | 3200 | 1.0924 | 9.9% |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_nn` | 1280 | 1.0911 | 9.9% |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nt` | 1936 | 1.0792 | 9.8% |
| `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false` | 640 | 1.0122 | 9.2% |
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` | 1936 | 0.861 | 7.8% |
| `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t, Flash_k` | 640 | 0.3862 | 3.5% |
| `triton_poi_fused__unsafe_view_add_fill_mul_sigmoid_silu_sub_view_5` | 640 | 0.3429 | 3.1% |
| `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11` | 1248 | 0.2889 | 2.6% |
| `triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_16` | 624 | 0.2374 | 2.1% |
| `triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_7` | 624 | 0.2162 | 2.0% |
| `triton_poi_fused__unsafe_view_mul_silu_9` | 640 | 0.2065 | 1.9% |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn` | 16 | 0.1414 | 1.3% |
| `triton_red_fused__to_copy_add_mul_sum_view_15` | 624 | 0.139 | 1.3% |

## 怎么读

- 矩阵乘占比越高越好（说明时间花在"有用"的计算上）；Triton 融合 kernel / 拷贝占比高，说明逐元素操作和访存是瓶颈。
- GPU 忙碌明显低于 100%：查数据加载（是否预取）、Python 开销、频繁的 .item()/float() 同步。
- MFU 30–50% 对这种规模的单卡训练是正常区间；继续提升通常靠更大的 micro batch、更长的序列、融合优化器 / 注意力。
- 时间线：把 `trace.json.gz` 解压后在 https://ui.perfetto.dev 打开，能看到每个微批的前向、反向和优化器。
