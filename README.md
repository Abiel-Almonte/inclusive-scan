# Device-Wide Prefix-Sum (Inclusive Scan)

High-performance CUDA implementation of an **inclusive scan** (prefix-sum) over large, device-resident vectors.  A single, fused kernel `single_pass_scan_4x` reaches DRAM throughput on an RTX 4070 Ti SUPER comparable to copy bandwidth and outperforms NVIDIA CUB’s production `DeviceScan::InclusiveSum` by **up to 1.5 ×**.

---

## 1. Results

Peak throughput: **≈ 613 GB/s** (93.4 % of theoretical DRAM bandwidth) at N ≈ 256 M elements, delivering a **1.5 ×** speed-up over CUB on the test GPU.

<p align="center">
  <img src="visualization/performance_chart.png" alt="Performance" width="70%">
</p>

<p align="center">
  <img src="visualization/roofline_chart.png" alt="Roofline" width="70%">
</p>

---

## 2. Algorithm & Kernel

* **Vectorised Kogge-Stone warp scan** on 4-element `float4` inputs (`_4x`).
* **Decoupled look-back** (Merrill & Garland) across blocks using a shared-memory prefix accumulator.
* **Single-pass fusion** of the traditional three-kernel GPU scan (block scan, prefix scan of block sums, add-back) into one launch.
* 512-thread blocks (16 warps) with cooperative warp-level intrinsics (`__shfl_*`).
* **Auto-tuned** launch configuration (block size, vector width, memory layout) via Optuna Bayesian HPO over a Jinja2-templated kernel generator.

---

## 3. Baseline Competitor

[NVIDIA CUB](https://nvidia.github.io/cub/) v2.x `cub::DeviceScan::InclusiveSum` (the standard for GPU prefix-scans).

During profiling we treat CUB’s two internal kernels — `DeviceScanInitKernel` and `DeviceScanKernel` — as a single logical operation and time-weight their metrics for a fair comparison.

---

## 4. Measurement Methodology

1. **Profiler:** `Nvidia Nsight Compute` via `build_and_run.sh profile <N>` 
2. **Metrics scraped:**
   * Memory Throughput (GByte/s)
   * DRAM Throughput (%) – “roofline” efficiency
   * Duration (µs / ms) → converted to seconds for weighting
3. **Problem sizes:** powers of two from 2^10 to 2^28 plus ±1 neighbours to observe cache edge performance.

---

## 5. Reproducing the Benchmark

```bash
# CUDA 12.x, Nsight Compute 2024+, Python 3.10 with matplotlib & pandas

# 1. Build & profile all sizes
python visualization/visualize_benchmark.py

# 2. Plot only (reuse previous CSV)
python visualization/visualize_benchmark.py --plot-only
```

Environment-specific tuning (SM clock, power limits) was **disabled** to publish vendor-neutral numbers.
