`inclusive-scan` - GPU prefix sum that hits the roof.

1.5x faster than NVIDIA CUB. 93.8% of theoretical DRAM bandwidth.

![roofline](images/roofline_chart.png)

Single-pass fused kernel. Kogge-Stone scan. Decoupled lookback.

---

> RTX 4070 Ti SUPER, CUDA 12.x, Nsight Compute profiled.
