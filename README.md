# GPU Utilization Optimizer

Real-time NVIDIA GPU monitoring dashboard plus CUDA SGEMM kernels that take matrix multiplication from naive to register-blocked, each validated against a CPU reference.

![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Plotly Dash](https://img.shields.io/badge/Plotly%20Dash-3F4F75?logo=plotly&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-064F8C?logo=cmake&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

![Dashboard overview](screenshots/dashboard-overview.png)

## Highlights

- **Live GPU telemetry:** reads utilization, memory, temperature and power through NVIDIA's NVML API and charts them in a Plotly Dash web dashboard that refreshes every 2 seconds.
- **Four CUDA SGEMM kernels,** each a step up from the last:
  1. naive, one thread per output
  2. shared-memory tiled (16×16 and 32×32 tiles)
  3. double-buffered, which loads the next tile while computing the current one
  4. register-blocked, with `float4` loads and 4 outputs per thread
- **Validated, not just timed:** `matmul_benchmark` compares every kernel against a CPU reference before timing it. It reports time, GFLOP/s, speedup over the CPU and max error. `--check` (run by `ctest`) also tests non-square shapes that aren't multiples of the tile size.
- **C++ NVML monitor:** a background sampling thread plus metrics collection and processing in C++17. The CMake build compiles it alongside the kernels.

## Results

`matmul_benchmark 2048` on a Tesla T4 (Google Colab), 2048×2048 float32:

| Kernel | Time (ms) | GFLOP/s | vs CPU |
|---|---:|---:|---:|
| naive | 39.3 | 437 | 46× |
| tiled 16×16 | 26.1 | 659 | 70× |
| tiled 32×32 | 19.8 | 869 | 92× |
| double-buffered 32×32 | 19.4 | 886 | 94× |
| float4, 4 outputs/thread | 13.5 | 1,277 | 135× |

- **Correctness:** every kernel matches the CPU reference, with a max abs error of 3.8e-5. All 45 `--check` cases pass.
- **Timing:** kernel-only, the mean of 10 launches measured with CUDA events.
- **CPU baseline:** "vs CPU" compares against a single-threaded triple loop, which took 1.82 s (9.45 GFLOP/s).
- **Headroom:** the fastest kernel reaches about 16% of the T4's 8.1 TFLOPS FP32 peak.

## Quick start

Requires an NVIDIA GPU with current drivers.

**Dashboard (Python):**

```bash
git clone https://github.com/Aadhavm10/GPU_Optimizer.git
cd GPU_Optimizer
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python simple_dashboard.py      # then open http://localhost:8050
```

**CUDA kernels (needs the CUDA Toolkit and CMake 3.20+):**

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75   # 86 for RTX 30xx, 89 for RTX 40xx
cmake --build build --config Release
./build/matmul_benchmark --check                     # correctness on odd shapes
./build/matmul_benchmark 2048                        # timing table
```

[RUN_GUIDE.md](RUN_GUIDE.md) covers the CLI monitor and troubleshooting.

## Command-line monitor

![Command-line monitor](screenshots/command-line-monitor.png)

## Project structure

```
src/cuda/matrix_ops/        SGEMM kernels and launchers
benchmarks/                 matmul_benchmark: CPU validation + CUDA-event timing
src/nvml/, src/common/      C++ NVML monitor, config, logging
src/main.cpp                gpu_optimizer CLI (C++ monitor)
src/python/gpu_optimizer/   Dash web dashboard (gpu-dashboard after pip install -e .)
simple_dashboard.py         Runs the dashboard without installing the package
simple_gpu_monitor.py       Python command-line monitor
CMakeLists.txt              Builds the kernels, benchmark and C++ monitor
```

## License

[MIT](LICENSE)
