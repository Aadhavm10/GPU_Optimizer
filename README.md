# GPU Utilization Optimizer

Real-time NVIDIA GPU monitoring dashboard plus hand-written CUDA kernels that take matrix multiplication from naive to optimized.

![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Plotly Dash](https://img.shields.io/badge/Plotly%20Dash-3F4F75?logo=plotly&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-064F8C?logo=cmake&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

![Dashboard overview](screenshots/dashboard-overview.png)

## Highlights

- **Live GPU telemetry:** reads utilization, memory, temperature and power through NVIDIA's NVML API and charts them in a Plotly Dash web dashboard that refreshes every 2 seconds.
- **CUDA kernel optimization:** matrix multiply written six ways (naive, tiled, shared memory, `float4`-vectorized, double-buffered, warp-tiled), plus parallel reductions and vector ops (SAXPY, dot product, norm, min/max).
- **Native C++ monitor:** NVML device discovery, metrics collection and processing, config and logging in C++17, built with CMake.
- **Cross-platform tooling:** build and setup scripts for both Windows and Linux.

## Quick start

Requires an NVIDIA GPU with current drivers.

```bash
git clone https://github.com/Aadhavm10/GPU_Optimizer.git
cd GPU_Optimizer
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python simple_dashboard.py      # then open http://localhost:8050
```

For the CLI monitor, the CUDA benchmark build and troubleshooting, see [RUN_GUIDE.md](RUN_GUIDE.md).

## Command-line monitor

![Command-line monitor](screenshots/command-line-monitor.png)

## Project structure

```
src/cuda/              CUDA kernels (matrix_ops, vector_ops) and profiling timers
src/nvml/              C++ NVML monitor and metrics collector
src/common/            Config and logging
simple_dashboard.py    Dash web dashboard
simple_gpu_monitor.py  Command-line monitor
cuda_demo.cu           Standalone matmul benchmark (naive / tiled / optimized)
gpu_load_test.py       Synthetic GPU load to exercise the dashboard
CMakeLists.txt         Native C++/CUDA build
```

## License

[MIT](LICENSE)
