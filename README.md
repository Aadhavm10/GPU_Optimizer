# GPU Utilization Optimizer

A compact toolkit for monitoring, profiling, and demonstrating GPU performance improvements on NVIDIA hardware. The project provides real-time telemetry, a Dash web dashboard for visualization, and CUDA performance demos that illustrate common kernel optimizations.

What it is
- A monitoring and optimization demonstration toolkit that collects NVML metrics, displays live and historical charts, and includes CUDA examples (naive, tiled, optimized) to illustrate performance improvements.

Why it exists
- To make GPU performance visible and actionable: identify underutilization, reproduce benchmarks, and validate kernel- and runtime-level optimizations.

How it was created (tech stack)
- Languages: Python (primary), C++ (small components), CUDA (.cu), shell/PowerShell for automation.
- Telemetry: NVML via Python bindings (pynvml / nvidia-ml-py).
- Visualization: Dash (Plotly) served from Python (Flask).
- CUDA toolchain: CUDA Toolkit (nvcc); optional Visual Studio Build Tools on Windows.
- Packaging: pip / virtualenv; simple build scripts (build.bat / build.sh); CMake available for native builds.

Screenshots
- Web dashboard (live metrics):
  ![Dashboard overview](screenshots/dashboard-overview.png)
- Command-line monitor:
  ![Command-line monitor](screenshots/command-line-monitor.png)

Key components (relevant)
- simple_gpu_monitor.py — CLI monitor using NVML
- simple_dashboard.py / gpu_dashboard.py — Dash web dashboard
- cuda_demo.cu, simple_cuda_demo.py — CUDA kernels and Python demo
- gpu_load_test.py — synthetic load generator
- build.bat, build.sh, setup_windows.ps1, CMakeLists.txt
- requirements.txt, setup.py

Next step
- See RUN_GUIDE.md for the minimal run instructions and troubleshooting.
