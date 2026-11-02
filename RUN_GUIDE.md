# Run Guide — GPU Utilization Optimizer

This guide provides minimal, practical steps to get the project running and to reproduce the demos.

Prerequisites
- NVIDIA GPU with CUDA compute capability 7.0+ and up-to-date drivers.
- Python 3.8+ (recommended 3.8–3.11).
- Optional (for compiling CUDA/C++): CUDA Toolkit 11.0+ and nvcc; on Windows, Visual Studio Build Tools.

Quick setup
1. Clone the repository
   git clone https://github.com/Aadhavm10/GPU_Optimizer.git
   cd GPU_Optimizer

2. Create and activate a Python virtual environment
   python -m venv venv
   (Windows) .\venv\Scripts\Activate.ps1
   (Linux/macOS) source venv/bin/activate

3. Install Python dependencies
   pip install --upgrade pip
   pip install -r requirements.txt

Run the CLI monitor
- Start the command-line monitor:
  python simple_gpu_monitor.py
  Expected: printed GPU name, utilization, memory, temperature, and power.

Run the web dashboard
- Start the dashboard server:
  python simple_dashboard.py
- Open: http://localhost:8050
- The dashboard refreshes metrics every ~2 seconds.

Build the CUDA kernels and C++ monitor
- Requires the CUDA Toolkit 11.0+ (nvcc) and CMake 3.20+.
- Set CMAKE_CUDA_ARCHITECTURES to your GPU: 75 = GTX 16xx / RTX 20xx, 86 = RTX 30xx, 89 = RTX 40xx.
  cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75
  cmake --build build --config Release
- Or run the wrapper scripts, which also set up the Python environment:
  (Windows) .\build.bat
  (Linux) ./build.sh
- With the Visual Studio generator on Windows, binaries land in build\Release\.

Validate and benchmark the CUDA kernels
- Check every kernel against the CPU reference on square and odd-sized shapes:
  ./build/matmul_benchmark --check
  (ctest --test-dir build runs the same check.)
- Time every kernel on N x N matrices (default 1024):
  ./build/matmul_benchmark 2048
  Expected: one row per kernel with time, GFLOP/s, speedup over the single-threaded
  CPU reference, max error, and PASS/FAIL. Large N takes a while because the CPU
  reference is computed first.
- Run it with the dashboard open to see utilization and power change.

Run the C++ NVML monitor
- ./build/gpu_optimizer prints utilization, memory, and temperature every 2 seconds.
  ./build/gpu_optimizer --help lists the options.

Common troubleshooting
- NVML/library not found: update NVIDIA drivers and ensure driver libraries are on the system path.
- Python module errors: verify virtual environment activation and run pip install -r requirements.txt.
- Dashboard port in use: check port 8050 and either stop the occupying process or change the port in simple_dashboard.py.
- CUDA compilation errors: confirm CUDA Toolkit and nvcc are installed and the target architecture (sm_XX) matches your GPU.

Notes
- These instructions assume local development access and administrative rights for driver/tool installs.
- For multi-node or containerized deployments, ensure consistent driver and CUDA versions across nodes and integrate with your scheduler/container runtime.
