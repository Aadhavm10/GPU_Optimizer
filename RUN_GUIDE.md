# Run Guide — GPU Utilization Optimizer

This guide provides minimal, practical steps to get the project running and to reproduce the demos.

Prerequisites
- NVIDIA GPU with CUDA compute capability 7.0+ and up-to-date drivers.
- Python 3.8+ (recommended 3.8–3.11).
- Optional (for compiling CUDA/C++): CUDA Toolkit 11.0+ and nvcc; on Windows, Visual Studio Build Tools.

Quick setup
1. Clone the repository
   git clone https://github.com/Aadhavm10/GPU_Optimzer.git
   cd GPU_Optimzer

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

Run the CUDA performance demo
- If binaries are not provided, compile the demo:
  nvcc -arch=sm_75 -O3 -o cuda_demo cuda_demo.cu
- Run the Python demo script:
  python simple_cuda_demo.py
- The script runs matrix benchmarks and prints CPU vs GPU timings.

Generate GPU load (optional)
- To create activity and observe dashboard changes:
  python gpu_load_test.py

Build native components (optional)
- Windows:
  .\build.bat
- Linux/macOS:
  ./build.sh

Common troubleshooting
- NVML/library not found: update NVIDIA drivers and ensure driver libraries are on the system path.
- Python module errors: verify virtual environment activation and run pip install -r requirements.txt.
- Dashboard port in use: check port 8050 and either stop the occupying process or change the port in simple_dashboard.py.
- CUDA compilation errors: confirm CUDA Toolkit and nvcc are installed and the target architecture (sm_XX) matches your GPU.

Notes
- These instructions assume local development access and administrative rights for driver/tool installs.
- For multi-node or containerized deployments, ensure consistent driver and CUDA versions across nodes and integrate with your scheduler/container runtime.
