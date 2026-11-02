"""
GPU Utilization Optimizer

Python side of the project: the live NVML dashboard (``gpu_optimizer.dashboard``,
run with ``gpu-dashboard``) and a requirements check. The CUDA kernels, their
benchmark and the C++ NVML monitor are built separately with CMake.
"""

__version__ = "1.0.0"
__author__ = "GPU Optimizer Team"
__email__ = "gpu-optimizer@example.com"


def check_requirements():
    """Check that NVML can see an NVIDIA GPU, which the dashboard needs."""
    issues = []

    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        if gpu_count == 0:
            issues.append("No NVIDIA GPUs detected")
    except Exception as e:
        issues.append(f"NVML initialization failed: {e}")

    if issues:
        print("Requirements check failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("All requirements met!")
    return True


__all__ = ["check_requirements", "__version__", "__author__"]
