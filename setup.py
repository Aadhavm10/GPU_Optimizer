#!/usr/bin/env python3

from setuptools import setup, find_packages

# Get project version
def get_version():
    """Extract version from CMakeLists.txt or use default"""
    try:
        with open('CMakeLists.txt', 'r') as f:
            content = f.read()
            # Look for version in CMakeLists.txt
            for line in content.split('\n'):
                if 'project(' in line and 'VERSION' in line:
                    return line.split('VERSION')[1].split()[0].strip()
    except:
        pass
    return "1.0.0"

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
            requirements = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove comments and optional dependencies
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if line and not line.startswith('cupy') and not line.startswith('numba'):
                        requirements.append(line)
            return requirements
    except FileNotFoundError:
        return []

# Pure-Python package: the CUDA kernels and NVML monitor are built separately
# with CMake (see build.sh / build.bat), so installing this package needs
# neither pybind11 nor a CUDA toolkit.
setup(
    name="gpu-utilization-optimizer",
    version=get_version(),
    author="GPU Optimizer Team",
    author_email="gpu-optimizer@example.com",
    description="Comprehensive CUDA-based GPU profiling and optimization tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gpu-utilization-optimizer",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
        "cuda": [
            "cupy-cuda11x>=9.0.0",
            "numba>=0.54.0",
        ],
        "ml": [
            "tensorflow>=2.7.0",
            "torch>=1.10.0",
            "scikit-learn>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gpu-dashboard=gpu_optimizer.dashboard.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gpu_optimizer": [
            "dashboard/templates/*.html",
            "dashboard/static/css/*.css",
            "dashboard/static/js/*.js",
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    zip_safe=False,
)
