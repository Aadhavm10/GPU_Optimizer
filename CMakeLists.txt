cmake_minimum_required(VERSION 3.20)

# Default to Turing (sm_75: GTX 16xx, RTX 20xx). PTX is embedded too, so newer
# GPUs JIT-compile it on first launch. Build native code for your card with
# e.g. -DCMAKE_CUDA_ARCHITECTURES=86.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(gpu_utilization_optimizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)  # also provides CUDA::nvml
find_package(Threads REQUIRED)

add_subdirectory(src/common)
add_subdirectory(src/cuda)
add_subdirectory(src/nvml)

# SGEMM kernel benchmark and correctness check
add_executable(matmul_benchmark benchmarks/matmul_benchmark.cu)
target_link_libraries(matmul_benchmark PRIVATE matmul_kernels)

# NVML monitor CLI
add_executable(gpu_optimizer src/main.cpp)
target_link_libraries(gpu_optimizer PRIVATE nvml_monitoring)

# Needs an NVIDIA GPU: checks every kernel against the CPU reference.
enable_testing()
add_test(NAME matmul_kernels_match_cpu_reference COMMAND matmul_benchmark --check)

install(TARGETS matmul_benchmark gpu_optimizer DESTINATION bin)
