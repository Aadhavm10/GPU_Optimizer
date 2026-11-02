#pragma once

#include <cuda_runtime.h>

// Single-precision matrix multiply, C = A * B, all matrices row-major:
//   A is M x K, B is K x N, C is M x N.
//
// Each launcher enqueues one kernel on `stream` and returns the launch status
// (cudaSuccess, or the error from cudaGetLastError()). They return
// cudaErrorInvalidValue for shapes the kernel does not support.
//
// Kernels, from slowest to fastest:
//   naive            one thread per output, reads A and B straight from global memory
//   tiled16/tiled32  stage TILE x TILE blocks of A and B in shared memory
//   double_buffered  32x32 tiles, loads the next tile while computing the current one
//   float4           32x32 tiles, float4 loads, each thread computes 4 outputs
//                    (needs N and K divisible by 4 and 16-byte aligned pointers)

cudaError_t launch_naive_matmul(const float* A, const float* B, float* C,
                                int M, int N, int K, cudaStream_t stream = 0);

cudaError_t launch_tiled16_matmul(const float* A, const float* B, float* C,
                                  int M, int N, int K, cudaStream_t stream = 0);

cudaError_t launch_tiled32_matmul(const float* A, const float* B, float* C,
                                  int M, int N, int K, cudaStream_t stream = 0);

cudaError_t launch_double_buffered_matmul(const float* A, const float* B, float* C,
                                          int M, int N, int K, cudaStream_t stream = 0);

cudaError_t launch_float4_matmul(const float* A, const float* B, float* C,
                                 int M, int N, int K, cudaStream_t stream = 0);
