#include "matrix_kernels.cuh"

#include <cstddef>
#include <cstdint>

namespace {

// Element (r, c) of a rows x cols row-major matrix, or 0 outside it. Used to
// zero-pad partial tiles at the matrix edges.
__device__ __forceinline__ float load_or_zero(const float* __restrict__ m,
                                              int r, int c, int rows, int cols) {
    return (r < rows && c < cols) ? m[static_cast<size_t>(r) * cols + c] : 0.0f;
}

// ---------------------------------------------------------------------------
// Naive: one thread per output element. Every thread streams a full row of A
// and column of B from global memory, so each input is re-read N (or M) times.
// ---------------------------------------------------------------------------
__global__ void naive_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
    }
    C[static_cast<size_t>(row) * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Tiled: a TILE x TILE thread block computes a TILE x TILE output tile. Each
// step loads one tile of A and one of B into shared memory (one element per
// thread), then every thread does TILE multiply-adds from shared memory, so
// each global element is read once per tile instead of once per output.
//
// Bank conflicts: within a warp, As[ty][k] is the same word for every thread
// in a row (a broadcast) and Bs[k][tx] is consecutive words, so this layout
// is conflict-free without padding.
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void tiled_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < K; t += TILE) {
        As[ty][tx] = load_or_zero(A, row, t + tx, M, K);
        Bs[ty][tx] = load_or_zero(B, t + ty, col, K, N);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();  // everyone is done reading before the next tile overwrites it
    }

    if (row < M && col < N) {
        C[static_cast<size_t>(row) * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Double-buffered: same 32x32 tiling, but with two shared-memory buffers. The
// global loads for tile t+1 go into registers before the multiply-adds on
// tile t, so their latency overlaps the math, and they are written to the
// other buffer afterwards. Because the next tile never overwrites the one
// being read, one __syncthreads() per tile is enough instead of two.
// ---------------------------------------------------------------------------
constexpr int kDbTile = 32;

__global__ void double_buffered_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                              float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[2][kDbTile][kDbTile];
    __shared__ float Bs[2][kDbTile][kDbTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kDbTile + ty;
    const int col = blockIdx.x * kDbTile + tx;
    const int num_tiles = (K + kDbTile - 1) / kDbTile;

    As[0][ty][tx] = load_or_zero(A, row, tx, M, K);
    Bs[0][ty][tx] = load_or_zero(B, ty, col, K, N);
    __syncthreads();

    float acc = 0.0f;
    for (int t = 0; t < num_tiles; ++t) {
        const int cur = t & 1;
        const bool has_next = t + 1 < num_tiles;

        float a_next = 0.0f;
        float b_next = 0.0f;
        if (has_next) {
            a_next = load_or_zero(A, row, (t + 1) * kDbTile + tx, M, K);
            b_next = load_or_zero(B, (t + 1) * kDbTile + ty, col, K, N);
        }

#pragma unroll
        for (int k = 0; k < kDbTile; ++k) {
            acc += As[cur][ty][k] * Bs[cur][k][tx];
        }

        if (has_next) {
            As[cur ^ 1][ty][tx] = a_next;
            Bs[cur ^ 1][ty][tx] = b_next;
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[static_cast<size_t>(row) * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// float4 / register-blocked: a 32x32 output tile per block of 8x32 threads.
// Each thread computes 4 adjacent outputs of one row, keeps them in a float4
// accumulator, and reuses each A value it reads from shared memory 4 times.
// Global loads, the B tile and the C store all use 16-byte float4 accesses.
// ---------------------------------------------------------------------------
constexpr int kVecTile = 32;
constexpr int kVecWidth = 4;

__global__ void float4_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                     float* __restrict__ C, int M, int N, int K) {
    // A warp is 8 threads (tx) x 4 rows (ty). With a 32-float row stride,
    // As[ty][k] for those 4 rows would all fall in the same bank (a 4-way
    // conflict). Padding each row by 4 floats moves the rows to different
    // banks and keeps every row 16-byte aligned for the float4 stores.
    __shared__ __align__(16) float As[kVecTile][kVecTile + kVecWidth];
    __shared__ float4 Bs[kVecTile][kVecTile / kVecWidth];

    const int tx = threadIdx.x;  // 0..7: which group of 4 columns
    const int ty = threadIdx.y;  // 0..31: row within the tile
    const int row = blockIdx.y * kVecTile + ty;
    const int col = blockIdx.x * kVecTile + tx * kVecWidth;  // first of this thread's 4 columns
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float4 acc = zero;
    for (int t = 0; t < K; t += kVecTile) {
        // N and K are multiples of 4, so a float4 is either fully inside a row or fully outside it.
        const int a_col = t + tx * kVecWidth;
        const float4 a = (row < M && a_col < K)
            ? *reinterpret_cast<const float4*>(&A[static_cast<size_t>(row) * K + a_col])
            : zero;
        *reinterpret_cast<float4*>(&As[ty][tx * kVecWidth]) = a;

        const int b_row = t + ty;
        Bs[ty][tx] = (b_row < K && col < N)
            ? *reinterpret_cast<const float4*>(&B[static_cast<size_t>(b_row) * N + col])
            : zero;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kVecTile; ++k) {
            const float a_k = As[ty][k];
            const float4 b = Bs[k][tx];
            acc.x += a_k * b.x;
            acc.y += a_k * b.y;
            acc.z += a_k * b.z;
            acc.w += a_k * b.w;
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        *reinterpret_cast<float4*>(&C[static_cast<size_t>(row) * N + col]) = acc;
    }
}

bool valid_shape(int M, int N, int K) { return M > 0 && N > 0 && K > 0; }

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

dim3 grid_for(int M, int N, int tile) {
    return dim3((N + tile - 1) / tile, (M + tile - 1) / tile);
}

}  // namespace

cudaError_t launch_naive_matmul(const float* A, const float* B, float* C,
                                int M, int N, int K, cudaStream_t stream) {
    if (!valid_shape(M, N, K)) return cudaErrorInvalidValue;
    const dim3 block(16, 16);
    naive_matmul_kernel<<<grid_for(M, N, 16), block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

cudaError_t launch_tiled16_matmul(const float* A, const float* B, float* C,
                                  int M, int N, int K, cudaStream_t stream) {
    if (!valid_shape(M, N, K)) return cudaErrorInvalidValue;
    const dim3 block(16, 16);
    tiled_matmul_kernel<16><<<grid_for(M, N, 16), block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

cudaError_t launch_tiled32_matmul(const float* A, const float* B, float* C,
                                  int M, int N, int K, cudaStream_t stream) {
    if (!valid_shape(M, N, K)) return cudaErrorInvalidValue;
    const dim3 block(32, 32);
    tiled_matmul_kernel<32><<<grid_for(M, N, 32), block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

cudaError_t launch_double_buffered_matmul(const float* A, const float* B, float* C,
                                          int M, int N, int K, cudaStream_t stream) {
    if (!valid_shape(M, N, K)) return cudaErrorInvalidValue;
    const dim3 block(kDbTile, kDbTile);
    double_buffered_matmul_kernel<<<grid_for(M, N, kDbTile), block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

cudaError_t launch_float4_matmul(const float* A, const float* B, float* C,
                                 int M, int N, int K, cudaStream_t stream) {
    if (!valid_shape(M, N, K)) return cudaErrorInvalidValue;
    if (N % kVecWidth != 0 || K % kVecWidth != 0) return cudaErrorInvalidValue;
    if (!aligned16(A) || !aligned16(B) || !aligned16(C)) return cudaErrorInvalidValue;
    const dim3 block(kVecTile / kVecWidth, kVecTile);
    float4_matmul_kernel<<<grid_for(M, N, kVecTile), block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}
