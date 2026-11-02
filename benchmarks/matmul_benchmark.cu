// Benchmarks and validates the SGEMM kernels in src/cuda/matrix_ops.
//
//   matmul_benchmark [N]    time every kernel on N x N matrices (default 1024)
//   matmul_benchmark --check
//                           correctness only, on square and non-square shapes
//                           that are not multiples of the tile size; exits
//                           non-zero on any mismatch (this is what ctest runs)
//
// Every kernel's output is compared against a CPU reference before it is
// timed, and speedups are measured against that same CPU run.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "cuda/matrix_ops/matrix_kernels.cuh"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err_));                               \
            std::exit(2);                                                         \
        }                                                                         \
    } while (0)

namespace {

using Launcher = cudaError_t (*)(const float*, const float*, float*, int, int, int, cudaStream_t);

struct Kernel {
    const char* name;
    Launcher launch;
};

const Kernel kKernels[] = {
    {"naive", launch_naive_matmul},
    {"tiled 16x16", launch_tiled16_matmul},
    {"tiled 32x32", launch_tiled32_matmul},
    {"double-buffered 32x32", launch_double_buffered_matmul},
    {"float4, 4 outputs/thread", launch_float4_matmul},
};

struct Shape {
    int M, N, K;
};

// Single-threaded i-k-j loop. Each C[i][j] is summed in increasing k, the
// same order the GPU kernels use, so the only differences are rounding.
void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    std::fill(C, C + static_cast<size_t>(M) * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        float* c_row = C + static_cast<size_t>(i) * N;
        for (int k = 0; k < K; ++k) {
            const float a = A[static_cast<size_t>(i) * K + k];
            const float* b_row = B + static_cast<size_t>(k) * N;
            for (int j = 0; j < N; ++j) c_row[j] += a * b_row[j];
        }
    }
}

// Inputs are uniform in [-1, 1]. The GPU fuses multiply-adds (FMA) and the CPU
// loop does not, and that rounding difference grows roughly with K; indexing or
// synchronization bugs produce errors several orders of magnitude larger.
float tolerance(int K) { return 1e-5f * static_cast<float>(K); }

double gflops(const Shape& s, double ms) {
    return 2.0 * s.M * s.N * s.K / (ms * 1e-3) / 1e9;
}

class DeviceBuffers {
public:
    explicit DeviceBuffers(const Shape& s) {
        CUDA_CHECK(cudaMalloc(&a_, bytes(s.M, s.K)));
        CUDA_CHECK(cudaMalloc(&b_, bytes(s.K, s.N)));
        CUDA_CHECK(cudaMalloc(&c_, bytes(s.M, s.N)));
    }
    ~DeviceBuffers() {
        cudaFree(a_);
        cudaFree(b_);
        cudaFree(c_);
    }
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;

    static size_t bytes(int rows, int cols) { return static_cast<size_t>(rows) * cols * sizeof(float); }
    float* a() const { return a_; }
    float* b() const { return b_; }
    float* c() const { return c_; }

private:
    float* a_ = nullptr;
    float* b_ = nullptr;
    float* c_ = nullptr;
};

enum class Status { kPass, kFail, kSkipped };

struct CheckResult {
    Status status;
    float max_err;
};

// Runs the kernel once into a C buffer pre-filled with NaN (so outputs the
// kernel never writes are caught) and compares against the CPU reference.
CheckResult check_kernel(const Kernel& kernel, const Shape& s, const DeviceBuffers& d,
                         const std::vector<float>& reference, std::vector<float>& host_c) {
    const size_t c_bytes = DeviceBuffers::bytes(s.M, s.N);
    CUDA_CHECK(cudaMemset(d.c(), 0xFF, c_bytes));

    const cudaError_t err = kernel.launch(d.a(), d.b(), d.c(), s.M, s.N, s.K, 0);
    if (err == cudaErrorInvalidValue) return {Status::kSkipped, 0.0f};
    CUDA_CHECK(err);
    CUDA_CHECK(cudaDeviceSynchronize());  // surfaces faults inside the kernel
    CUDA_CHECK(cudaMemcpy(host_c.data(), d.c(), c_bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float e = std::fabs(host_c[i] - reference[i]);
        if (!(e <= max_err)) max_err = std::isnan(e) ? INFINITY : e;
    }
    return {max_err <= tolerance(s.K) ? Status::kPass : Status::kFail, max_err};
}

// Mean kernel time over `iters` back-to-back launches, measured with CUDA
// events after `warmup` untimed launches. Excludes host<->device copies.
float time_kernel(const Kernel& kernel, const Shape& s, const DeviceBuffers& d,
                  int warmup = 3, int iters = 10) {
    for (int i = 0; i < warmup; ++i) CUDA_CHECK(kernel.launch(d.a(), d.b(), d.c(), s.M, s.N, s.K, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) CUDA_CHECK(kernel.launch(d.a(), d.b(), d.c(), s.M, s.N, s.K, 0));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / iters;
}

void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

struct Problem {
    Shape s;
    std::vector<float> a, b, reference;
    double cpu_ms = 0.0;

    Problem(const Shape& shape, std::mt19937& rng)
        : s(shape),
          a(static_cast<size_t>(shape.M) * shape.K),
          b(static_cast<size_t>(shape.K) * shape.N),
          reference(static_cast<size_t>(shape.M) * shape.N) {
        fill_random(a, rng);
        fill_random(b, rng);
        const auto t0 = std::chrono::steady_clock::now();
        cpu_matmul(a.data(), b.data(), reference.data(), s.M, s.N, s.K);
        cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    void upload(const DeviceBuffers& d) const {
        CUDA_CHECK(cudaMemcpy(d.a(), a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.b(), b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
};

const char* status_name(Status s) {
    switch (s) {
        case Status::kPass: return "PASS";
        case Status::kFail: return "FAIL";
        case Status::kSkipped: return "skip";
    }
    return "?";
}

int run_check() {
    // Odd and non-square shapes exercise the edge handling of every tile size.
    const Shape shapes[] = {
        {1, 1, 1},   {7, 5, 3},     {16, 16, 16},  {32, 32, 32},  {33, 33, 33},
        {5, 4, 8},   {64, 48, 80},  {100, 36, 52}, {31, 65, 17},  {129, 96, 257},
    };
    std::mt19937 rng(42);
    int failures = 0;
    int passes = 0;

    for (const Shape& s : shapes) {
        Problem p(s, rng);
        DeviceBuffers d(s);
        p.upload(d);
        std::vector<float> host_c(p.reference.size());
        for (const Kernel& k : kKernels) {
            const CheckResult r = check_kernel(k, s, d, p.reference, host_c);
            if (r.status == Status::kSkipped) {
                std::printf("  skip  %-26s M=%-4d N=%-4d K=%-4d (needs N and K divisible by 4)\n",
                            k.name, s.M, s.N, s.K);
                continue;
            }
            std::printf("  %s  %-26s M=%-4d N=%-4d K=%-4d max abs err %.2e (tol %.1e)\n",
                        status_name(r.status), k.name, s.M, s.N, s.K, r.max_err, tolerance(s.K));
            (r.status == Status::kPass ? passes : failures)++;
        }
    }

    std::printf("\n%d passed, %d failed\n", passes, failures);
    return failures == 0 ? 0 : 1;
}

int run_benchmark(int n) {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU:    %s (compute capability %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("Matrix: %d x %d, float32, row-major\n", n, n);

    const Shape s{n, n, n};
    std::printf("CPU:    running reference (single-threaded triple loop)...\n");
    std::fflush(stdout);
    std::mt19937 rng(42);
    Problem p(s, rng);
    std::printf("CPU:    %.1f ms (%.2f GFLOP/s)\n\n", p.cpu_ms, gflops(s, p.cpu_ms));

    DeviceBuffers d(s);
    p.upload(d);
    std::vector<float> host_c(p.reference.size());

    std::printf("%-26s %10s %10s %9s %13s  %s\n", "Kernel", "Time (ms)", "GFLOP/s", "vs CPU",
                "Max abs err", "Check");
    int failures = 0;
    for (const Kernel& k : kKernels) {
        const CheckResult r = check_kernel(k, s, d, p.reference, host_c);
        if (r.status == Status::kSkipped) {
            std::printf("%-26s %10s %10s %9s %13s  skip (N must be divisible by 4)\n", k.name, "-", "-", "-", "-");
            continue;
        }
        if (r.status == Status::kFail) {
            ++failures;
            std::printf("%-26s %10s %10s %9s %13.2e  FAIL\n", k.name, "-", "-", "-", r.max_err);
            continue;
        }
        const float ms = time_kernel(k, s, d);
        std::printf("%-26s %10.3f %10.1f %8.1fx %13.2e  PASS\n", k.name, ms, gflops(s, ms), p.cpu_ms / ms,
                    r.max_err);
    }

    std::printf("\nGPU times are kernel-only: mean of 10 launches after 3 warm-up launches, timed\n"
                "with CUDA events. \"vs CPU\" is the single-threaded CPU reference time above\n"
                "divided by the kernel time.\n");
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--check") == 0) return run_check();

    int n = 1024;
    if (argc > 1) {
        n = std::atoi(argv[1]);
        if (n <= 0) {
            std::fprintf(stderr, "usage: %s [N] | --check\n", argv[0]);
            return 2;
        }
    }
    return run_benchmark(n);
}
