# SGEMM kernels; see matrix_ops/matrix_kernels.cuh for the launcher API
add_library(matmul_kernels STATIC
    matrix_ops/matrix_kernels.cu
)

target_include_directories(matmul_kernels PUBLIC ${PROJECT_SOURCE_DIR}/src)
