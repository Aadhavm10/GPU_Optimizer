# NVML monitoring library
add_library(nvml_monitoring STATIC
    nvml_monitor.cpp
    metrics_collector.cpp
    data_processor.cpp
    gpu_metrics.cpp
)

target_include_directories(nvml_monitoring PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(nvml_monitoring PUBLIC
    gpu_common
    CUDA::nvml
    Threads::Threads
)
