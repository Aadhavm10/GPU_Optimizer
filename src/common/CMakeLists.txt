# Config file parsing and logging, shared by the NVML monitor
add_library(gpu_common STATIC
    config.cpp
    logger.cpp
)

target_include_directories(gpu_common PUBLIC ${PROJECT_SOURCE_DIR}/src)
