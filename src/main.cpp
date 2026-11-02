#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <signal.h>

#include "nvml/nvml_monitor.h"
#include "common/config.h"
#include "common/logger.h"

volatile bool g_running = true;

void signal_handler(int signal) {
    Logger::info("Received signal {}, shutting down gracefully...", signal);
    g_running = false;
}

void print_usage(const char* program_name) {
    std::cout << "GPU Utilization Optimizer - NVML monitor\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "  -c, --config FILE       Configuration file path\n";
    std::cout << "  -v, --verbose           Enable verbose logging\n";
    std::cout << "  -d, --daemon            Sample in the background without printing\n";
    std::cout << "  -s, --sample-rate HZ    NVML sampling rate (default: 10Hz)\n";
    std::cout << "\nFor the CUDA matmul kernels, run matmul_benchmark.\n";
}

int main(int argc, char* argv[]) {
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    Config config;
    bool daemon_mode = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--verbose") {
            config.log_level = LogLevel::DEBUG;
        } else if (arg == "-d" || arg == "--daemon") {
            daemon_mode = true;
        } else if (arg == "-s" || arg == "--sample-rate") {
            if (i + 1 < argc) {
                config.nvml_sample_rate = std::stoi(argv[++i]);
            }
        } else if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                config.config_file = argv[++i];
            }
        }
    }

    // Initialize logger
    Logger::init(config.log_level);
    Logger::info("GPU Utilization Optimizer starting...");

    try {
        auto nvml_monitor = std::make_unique<NVMLMonitor>(config);

        if (daemon_mode) {
            Logger::info("Sampling at {} Hz in the background", config.nvml_sample_rate);
            
            // Start NVML monitoring in background thread
            nvml_monitor->start_monitoring();
            
            // Main monitoring loop
            while (g_running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            nvml_monitor->stop_monitoring();
        } else {
            // Interactive mode: print the latest sample every 2 seconds.
            nvml_monitor->start_monitoring();
            
            Logger::info("Press Ctrl+C to stop monitoring...");
            while (g_running) {
                auto metrics = nvml_monitor->get_current_metrics();
                if (metrics.is_valid()) {
                    Logger::info("GPU {}: {}% util, {:.1f}% mem, {}°C", 
                               metrics.device_id, 
                               metrics.utilization.gpu_utilization_percentage,
                               metrics.memory.usage_percentage,
                               metrics.thermal.gpu_temperature_c);
                }
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            
            nvml_monitor->stop_monitoring();
        }

    } catch (const std::exception& e) {
        Logger::error("Fatal error: {}", e.what());
        return 1;
    }

    Logger::info("GPU Utilization Optimizer shutting down...");
    return 0;
}
