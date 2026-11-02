#pragma once

#include <string>
#include <memory>
#include <mutex>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "config.h"

class Logger {
public:
    static void init(LogLevel level, const std::string& log_file = "");
    static void set_level(LogLevel level);
    
    template<typename... Args>
    static void debug(const std::string& format, Args&&... args) {
        log(LogLevel::DEBUG, format, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void info(const std::string& format, Args&&... args) {
        log(LogLevel::INFO, format, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void warn(const std::string& format, Args&&... args) {
        log(LogLevel::WARN, format, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void error(const std::string& format, Args&&... args) {
        log(LogLevel::ERROR, format, std::forward<Args>(args)...);
    }

private:
    static LogLevel current_level_;
    static std::unique_ptr<std::ofstream> log_file_;
    static std::mutex log_mutex_;
    
    static std::string get_timestamp();
    static std::string level_to_string(LogLevel level);
    
    template<typename... Args>
    static void log(LogLevel level, const std::string& format, Args&&... args) {
        if (level < current_level_) return;
        
        std::lock_guard<std::mutex> lock(log_mutex_);
        
        std::string message = format_string(format, std::forward<Args>(args)...);
        std::string log_line = "[" + get_timestamp() + "] [" + level_to_string(level) + "] " + message + "\n";
        
        // Output to console
        std::cout << log_line;
        std::cout.flush();
        
        // Output to file if configured
        if (log_file_ && log_file_->is_open()) {
            *log_file_ << log_line;
            log_file_->flush();
        }
    }
    
    template<typename... Args>
    static std::string format_string(const std::string& format, Args&&... args) {
        // Simple format string implementation
        // In production, consider using fmt library or similar
        std::ostringstream oss;
        format_string_impl(oss, format, std::forward<Args>(args)...);
        return oss.str();
    }
    
    // Replaces each "{}" or "{:.Nf}" (fixed-point, N decimals) with the next
    // argument. Other text, including other braces, is copied unchanged.
    template<typename T, typename... Args>
    static void format_string_impl(std::ostringstream& oss, const std::string& format, T&& value, Args&&... args) {
        size_t search = 0;
        while (true) {
            const size_t open = format.find('{', search);
            const size_t close = (open == std::string::npos) ? std::string::npos : format.find('}', open);
            if (close == std::string::npos) {
                oss << format;  // more arguments than placeholders
                return;
            }
            const std::string spec = format.substr(open + 1, close - open - 1);
            const bool plain = spec.empty();
            const bool fixed = spec.size() >= 4 && spec.compare(0, 2, ":.") == 0 && spec.back() == 'f' &&
                               spec.find_first_not_of("0123456789", 2) == spec.size() - 1;
            if (!plain && !fixed) {
                search = open + 1;  // not a placeholder; keep looking
                continue;
            }
            std::ostringstream field;
            if (fixed) {
                field << std::fixed << std::setprecision(std::stoi(spec.substr(2, spec.size() - 3)));
            }
            field << value;
            oss << format.substr(0, open) << field.str();
            format_string_impl(oss, format.substr(close + 1), std::forward<Args>(args)...);
            return;
        }
    }
    
    static void format_string_impl(std::ostringstream& oss, const std::string& format) {
        oss << format;
    }
};






