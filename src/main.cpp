#include <SFML/Graphics.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <filesystem>
#include "Game.hpp"

int main() {
    // Configure logging
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");
    
    // Create necessary directories
    std::filesystem::create_directories("assets");
    std::filesystem::create_directories("models");
    std::filesystem::create_directories("data");
    std::filesystem::create_directories("logs");
    
    // Log to file and console
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/game.log", true);
    
    std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>("SnakeAI", sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);
    
    spdlog::info("=== SnakeAI-MLOps Starting ===");
    
    try {
        Game game;
        game.run();
    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return -1;
    }
    
    spdlog::info("=== SnakeAI-MLOps Shutdown Complete ===");
    return 0;
}