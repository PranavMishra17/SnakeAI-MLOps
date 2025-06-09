#include <SFML/Graphics.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <filesystem>
#include <iostream>
#include "Game.hpp"

int main() {
    // Configure logging to console AND file
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");
    
    // Create necessary directories
    std::filesystem::create_directories("assets");
    std::filesystem::create_directories("models");
    std::filesystem::create_directories("data");
    std::filesystem::create_directories("logs");
    
    // Log to console and file simultaneously
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/debug.log", true);
    
    std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>("SnakeAI", sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);
    
    std::cout << "=== SnakeAI-MLOps Debug Starting ===" << std::endl;
    spdlog::info("=== SnakeAI-MLOps Starting ===");
    
    try {
        spdlog::info("Creating Game object...");
        std::cout << "Creating Game object..." << std::endl;
        
        Game game;
        
        spdlog::info("Game object created successfully, starting game loop...");
        std::cout << "Game created, starting loop..." << std::endl;
        
        game.run();
        
    } catch (const std::exception& e) {
        std::cout << "FATAL ERROR: " << e.what() << std::endl;
        spdlog::error("Fatal error: {}", e.what());
        std::cout << "Press Enter to exit..." << std::endl;
        std::cin.get();
        return -1;
    }
    
    spdlog::info("=== SnakeAI-MLOps Shutdown Complete ===");
    std::cout << "=== Shutdown Complete ===" << std::endl;
    return 0;
}