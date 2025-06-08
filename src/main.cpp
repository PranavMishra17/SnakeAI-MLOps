#include <SFML/Graphics.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

int main() {
    spdlog::info("SnakeAI-MLOps starting...");
    
    sf::RenderWindow window(sf::VideoMode(800, 600), "SnakeAI-MLOps");
    
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        
        window.clear(sf::Color::Black);
        // Game rendering will go here
        window.display();
    }
    
    spdlog::info("SnakeAI-MLOps shutting down...");
    return 0;
}