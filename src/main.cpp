#include <SFML/Graphics.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <iostream>

int main() {
    spdlog::info("SnakeAI-MLOps starting...");
    
    sf::RenderWindow window(sf::VideoMode({800u, 600u}), "SnakeAI-MLOps");
    
    // Load texture from file
    sf::Texture texture;
    if (!texture.loadFromFile("assets/test.png")) {
        spdlog::error("Failed to load assets/test.png");
        return -1;
    }
    
    // Create sprite from texture
    sf::Sprite sprite(texture);
    
    // Center the sprite
    sf::Vector2u textureSize = texture.getSize();
    sprite.setPosition(
        sf::Vector2f(
            (800.f - static_cast<float>(textureSize.x)) / 2.0f,
            (600.f - static_cast<float>(textureSize.y)) / 2.0f
        )
    );
    
    spdlog::info("Image loaded successfully: {}x{}", textureSize.x, textureSize.y);
    
    while (window.isOpen()) {
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
        }
        
        window.clear(sf::Color::Black);
        window.draw(sprite);
        window.display();
    }
    
    spdlog::info("SnakeAI-MLOps shutting down...");
    return 0;
}