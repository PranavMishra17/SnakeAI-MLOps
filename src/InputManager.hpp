 #pragma once
#include <SFML/Graphics.hpp>
#include "Snake.hpp"

class InputManager {
public:
    void handleSnakeInput(const sf::Event& event, Snake& snake);
};