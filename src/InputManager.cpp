#include "InputManager.hpp"
#include <SFML/Window/Event.hpp> // Explicitly include Event for clarity
#include "Snake.hpp" // Assuming Snake.hpp defines Snake class and Direction enum

void InputManager::handleSnakeInput(const sf::Event& event, Snake& snake) {
    // In SFML 3, you use event.getIf<T>() to check the type and access data.
    if (const auto* keyPressedEvent = event.getIf<sf::Event::KeyPressed>()) {
        // Access the scancode from the keyPressedEvent pointer
        switch (keyPressedEvent->scancode) { // Use scancode for SFML 3 keyboard events
            case sf::Keyboard::Scancode::Up:    // Use scoped enums
            case sf::Keyboard::Scancode::W:
                snake.setDirection(Direction::UP);
                break;
            case sf::Keyboard::Scancode::Down:  // Use scoped enums
            case sf::Keyboard::Scancode::S:
                snake.setDirection(Direction::DOWN);
                break;
            case sf::Keyboard::Scancode::Left:  // Use scoped enums
            case sf::Keyboard::Scancode::A:
                snake.setDirection(Direction::LEFT);
                break;
            case sf::Keyboard::Scancode::Right: // Use scoped enums
            case sf::Keyboard::Scancode::D:
                snake.setDirection(Direction::RIGHT);
                break;
            default:
                break; // Good practice to have a default in switch statements
        }
    }
}