 
#include "InputManager.hpp"

void InputManager::handleSnakeInput(const sf::Event& event, Snake& snake) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        switch (keyPressed->code) {
            case sf::Keyboard::Key::Up:
            case sf::Keyboard::Key::W:
                snake.setDirection(Direction::UP);
                break;
            case sf::Keyboard::Key::Down:
            case sf::Keyboard::Key::S:
                snake.setDirection(Direction::DOWN);
                break;
            case sf::Keyboard::Key::Left:
            case sf::Keyboard::Key::A:
                snake.setDirection(Direction::LEFT);
                break;
            case sf::Keyboard::Key::Right:
            case sf::Keyboard::Key::D:
                snake.setDirection(Direction::RIGHT);
                break;
        }
    }
}