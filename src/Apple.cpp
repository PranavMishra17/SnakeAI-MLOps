#include "Apple.hpp"
#include "Grid.hpp"
#include "Snake.hpp"

Apple::Apple(Grid* grid) 
    : m_grid(grid)
    , m_active(true)
    , m_rng(std::random_device{}()) {
    
    float cellSize = m_grid->getCellSize();
    m_shape.setRadius(cellSize / 2.0f * 0.8f);
    m_shape.setFillColor(sf::Color(255, 99, 71)); // Tomato red - bright and light
    m_shape.setOutlineThickness(2.0f);
    m_shape.setOutlineColor(sf::Color(220, 20, 60)); // Crimson outline
    m_shape.setOrigin(sf::Vector2f(m_shape.getRadius(), m_shape.getRadius()));
}

void Apple::reset() {
    m_position = sf::Vector2i(m_grid->getSize() / 4, m_grid->getSize() / 4);
    m_active = true;
}

void Apple::respawn(const Snake& snake) {
    std::uniform_int_distribution<int> dist(0, m_grid->getSize() - 1);
    
    do {
        m_position = sf::Vector2i(dist(m_rng), dist(m_rng));
    } while (snake.isPositionOnSnake(m_position));
    
    m_active = true;
}

void Apple::setPosition(sf::Vector2i pos) {
    m_position = pos;
}

void Apple::render(sf::RenderWindow& window) {
    if (m_active) {
        auto screenPos = m_grid->gridToScreen(m_position);
        m_shape.setPosition(sf::Vector2f(screenPos.x + m_grid->getCellSize() / 2,
                                        screenPos.y + m_grid->getCellSize() / 2));
        window.draw(m_shape);
    }
}