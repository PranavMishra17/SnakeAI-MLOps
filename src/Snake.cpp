#include "Snake.hpp"
#include "Grid.hpp"
#include <spdlog/spdlog.h>

Snake::Snake(Grid* grid)
    : m_grid(grid)
    , m_direction(Direction::RIGHT)
    , m_nextDirection(Direction::RIGHT)
    , m_hasGrown(false)
{
    spdlog::info("Snake: Initializing snake...");
    
    float cellSize = m_grid->getCellSize();
    m_segmentShape.setSize(sf::Vector2f(cellSize * 0.9f, cellSize * 0.9f));
    m_segmentShape.setFillColor(sf::Color(144, 238, 144)); // Light green
    
    spdlog::info("Snake: Snake initialized successfully");
    reset();
}

void Snake::reset() {
    spdlog::info("Snake: Resetting snake position");
    m_body.clear();
    int center = m_grid->getSize() / 2;
    m_body.push_back(sf::Vector2i(center, center));
    m_body.push_back(sf::Vector2i(center - 1, center));
    m_direction = Direction::RIGHT;
    m_nextDirection = Direction::RIGHT;
    m_hasGrown = false;
    spdlog::info("Snake: Reset complete, body size: {}", m_body.size());
}

void Snake::move() {
    if (!isMovingOpposite(m_nextDirection)) {
        m_direction = m_nextDirection;
    }
    
    sf::Vector2i newHead = m_body.front();
    switch (m_direction) {
        case Direction::UP:    newHead.y--; break;
        case Direction::DOWN:  newHead.y++; break;
        case Direction::LEFT:  newHead.x--; break;
        case Direction::RIGHT: newHead.x++; break;
    }
    
    m_body.push_front(newHead);
    
    if (!m_hasGrown) {
        m_body.pop_back();
    } else {
        m_hasGrown = false;
    }
}

void Snake::grow() {
    m_hasGrown = true;
    spdlog::info("Snake: Growing! New length will be: {}", m_body.size() + 1);
}

void Snake::setDirection(Direction dir) {
    if (!isMovingOpposite(dir)) {
        m_nextDirection = dir;
    }
}

bool Snake::isMovingOpposite(Direction newDir) const {
    return (m_direction == Direction::UP && newDir == Direction::DOWN) ||
           (m_direction == Direction::DOWN && newDir == Direction::UP) ||
           (m_direction == Direction::LEFT && newDir == Direction::RIGHT) ||
           (m_direction == Direction::RIGHT && newDir == Direction::LEFT);
}

bool Snake::checkSelfCollision() const {
    auto head = m_body.front();
    for (size_t i = 1; i < m_body.size(); ++i) {
        if (m_body[i] == head) {
            spdlog::warn("Snake: Self collision detected!");
            return true;
        }
    }
    return false;
}

bool Snake::checkWallCollision() const {
    auto head = m_body.front();
    bool collision = !m_grid->isValidPosition(head);
    if (collision) {
        spdlog::warn("Snake: Wall collision detected at ({}, {})", head.x, head.y);
    }
    return collision;
}

bool Snake::isPositionOnSnake(sf::Vector2i pos) const {
    for (const auto& segment : m_body) {
        if (segment == pos) {
            return true;
        }
    }
    return false;
}

void Snake::render(sf::RenderWindow& window) {
    for (size_t i = 0; i < m_body.size(); ++i) {
        auto screenPos = m_grid->gridToScreen(m_body[i]);
        
        // Light, bright colors for snake
        if (i == 0) {
            // Head - bright lime green
            m_segmentShape.setFillColor(sf::Color(50, 205, 50)); // Lime green
            m_segmentShape.setOutlineThickness(2.0f);
            m_segmentShape.setOutlineColor(sf::Color(34, 139, 34)); // Forest green outline
        } else {
            // Body - light green
            m_segmentShape.setFillColor(sf::Color(144, 238, 144)); // Light green
            m_segmentShape.setOutlineThickness(1.0f);
            m_segmentShape.setOutlineColor(sf::Color(107, 142, 35)); // Olive drab outline
        }
        
        m_segmentShape.setPosition(screenPos);
        window.draw(m_segmentShape);
    }
}