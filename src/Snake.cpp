#include "Snake.hpp"
#include "Grid.hpp"
#include <spdlog/spdlog.h>

Snake::Snake(Grid* grid)
    : m_grid(grid)
    , m_direction(Direction::RIGHT)
    , m_nextDirection(Direction::RIGHT)
    , m_hasGrown(false)
{
    // Load textures
    if (!m_headTexture.loadFromFile("assets/head.jpg")) {
        spdlog::error("Failed to load head.jpg");
    }
    if (!m_bodyTexture.loadFromFile("assets/skin.jpg")) {
        spdlog::error("Failed to load skin.jpg");
    }

    // Initialize sprites after textures are loaded
    m_headSprite.setTexture(m_headTexture);
    m_bodySprite.setTexture(m_bodyTexture);

    float cellSize = m_grid->getCellSize();
    m_headSprite.setScale(sf::Vector2f(cellSize / m_headTexture.getSize().x,
                                       cellSize / m_headTexture.getSize().y));
    m_bodySprite.setScale(sf::Vector2f(cellSize / m_bodyTexture.getSize().x,
                                       cellSize / m_bodyTexture.getSize().y));

    reset();
}


void Snake::reset() {
    m_body.clear();
    int center = m_grid->getSize() / 2;
    m_body.push_back(sf::Vector2i(center, center));
    m_body.push_back(sf::Vector2i(center - 1, center));
    m_direction = Direction::RIGHT;
    m_nextDirection = Direction::RIGHT;
    m_hasGrown = false;
    
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
            return true;
        }
    }
    return false;
}

bool Snake::checkWallCollision() const {
    auto head = m_body.front();
    return !m_grid->isValidPosition(head);
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
        
        if (i == 0) {
            m_headSprite.setPosition(screenPos);
            
            // Rotate head based on direction
            float rotation = 0.0f;
            switch (m_direction) {
                case Direction::UP:    rotation = -90.0f; break;
                case Direction::DOWN:  rotation = 90.0f; break;
                case Direction::LEFT:  rotation = 180.0f; break;
                case Direction::RIGHT: rotation = 0.0f; break;
            }
            m_headSprite.setRotation(sf::degrees(rotation));
            m_headSprite.setOrigin(sf::Vector2f(m_headTexture.getSize().x / 2.0f,
                                               m_headTexture.getSize().y / 2.0f));
            m_headSprite.move(sf::Vector2f(m_grid->getCellSize() / 2, 
                                          m_grid->getCellSize() / 2));
            
            window.draw(m_headSprite);
        } else {
            m_bodySprite.setPosition(screenPos);
            window.draw(m_bodySprite);
        }
    }
}