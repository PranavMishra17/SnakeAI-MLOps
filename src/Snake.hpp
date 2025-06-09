#pragma once
#include <SFML/Graphics.hpp>
#include <deque>
#include "GameState.hpp"

class Grid;

class Snake {
public:
    Snake(Grid* grid);
    void reset();
    void move();
    void grow();
    void setDirection(Direction dir);
    bool checkSelfCollision() const;
    bool checkWallCollision() const;
    void render(sf::RenderWindow& window);
    
    sf::Vector2i getHeadPosition() const { return m_body.front(); }
    Direction getDirection() const { return m_direction; }
    const std::deque<sf::Vector2i>& getBody() const { return m_body; }
    int getLength() const { return m_body.size(); }
    
    bool isMovingOpposite(Direction newDir) const;
    bool isPositionOnSnake(sf::Vector2i pos) const;
    
private:
    Grid* m_grid;
    std::deque<sf::Vector2i> m_body;
    Direction m_direction;
    Direction m_nextDirection;
    bool m_hasGrown;
    
    sf::RectangleShape m_segmentShape;
};