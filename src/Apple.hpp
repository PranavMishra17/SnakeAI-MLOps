 
#pragma once
#include <SFML/Graphics.hpp>
#include <random>

class Grid;
class Snake;

class Apple {
public:
    Apple(Grid* grid);
    void reset();
    void respawn(const Snake& snake);
    void setPosition(sf::Vector2i pos);
    void render(sf::RenderWindow& window);
    
    sf::Vector2i getPosition() const { return m_position; }
    bool isActive() const { return m_active; }
    void setActive(bool active) { m_active = active; }
    
private:
    Grid* m_grid;
    sf::Vector2i m_position;
    bool m_active;
    
    sf::CircleShape m_shape;
    std::mt19937 m_rng;
};