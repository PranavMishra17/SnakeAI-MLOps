#pragma once
#include <SFML/Graphics.hpp>
#include <vector>

class Grid {
public:
    Grid(int gridSize);
    void initialize(sf::RenderWindow& window);
    void render(sf::RenderWindow& window);
    
    sf::Vector2i screenToGrid(sf::Vector2i screenPos) const;
    sf::Vector2f gridToScreen(sf::Vector2i gridPos) const;
    
    int getSize() const { return m_gridSize; }
    float getCellSize() const { return m_cellSize; }
    sf::Vector2f getGridOrigin() const { return m_gridOrigin; }
    
    bool isValidPosition(sf::Vector2i pos) const;
    
private:
    int m_gridSize;          // Number of cells per side
    float m_cellSize;        // Size of each cell in pixels
    sf::Vector2f m_gridOrigin; // Top-left corner of grid
    
    std::vector<sf::RectangleShape> m_gridLines;
    sf::RectangleShape m_background;
};