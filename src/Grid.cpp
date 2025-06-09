#include "Grid.hpp"

Grid::Grid(int gridSize) : m_gridSize(gridSize) {}

void Grid::initialize(sf::RenderWindow& window) {
    sf::Vector2u windowSize = window.getSize();
    
    // Calculate cell size to fit square grid in window
    float minDimension = std::min(static_cast<float>(windowSize.x), static_cast<float>(windowSize.y));
    float padding = 100.0f; // Leave some padding
    m_cellSize = (minDimension - padding) / m_gridSize;
    
    // Center the grid
    float gridTotalSize = m_cellSize * m_gridSize;
    m_gridOrigin.x = (windowSize.x - gridTotalSize) / 2.0f;
    m_gridOrigin.y = (windowSize.y - gridTotalSize) / 2.0f;
    
    // Create background
    m_background.setSize(sf::Vector2f(gridTotalSize, gridTotalSize));
    m_background.setPosition(m_gridOrigin);
    m_background.setFillColor(sf::Color(50, 50, 50));
    m_background.setOutlineThickness(2.0f);
    m_background.setOutlineColor(sf::Color(100, 100, 100));
    
    // Create grid lines
    m_gridLines.clear();
    
    // Vertical lines
    for (int i = 0; i <= m_gridSize; ++i) {
        sf::RectangleShape line(sf::Vector2f(1.0f, gridTotalSize));
        line.setPosition(sf::Vector2f(m_gridOrigin.x + i * m_cellSize, m_gridOrigin.y));
        line.setFillColor(sf::Color(80, 80, 80));
        m_gridLines.push_back(line);
    }
    
    // Horizontal lines
    for (int i = 0; i <= m_gridSize; ++i) {
        sf::RectangleShape line(sf::Vector2f(gridTotalSize, 1.0f));
        line.setPosition(sf::Vector2f(m_gridOrigin.x, m_gridOrigin.y + i * m_cellSize));
        line.setFillColor(sf::Color(80, 80, 80));
        m_gridLines.push_back(line);
    }
}

void Grid::render(sf::RenderWindow& window) {
    window.draw(m_background);
    
    for (const auto& line : m_gridLines) {
        window.draw(line);
    }
}

sf::Vector2i Grid::screenToGrid(sf::Vector2i screenPos) const {
    int x = static_cast<int>((screenPos.x - m_gridOrigin.x) / m_cellSize);
    int y = static_cast<int>((screenPos.y - m_gridOrigin.y) / m_cellSize);
    return sf::Vector2i(x, y);
}

sf::Vector2f Grid::gridToScreen(sf::Vector2i gridPos) const {
    float x = m_gridOrigin.x + gridPos.x * m_cellSize;
    float y = m_gridOrigin.y + gridPos.y * m_cellSize;
    return sf::Vector2f(x, y);
}

bool Grid::isValidPosition(sf::Vector2i pos) const {
    return pos.x >= 0 && pos.x < m_gridSize && 
           pos.y >= 0 && pos.y < m_gridSize;
}