 
#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <functional>
#include "GameState.hpp"

struct MenuItem {
    std::string text;
    GameMode mode;
    sf::Text displayText;
    bool selected;
};

class Menu {
public:
    Menu();
    void initialize(sf::RenderWindow& window);
    void handleEvent(const sf::Event& event);
    void update();
    void render(sf::RenderWindow& window);
    
    void setSelectionCallback(std::function<void(GameMode)> callback);
    
private:
    std::vector<MenuItem> m_items;
    int m_selectedIndex;
    sf::Font m_font;
    sf::Text m_title;
    sf::Text m_instructions;
    sf::RectangleShape m_background;
    
    std::function<void(GameMode)> m_selectionCallback;
    
    void updateSelection();
};