#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <functional>
#include "GameState.hpp"

struct MenuItem {
    std::string text;
    GameMode mode;
    std::unique_ptr<sf::Text> displayText;
    bool selected;
    
    MenuItem(const std::string& t, GameMode m) : text(t), mode(m), selected(false) {}
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
    std::unique_ptr<sf::Text> m_title;
    std::unique_ptr<sf::Text> m_instructions;
    sf::RectangleShape m_background;
    
    std::function<void(GameMode)> m_selectionCallback;
    
    void updateSelection();
};