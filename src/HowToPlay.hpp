
// HowToPlay.hpp
#pragma once
#include <SFML/Graphics.hpp>
#include <functional>
#include <memory>

class HowToPlay {
public:
    HowToPlay();
    void initialize(sf::RenderWindow& window);
    void handleEvent(const sf::Event& event);
    void update();
    void render(sf::RenderWindow& window);
    
    void setBackCallback(std::function<void()> callback) { m_backCallback = callback; }

private:
    sf::Font m_font;
    std::unique_ptr<sf::Text> m_title;
    std::unique_ptr<sf::Text> m_instructions;
    std::vector<std::unique_ptr<sf::Text>> m_helpTexts;
    sf::RectangleShape m_background;
    sf::RectangleShape m_titlePanel;
    sf::RectangleShape m_contentPanel;
    
    std::function<void()> m_backCallback;
    
    void createHelpContent();
};