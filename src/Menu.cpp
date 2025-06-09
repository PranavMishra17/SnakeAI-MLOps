#include "Menu.hpp"
#include <spdlog/spdlog.h>

Menu::Menu() : m_selectedIndex(0) {}

void Menu::initialize(sf::RenderWindow& window) {
    if (!m_font.openFromFile("assets/fonts/arial.ttf")) {
        spdlog::error("Failed to load font");
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Background
    m_background.setSize(sf::Vector2f(static_cast<float>(windowSize.x), 
                                     static_cast<float>(windowSize.y)));
    m_background.setFillColor(sf::Color(20, 20, 20));
    
    // Title
    m_title = std::make_unique<sf::Text>(m_font);
    m_title->setString("SnakeAI-MLOps");
    m_title->setCharacterSize(48);
    m_title->setFillColor(sf::Color::White);
    m_title->setPosition(sf::Vector2f(windowSize.x / 2.0f - 200.0f, 100.0f));
    
    // Menu items
    m_items.clear();
    m_items.emplace_back("Single Player", GameMode::SINGLE_PLAYER);
    m_items.emplace_back("Agent vs Player", GameMode::AGENT_VS_PLAYER);
    m_items.emplace_back("Agent vs System", GameMode::AGENT_VS_SYSTEM);
    
    float startY = 300.0f;
    for (size_t i = 0; i < m_items.size(); ++i) {
        m_items[i].displayText = std::make_unique<sf::Text>(m_font);
        m_items[i].displayText->setString(m_items[i].text);
        m_items[i].displayText->setCharacterSize(32);
        m_items[i].displayText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 150.0f, 
                                                        startY + i * 80.0f));
    }
    
    // Instructions
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setString("Use UP/DOWN arrows to select, ENTER to start");
    m_instructions->setCharacterSize(20);
    m_instructions->setFillColor(sf::Color(150, 150, 150));
    m_instructions->setPosition(sf::Vector2f(windowSize.x / 2.0f - 250.0f, 
                                           windowSize.y - 100.0f));
    
    updateSelection();
}

void Menu::handleEvent(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        switch (keyPressed->code) {
            case sf::Keyboard::Key::Up:
                m_selectedIndex = (m_selectedIndex - 1 + m_items.size()) % m_items.size();
                updateSelection();
                break;
            case sf::Keyboard::Key::Down:
                m_selectedIndex = (m_selectedIndex + 1) % m_items.size();
                updateSelection();
                break;
            case sf::Keyboard::Key::Enter:
                if (m_selectionCallback) {
                    m_selectionCallback(m_items[m_selectedIndex].mode);
                }
                break;
        }
    }
}

void Menu::update() {
    // Animation or other updates if needed
}

void Menu::render(sf::RenderWindow& window) {
    window.draw(m_background);
    if (m_title) window.draw(*m_title);
    
    for (const auto& item : m_items) {
        if (item.displayText) window.draw(*item.displayText);
    }
    
    if (m_instructions) window.draw(*m_instructions);
}

void Menu::setSelectionCallback(std::function<void(GameMode)> callback) {
    m_selectionCallback = callback;
}

void Menu::updateSelection() {
    for (size_t i = 0; i < m_items.size(); ++i) {
        if (m_items[i].displayText) {
            if (i == m_selectedIndex) {
                m_items[i].displayText->setFillColor(sf::Color::Green);
                m_items[i].displayText->setStyle(sf::Text::Bold);
            } else {
                m_items[i].displayText->setFillColor(sf::Color::White);
                m_items[i].displayText->setStyle(sf::Text::Regular);
            }
        }
    }
}