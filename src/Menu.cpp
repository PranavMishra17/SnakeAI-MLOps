#include "Menu.hpp"
#include <spdlog/spdlog.h>

Menu::Menu() : m_selectedIndex(0), m_currentSection(MenuSection::MAIN) {}

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
    m_title->setPosition(sf::Vector2f(windowSize.x / 2.0f - 200.0f, 80.0f));
    
    // Section title
    m_sectionTitle = std::make_unique<sf::Text>(m_font);
    m_sectionTitle->setCharacterSize(28);
    m_sectionTitle->setFillColor(sf::Color::Cyan);
    m_sectionTitle->setPosition(sf::Vector2f(windowSize.x / 2.0f - 100.0f, 180.0f));
    
    // Instructions
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setCharacterSize(20);
    m_instructions->setFillColor(sf::Color(150, 150, 150));
    m_instructions->setPosition(sf::Vector2f(50.0f, windowSize.y - 80.0f));
    
    setupMainMenu();
    setupPlayModeMenu();
    updateSelection();
}

void Menu::setupMainMenu() {
    m_mainItems.clear();
    m_mainItems.emplace_back("🎮 Play Game", GameMode::SINGLE_PLAYER, "", false);
    m_mainItems.emplace_back("⚙️ Settings", GameMode::SINGLE_PLAYER, "", false);
    m_mainItems.emplace_back("❓ How to Play", GameMode::SINGLE_PLAYER, "", false);
    m_mainItems.emplace_back("🏆 Leaderboard", GameMode::SINGLE_PLAYER, "", false);
    
    float startY = 250.0f;
    for (size_t i = 0; i < m_mainItems.size(); ++i) {
        m_mainItems[i].displayText = std::make_unique<sf::Text>(m_font);
        m_mainItems[i].displayText->setString(m_mainItems[i].text);
        m_mainItems[i].displayText->setCharacterSize(32);
        m_mainItems[i].displayText->setPosition(sf::Vector2f(400.0f, startY + i * 80.0f));
    }
}

void Menu::setupPlayModeMenu() {
    m_playModeItems.clear();
    m_playModeItems.emplace_back("Single Player", GameMode::SINGLE_PLAYER, 
        "Classic Snake: You control the snake with arrow keys.\nEat apples to grow and achieve the highest score!");
    m_playModeItems.emplace_back("Agent vs Player", GameMode::AGENT_VS_PLAYER,
        "AI Snake: AI controls the snake, you place apples.\nTest AI behavior by strategically placing food!");
    m_playModeItems.emplace_back("Agent vs System", GameMode::AGENT_VS_SYSTEM,
        "Pure AI Training: AI controls snake, system spawns apples.\nWatch the AI learn and improve autonomously!");
    
    float startY = 250.0f;
    for (size_t i = 0; i < m_playModeItems.size(); ++i) {
        auto& item = m_playModeItems[i];
        
        // Main text
        item.displayText = std::make_unique<sf::Text>(m_font);
        item.displayText->setString(item.text);
        item.displayText->setCharacterSize(28);
        item.displayText->setPosition(sf::Vector2f(200.0f, startY + i * 120.0f));
        
        // Explanation box
        item.explanationBox.setSize(sf::Vector2f(600.0f, 80.0f));
        item.explanationBox.setPosition(sf::Vector2f(180.0f, startY + i * 120.0f + 35.0f));
        item.explanationBox.setFillColor(sf::Color(40, 40, 60, 200));
        item.explanationBox.setOutlineThickness(1.0f);
        item.explanationBox.setOutlineColor(sf::Color(100, 100, 150));
        
        // Explanation text
        item.explanationText = std::make_unique<sf::Text>(m_font);
        item.explanationText->setString(item.explanation);
        item.explanationText->setCharacterSize(16);
        item.explanationText->setFillColor(sf::Color(200, 200, 220));
        item.explanationText->setPosition(sf::Vector2f(190.0f, startY + i * 120.0f + 45.0f));
    }
}

void Menu::handleEvent(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        switch (keyPressed->code) {
            case sf::Keyboard::Key::Up:
                if (m_currentSection == MenuSection::MAIN) {
                    m_selectedIndex = (m_selectedIndex - 1 + m_mainItems.size()) % m_mainItems.size();
                } else {
                    m_selectedIndex = (m_selectedIndex - 1 + m_playModeItems.size()) % m_playModeItems.size();
                }
                updateSelection();
                break;
            case sf::Keyboard::Key::Down:
                if (m_currentSection == MenuSection::MAIN) {
                    m_selectedIndex = (m_selectedIndex + 1) % m_mainItems.size();
                } else {
                    m_selectedIndex = (m_selectedIndex + 1) % m_playModeItems.size();
                }
                updateSelection();
                break;
            case sf::Keyboard::Key::Enter:
                if (m_currentSection == MenuSection::MAIN) {
                    if (m_selectedIndex == 0) { // Play Game
                        m_currentSection = MenuSection::PLAY_MODES;
                        m_selectedIndex = 0;
                        updateSelection();
                    } else if (m_selectedIndex == 1 && m_settingsCallback) { // Settings
                        m_settingsCallback();
                    } else if (m_selectedIndex == 2 && m_howToPlayCallback) { // How to Play
                        m_howToPlayCallback();
                    } else if (m_selectedIndex == 3) { // Leaderboard - handled by game
                        // This will be handled by the global key handler
                    }
                } else if (m_currentSection == MenuSection::PLAY_MODES) {
                    if (m_selectionCallback) {
                        m_selectionCallback(m_playModeItems[m_selectedIndex].mode);
                    }
                }
                break;
            case sf::Keyboard::Key::Escape:
                if (m_currentSection == MenuSection::PLAY_MODES) {
                    m_currentSection = MenuSection::MAIN;
                    m_selectedIndex = 0;
                    updateSelection();
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
    
    if (m_currentSection == MenuSection::MAIN) {
        renderMainMenu(window);
    } else {
        renderPlayModeMenu(window);
    }
}

void Menu::renderMainMenu(sf::RenderWindow& window) {
    m_sectionTitle->setString("Main Menu");
    window.draw(*m_sectionTitle);
    
    for (const auto& item : m_mainItems) {
        if (item.displayText) window.draw(*item.displayText);
    }
    
    m_instructions->setString("UP/DOWN: Navigate | ENTER: Select | F1: Leaderboard");
    window.draw(*m_instructions);
}

void Menu::renderPlayModeMenu(sf::RenderWindow& window) {
    m_sectionTitle->setString("Select Game Mode");
    window.draw(*m_sectionTitle);
    
    for (const auto& item : m_playModeItems) {
        window.draw(item.explanationBox);
        if (item.displayText) window.draw(*item.displayText);
        if (item.explanationText) window.draw(*item.explanationText);
    }
    
    m_instructions->setString("UP/DOWN: Navigate | ENTER: Start Game | ESC: Back to Main Menu");
    window.draw(*m_instructions);
}

void Menu::setSelectionCallback(std::function<void(GameMode)> callback) {
    m_selectionCallback = callback;
}

void Menu::setSettingsCallback(std::function<void()> callback) {
    m_settingsCallback = callback;
}

void Menu::setHowToPlayCallback(std::function<void()> callback) {
    m_howToPlayCallback = callback;
}

void Menu::updateSelection() {
    if (m_currentSection == MenuSection::MAIN) {
        for (size_t i = 0; i < m_mainItems.size(); ++i) {
            if (m_mainItems[i].displayText) {
                if (i == m_selectedIndex) {
                    m_mainItems[i].displayText->setFillColor(sf::Color::Green);
                    m_mainItems[i].displayText->setStyle(sf::Text::Bold);
                } else {
                    m_mainItems[i].displayText->setFillColor(sf::Color::White);
                    m_mainItems[i].displayText->setStyle(sf::Text::Regular);
                }
            }
        }
    } else {
        for (size_t i = 0; i < m_playModeItems.size(); ++i) {
            if (m_playModeItems[i].displayText) {
                if (i == m_selectedIndex) {
                    m_playModeItems[i].displayText->setFillColor(sf::Color::Green);
                    m_playModeItems[i].displayText->setStyle(sf::Text::Bold);
                    m_playModeItems[i].explanationBox.setOutlineColor(sf::Color::Green);
                } else {
                    m_playModeItems[i].displayText->setFillColor(sf::Color::White);
                    m_playModeItems[i].displayText->setStyle(sf::Text::Regular);
                    m_playModeItems[i].explanationBox.setOutlineColor(sf::Color(100, 100, 150));
                }
            }
        }
    }
}