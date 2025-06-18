#include "Menu.hpp"
#include <spdlog/spdlog.h>

Menu::Menu() : m_selectedIndex(0), m_currentSection(MenuSection::MAIN) {}

void Menu::initialize(sf::RenderWindow& window) {
    // SFML 3 font loading with multiple paths
    bool fontLoaded = false;
    std::vector<std::string> fontPaths = {
        "assets/fonts/ARIAL.TTF",
        "assets/fonts/arial.ttf", 
        "assets/fonts/ArialCE.ttf",
        "assets/fonts/Roboto.ttf"
    };
    
    for (const auto& path : fontPaths) {
        if (m_font.openFromFile(path)) {
            fontLoaded = true;
            spdlog::info("Menu: Font loaded from: {}", path);
            break;
        }
    }
    
    if (!fontLoaded) {
        spdlog::error("Menu: Failed to load any font");
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Light cream background
    m_background.setSize(sf::Vector2f(static_cast<float>(windowSize.x), 
                                     static_cast<float>(windowSize.y)));
    m_background.setFillColor(sf::Color(245, 245, 220)); // Light beige
    
    // Title with dark green color
    m_title = std::make_unique<sf::Text>(m_font);
    m_title->setString("SnakeAI-MLOps");
    m_title->setCharacterSize(48);
    m_title->setFillColor(sf::Color(47, 79, 47)); // Dark green
    m_title->setPosition(sf::Vector2f(windowSize.x / 2.0f - 200.0f, 80.0f));
    
    // Section title with light blue
    m_sectionTitle = std::make_unique<sf::Text>(m_font);
    m_sectionTitle->setCharacterSize(28);
    m_sectionTitle->setFillColor(sf::Color(70, 130, 180)); // Steel blue
    m_sectionTitle->setPosition(sf::Vector2f(windowSize.x / 2.0f - 100.0f, 180.0f));
    
    // Instructions with dark green
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setCharacterSize(20);
    m_instructions->setFillColor(sf::Color(47, 79, 47)); // Dark green
    m_instructions->setPosition(sf::Vector2f(50.0f, windowSize.y - 80.0f));
    
    setupMainMenu();
    setupPlayModeMenu();
    updateSelection();
}

void Menu::setupMainMenu() {
    m_mainItems.clear();
    m_mainItems.emplace_back("Play Game", GameMode::SINGLE_PLAYER, "", false, false);
    m_mainItems.emplace_back("Settings", GameMode::SINGLE_PLAYER, "", false, true);
    m_mainItems.emplace_back("How to Play", GameMode::SINGLE_PLAYER, "", false, true);
    m_mainItems.emplace_back("Leaderboard", GameMode::SINGLE_PLAYER, "", false, true);
    m_mainItems.emplace_back("Quit Game", GameMode::SINGLE_PLAYER, "", false, true);
    
    float startY = 250.0f;
    for (size_t i = 0; i < m_mainItems.size(); ++i) {
        auto& item = m_mainItems[i];
        
        // Button background
        item.buttonBackground.setSize(sf::Vector2f(300.0f, 60.0f));
        item.buttonBackground.setPosition(sf::Vector2f(350.0f, startY + i * 80.0f));
        item.buttonBackground.setFillColor(sf::Color(144, 238, 144)); // Light green
        item.buttonBackground.setOutlineThickness(3.0f);
        item.buttonBackground.setOutlineColor(sf::Color(34, 139, 34)); // Forest green
        
        // Button text
        item.displayText = std::make_unique<sf::Text>(m_font);
        item.displayText->setString(item.text);
        item.displayText->setCharacterSize(28);
        item.displayText->setFillColor(sf::Color(47, 79, 47)); // Dark green
        item.displayText->setPosition(sf::Vector2f(380.0f, startY + i * 80.0f + 15.0f));
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
        
        // Button background
        item.buttonBackground.setSize(sf::Vector2f(500.0f, 50.0f));
        item.buttonBackground.setPosition(sf::Vector2f(150.0f, startY + i * 120.0f));
        item.buttonBackground.setFillColor(sf::Color(255, 255, 224)); // Light yellow
        item.buttonBackground.setOutlineThickness(2.0f);
        item.buttonBackground.setOutlineColor(sf::Color(218, 165, 32)); // Goldenrod
        
        // Main text
        item.displayText = std::make_unique<sf::Text>(m_font);
        item.displayText->setString(item.text);
        item.displayText->setCharacterSize(26);
        item.displayText->setFillColor(sf::Color(47, 79, 47)); // Dark green
        item.displayText->setPosition(sf::Vector2f(170.0f, startY + i * 120.0f + 12.0f));
        
        // Explanation box
        item.explanationBox.setSize(sf::Vector2f(500.0f, 60.0f));
        item.explanationBox.setPosition(sf::Vector2f(150.0f, startY + i * 120.0f + 55.0f));
        item.explanationBox.setFillColor(sf::Color(240, 248, 255)); // Alice blue
        item.explanationBox.setOutlineThickness(1.0f);
        item.explanationBox.setOutlineColor(sf::Color(173, 216, 230)); // Light blue
        
        // Explanation text
        item.explanationText = std::make_unique<sf::Text>(m_font);
        item.explanationText->setString(item.explanation);
        item.explanationText->setCharacterSize(16);
        item.explanationText->setFillColor(sf::Color(25, 25, 112)); // Midnight blue
        item.explanationText->setPosition(sf::Vector2f(160.0f, startY + i * 120.0f + 65.0f));
    }
}

void Menu::handleEvent(const sf::Event& event) {
    if (const auto* keyPressedEvent = event.getIf<sf::Event::KeyPressed>()) {
        switch (keyPressedEvent->scancode) {
            case sf::Keyboard::Scancode::Up:
                if (m_currentSection == MenuSection::MAIN) {
                    m_selectedIndex = (m_selectedIndex - 1 + m_mainItems.size()) % m_mainItems.size();
                } else {
                    m_selectedIndex = (m_selectedIndex - 1 + m_playModeItems.size()) % m_playModeItems.size();
                }
                updateSelection();
                break;
            case sf::Keyboard::Scancode::Down:
                if (m_currentSection == MenuSection::MAIN) {
                    m_selectedIndex = (m_selectedIndex + 1) % m_mainItems.size();
                } else {
                    m_selectedIndex = (m_selectedIndex + 1) % m_playModeItems.size();
                }
                updateSelection();
                break;
            case sf::Keyboard::Scancode::Enter:
                if (m_currentSection == MenuSection::MAIN) {
                    if (m_selectedIndex == 0) { // Play Game
                        m_currentSection = MenuSection::PLAY_MODES;
                        m_selectedIndex = 0;
                        updateSelection();
                    } else if (m_selectedIndex == 1 && m_settingsCallback) { // Settings
                        m_settingsCallback();
                    } else if (m_selectedIndex == 2 && m_howToPlayCallback) { // How to Play
                        m_howToPlayCallback();
                    } else if (m_selectedIndex == 3 && m_leaderboardCallback) { // Leaderboard
                        m_leaderboardCallback();
                    } else if (m_selectedIndex == 4 && m_quitCallback) { // Quit
                        m_quitCallback();
                    }
                } else if (m_currentSection == MenuSection::PLAY_MODES) {
                    if (m_selectionCallback) {
                        m_selectionCallback(m_playModeItems[m_selectedIndex].mode);
                    }
                }
                break;
            case sf::Keyboard::Scancode::Escape:
                if (m_currentSection == MenuSection::PLAY_MODES) {
                    m_currentSection = MenuSection::MAIN;
                    m_selectedIndex = 0;
                    updateSelection();
                }
                break;
            default:
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
        window.draw(item.buttonBackground);
        if (item.displayText) window.draw(*item.displayText);
    }
    
    m_instructions->setString("UP/DOWN: Navigate | ENTER: Select | F1: Leaderboard");
    window.draw(*m_instructions);
}

void Menu::renderPlayModeMenu(sf::RenderWindow& window) {
    m_sectionTitle->setString("Select Game Mode");
    window.draw(*m_sectionTitle);
    
    for (const auto& item : m_playModeItems) {
        window.draw(item.buttonBackground);
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

void Menu::setLeaderboardCallback(std::function<void()> callback) {
    m_leaderboardCallback = callback;
}

void Menu::setQuitCallback(std::function<void()> callback) {
    m_quitCallback = callback;
}

void Menu::updateSelection() {
    if (m_currentSection == MenuSection::MAIN) {
        for (size_t i = 0; i < m_mainItems.size(); ++i) {
            if (i == m_selectedIndex) {
                // Highlighted selection - bright yellow
                m_mainItems[i].buttonBackground.setFillColor(sf::Color(255, 255, 0)); // Bright yellow
                m_mainItems[i].buttonBackground.setOutlineColor(sf::Color(255, 140, 0)); // Dark orange
                m_mainItems[i].buttonBackground.setOutlineThickness(4.0f);
                
                if (m_mainItems[i].displayText) {
                    m_mainItems[i].displayText->setFillColor(sf::Color(139, 69, 19)); // Saddle brown
                    m_mainItems[i].displayText->setStyle(sf::Text::Bold);
                }
            } else {
                // Normal appearance - light green
                m_mainItems[i].buttonBackground.setFillColor(sf::Color(144, 238, 144)); // Light green
                m_mainItems[i].buttonBackground.setOutlineColor(sf::Color(34, 139, 34)); // Forest green
                m_mainItems[i].buttonBackground.setOutlineThickness(3.0f);
                
                if (m_mainItems[i].displayText) {
                    m_mainItems[i].displayText->setFillColor(sf::Color(47, 79, 47)); // Dark green
                    m_mainItems[i].displayText->setStyle(sf::Text::Regular);
                }
            }
        }
    } else {
        for (size_t i = 0; i < m_playModeItems.size(); ++i) {
            if (i == m_selectedIndex) {
                // Highlighted selection - bright yellow
                m_playModeItems[i].buttonBackground.setFillColor(sf::Color(255, 255, 0)); // Bright yellow
                m_playModeItems[i].buttonBackground.setOutlineColor(sf::Color(255, 140, 0)); // Dark orange
                m_playModeItems[i].buttonBackground.setOutlineThickness(3.0f);
                m_playModeItems[i].explanationBox.setOutlineColor(sf::Color(255, 140, 0)); // Dark orange
                
                if (m_playModeItems[i].displayText) {
                    m_playModeItems[i].displayText->setFillColor(sf::Color(139, 69, 19)); // Saddle brown
                    m_playModeItems[i].displayText->setStyle(sf::Text::Bold);
                }
            } else {
                // Normal appearance - light yellow
                m_playModeItems[i].buttonBackground.setFillColor(sf::Color(255, 255, 224)); // Light yellow
                m_playModeItems[i].buttonBackground.setOutlineColor(sf::Color(218, 165, 32)); // Goldenrod
                m_playModeItems[i].buttonBackground.setOutlineThickness(2.0f);
                m_playModeItems[i].explanationBox.setOutlineColor(sf::Color(173, 216, 230)); // Light blue
                
                if (m_playModeItems[i].displayText) {
                    m_playModeItems[i].displayText->setFillColor(sf::Color(47, 79, 47)); // Dark green
                    m_playModeItems[i].displayText->setStyle(sf::Text::Regular);
                }
            }
        }
    }
}