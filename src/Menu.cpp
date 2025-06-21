#include "Menu.hpp"
#include <spdlog/spdlog.h>

// Default constructor
Menu::Menu()
    : m_selectedIndex(0),
      m_currentSection(MenuSection::MAIN),
      m_backgroundImageSprite(m_backgroundImageTexture), // Initialize with texture
      m_hasBackgroundImage(false)
{
}

void Menu::initialize(sf::RenderWindow& window) {
    // SFML 3 font loading with multiple paths
    std::vector<std::string> fontPaths = {
        "assets/fonts/ARIAL.TTF", "assets/fonts/arial.ttf", 
        "assets/fonts/ArialCE.ttf", "assets/fonts/Roboto.ttf"
    };
    
    for (const auto& path : fontPaths) {
        if (m_font.openFromFile(path)) {
            spdlog::info("Menu: Font loaded from: {}", path);
            break;
        }
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Load and setup background image
    loadBackgroundImage("assets/menu.png");
    setupBackgroundImage(window);
    
    // Enhanced background with gradient
    m_background.setSize(sf::Vector2f(windowSize.x, windowSize.y));
    m_background.setFillColor(sf::Color(245, 245, 220, 240)); // Semi-transparent beige
    
    setupEnhancedLayout(window);
    setupMainMenu();
    setupPlayModeMenu();
    updateSelection();
}

bool Menu::loadBackgroundImage(const std::string& imagePath) {
    if (m_backgroundImageTexture.loadFromFile(imagePath)) {
        // Create sprite with loaded texture (like ImageViewer)
        m_backgroundImageSprite = sf::Sprite(m_backgroundImageTexture);
        m_hasBackgroundImage = true;
        spdlog::info("Menu: Background image loaded: {}", imagePath);
        return true;
    } else {
        spdlog::warn("Menu: Failed to load background image: {}", imagePath);
        m_hasBackgroundImage = false;
        return false;
    }
}

void Menu::setupBackgroundImage(sf::RenderWindow& window) {
    if (!m_hasBackgroundImage) return;
    
    sf::Vector2u windowSize = window.getSize();
    sf::Vector2u imageSize = m_backgroundImageTexture.getSize();
    
    // Scale to fit nicely on right side
    float scaleX = static_cast<float>(windowSize.x) * 0.4f / imageSize.x; 
    float scaleY = static_cast<float>(windowSize.y) * 0.8f / imageSize.y; 
    float scale = std::min(scaleX, scaleY);
    
    m_backgroundImageSprite.setScale(sf::Vector2f(scale, scale));
    
    // Position on right side
    float scaledWidth = imageSize.x * scale;
    float scaledHeight = imageSize.y * scale;
    float posX = windowSize.x - scaledWidth - 50.0f;
    float posY = (windowSize.y - scaledHeight) / 2.0f;
    
    m_backgroundImageSprite.setPosition(sf::Vector2f(posX, posY));
    
    // Less transparency - make image more visible
    m_backgroundImageSprite.setColor(sf::Color(255, 255, 255, 220));
    
    spdlog::info("Menu: Background image positioned at ({}, {}) with scale {}", posX, posY, scale);
}

void Menu::setupEnhancedLayout(sf::RenderWindow& window) {
    sf::Vector2u windowSize = window.getSize();
    
    // Enhanced background with light yellow theme
    m_background.setSize(sf::Vector2f(windowSize.x, windowSize.y));
    m_background.setFillColor(sf::Color(255, 253, 208)); // Light yellow
    
    // Main title - larger and more prominent
    m_title = std::make_unique<sf::Text>(m_font);
    m_title->setString("SNAKEAI MLOPS");
    m_title->setCharacterSize(64);
    m_title->setFillColor(sf::Color(139, 69, 19)); // Saddle brown for yellow theme
    m_title->setStyle(sf::Text::Bold);
    
    // Center title horizontally
    auto titleBounds = m_title->getLocalBounds();
    m_title->setPosition(sf::Vector2f((windowSize.x - titleBounds.size.x) / 2.0f, 40.0f));
    
    // Enhanced subtitle
    m_versionText = std::make_unique<sf::Text>(m_font);
    m_versionText->setString("Advanced Reinforcement Learning Game");
    m_versionText->setCharacterSize(20);
    m_versionText->setFillColor(sf::Color(160, 82, 45)); // Saddle brown lighter
    m_versionText->setStyle(sf::Text::Italic);
    
    // Center subtitle
    auto subtitleBounds = m_versionText->getLocalBounds();
    m_versionText->setPosition(sf::Vector2f((windowSize.x - subtitleBounds.size.x) / 2.0f, 110.0f));
    
    // Version number in bottom right
    m_versionNumber = std::make_unique<sf::Text>(m_font);
    m_versionNumber->setString("v3.1");
    m_versionNumber->setCharacterSize(16);
    m_versionNumber->setFillColor(sf::Color(139, 69, 19, 150)); // Faded brown
    m_versionNumber->setPosition(sf::Vector2f(windowSize.x - 80.0f, windowSize.y - 40.0f));
    
    // Instructions with better positioning
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setCharacterSize(18);
    m_instructions->setFillColor(sf::Color(101, 67, 33)); // Dark brown
    m_instructions->setPosition(sf::Vector2f(50.0f, windowSize.y - 60.0f));
    
    createVisualPanels(window);
}

void Menu::createVisualPanels(sf::RenderWindow& window) {
    sf::Vector2u windowSize = window.getSize();
    
    // Title panel - yellow theme
    m_titlePanel.setSize(sf::Vector2f(windowSize.x - 100.0f, 120.0f));
    m_titlePanel.setPosition(sf::Vector2f(50.0f, 20.0f));
    m_titlePanel.setFillColor(sf::Color(255, 248, 220, 150)); // Light cornsilk
    m_titlePanel.setOutlineThickness(2.0f);
    m_titlePanel.setOutlineColor(sf::Color(218, 165, 32, 100)); // Goldenrod
    
    // Content panel for menu items - smaller width
    float panelWidth = m_hasBackgroundImage ? windowSize.x * 0.45f : windowSize.x * 0.6f;
    m_contentPanel.setSize(sf::Vector2f(panelWidth, windowSize.y * 0.55f));
    m_contentPanel.setPosition(sf::Vector2f(50.0f, 160.0f));
    m_contentPanel.setFillColor(sf::Color(255, 250, 205, 180)); // Light lemon chiffon
    m_contentPanel.setOutlineThickness(2.0f);
    m_contentPanel.setOutlineColor(sf::Color(218, 165, 32, 150));
}

void Menu::setupMainMenu() {
    m_mainItems.clear();
    m_mainItems.emplace_back("Play Game", GameMode::SINGLE_PLAYER, "", false, false);
    m_mainItems.emplace_back("Settings", GameMode::SINGLE_PLAYER, "", false, true);
    m_mainItems.emplace_back("How to Play", GameMode::SINGLE_PLAYER, "", false, true);
    m_mainItems.emplace_back("Stats & Analysis", GameMode::SINGLE_PLAYER, "", false, true);
    m_mainItems.emplace_back("Leaderboard", GameMode::SINGLE_PLAYER, "", false, true);
    m_mainItems.emplace_back("Quit Game", GameMode::SINGLE_PLAYER, "", false, true);
    
    float startY = 200.0f;
    float buttonWidth = m_hasBackgroundImage ? 350.0f : 400.0f; // Smaller width
    float buttonHeight = 50.0f; // Smaller height
    float spacing = 65.0f; // Tighter spacing
    
    for (size_t i = 0; i < m_mainItems.size(); ++i) {
        auto& item = m_mainItems[i];
        
        // Smaller button styling
        item.buttonBackground.setSize(sf::Vector2f(buttonWidth, buttonHeight));
        item.buttonBackground.setPosition(sf::Vector2f(80.0f, startY + i * spacing));
        item.buttonBackground.setFillColor(sf::Color(255, 248, 220, 240)); // Light cornsilk
        item.buttonBackground.setOutlineThickness(2.0f);
        item.buttonBackground.setOutlineColor(sf::Color(218, 165, 32)); // Goldenrod
        
        // Button text
        item.displayText = std::make_unique<sf::Text>(m_font);
        item.displayText->setString(item.text);
        item.displayText->setCharacterSize(22);
        item.displayText->setFillColor(sf::Color(139, 69, 19)); // Saddle brown
        item.displayText->setStyle(sf::Text::Bold);
        item.displayText->setPosition(sf::Vector2f(100.0f, startY + i * spacing + 12.0f));
    }
}


void Menu::setupPlayModeMenu() {
    m_playModeItems.clear();
    m_playModeItems.emplace_back("Single Player", GameMode::SINGLE_PLAYER,
        "Classic Snake: Control the snake with arrow keys.\nEat apples to grow and achieve high scores!\nPerfect for learning the game mechanics.");
    m_playModeItems.emplace_back("Agent vs Player", GameMode::AGENT_VS_PLAYER,
        "AI Snake Challenge: AI controls the snake, you place apples.\nTest different AI strategies by controlling food placement.\nGreat for understanding AI behavior patterns.");
    m_playModeItems.emplace_back("Agent vs System", GameMode::AGENT_VS_SYSTEM,
        "Pure AI Showcase: AI controls snake, system spawns apples.\nWatch trained models compete for high scores.\nCompare performance between different AI techniques!");
    
    float startY = 200.0f;
    float buttonWidth = m_hasBackgroundImage ? 500.0f : 600.0f;
    
    for (size_t i = 0; i < m_playModeItems.size(); ++i) {
        auto& item = m_playModeItems[i];
        
        // Enhanced mode button
        item.buttonBackground.setSize(sf::Vector2f(buttonWidth, 55.0f));
        item.buttonBackground.setPosition(sf::Vector2f(80.0f, startY + i * 130.0f));
        item.buttonBackground.setFillColor(sf::Color(255, 255, 224, 220)); // Light yellow
        item.buttonBackground.setOutlineThickness(3.0f);
        item.buttonBackground.setOutlineColor(sf::Color(218, 165, 32)); // Goldenrod
        
        // Mode title
        item.displayText = std::make_unique<sf::Text>(m_font);
        item.displayText->setString(item.text);
        item.displayText->setCharacterSize(24);
        item.displayText->setFillColor(sf::Color(47, 79, 47));
        item.displayText->setStyle(sf::Text::Bold);
        item.displayText->setPosition(sf::Vector2f(100.0f, startY + i * 130.0f + 15.0f));
        
        // Enhanced explanation box
        item.explanationBox.setSize(sf::Vector2f(buttonWidth, 65.0f));
        item.explanationBox.setPosition(sf::Vector2f(80.0f, startY + i * 130.0f + 60.0f));
        item.explanationBox.setFillColor(sf::Color(240, 248, 255, 180)); // Alice blue
        item.explanationBox.setOutlineThickness(2.0f);
        item.explanationBox.setOutlineColor(sf::Color(173, 216, 230)); // Light blue
        
        // Enhanced explanation text
        item.explanationText = std::make_unique<sf::Text>(m_font);
        item.explanationText->setString(item.explanation);
        item.explanationText->setCharacterSize(15);
        item.explanationText->setFillColor(sf::Color(25, 25, 112)); // Midnight blue
        item.explanationText->setPosition(sf::Vector2f(90.0f, startY + i * 130.0f + 68.0f));
    }
}

void Menu::renderBackground(sf::RenderWindow& window) {
    // Render background
    window.draw(m_background);
    
    // Draw panels first
    window.draw(m_titlePanel);
    window.draw(m_contentPanel);
    
    // Then background image
    if (m_hasBackgroundImage) {
        window.draw(m_backgroundImageSprite);
    }
}

void Menu::render(sf::RenderWindow& window) {
    renderBackground(window);
    
    // Render title elements
    if (m_title) window.draw(*m_title);
    if (m_versionText) window.draw(*m_versionText);
    if (m_versionNumber) window.draw(*m_versionNumber);
    
    if (m_currentSection == MenuSection::MAIN) {
        renderMainMenu(window);
    } else {
        renderPlayModeMenu(window);
    }
}

void Menu::renderMainMenu(sf::RenderWindow& window) {
    for (const auto& item : m_mainItems) {
        window.draw(item.buttonBackground);
        if (item.displayText) window.draw(*item.displayText);
    }
    
    m_instructions->setString("Navigate: UP/DOWN | Select: ENTER | Quick Access: F1");
    window.draw(*m_instructions);
}
void Menu::renderPlayModeMenu(sf::RenderWindow& window) {
    
    // Add mode selection hint
    sf::Text hintText(m_font);
    hintText.setString("Choose how you want to experience AI-powered Snake:");
    hintText.setCharacterSize(18);
    hintText.setFillColor(sf::Color(70, 130, 180));
    hintText.setPosition(sf::Vector2f(120.0f, 160.0f));
    window.draw(hintText);
    
    for (const auto& item : m_playModeItems) {
        window.draw(item.buttonBackground);
        window.draw(item.explanationBox);
        if (item.displayText) window.draw(*item.displayText);
        if (item.explanationText) window.draw(*item.explanationText);
    }
    
    m_instructions->setString("UP/DOWN: Navigate | ENTER: Start Game | ESC: Back to Main Menu");
    window.draw(*m_instructions);
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
                    } else if (m_selectedIndex == 3 && m_statsCallback) { // Leaderboard
                        m_statsCallback();
                    } else if (m_selectedIndex == 4 && m_leaderboardCallback) { // Leaderboard
                        m_leaderboardCallback();
                    } else if (m_selectedIndex == 5 && m_quitCallback) { // Quit
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
    // Future: Add animations or dynamic effects
}

void Menu::updateSelection() {
    if (m_currentSection == MenuSection::MAIN) {
        for (size_t i = 0; i < m_mainItems.size(); ++i) {
            if (i == m_selectedIndex) {
                // Golden highlight for selected item
                m_mainItems[i].buttonBackground.setFillColor(sf::Color(255, 215, 0, 250)); // Gold
                m_mainItems[i].buttonBackground.setOutlineColor(sf::Color(255, 140, 0)); // Dark orange
                m_mainItems[i].buttonBackground.setOutlineThickness(3.0f);
                
                if (m_mainItems[i].displayText) {
                    m_mainItems[i].displayText->setFillColor(sf::Color(139, 69, 19)); // Saddle brown
                    m_mainItems[i].displayText->setStyle(sf::Text::Bold);
                }
            } else {
                // Normal yellow theme appearance
                m_mainItems[i].buttonBackground.setFillColor(sf::Color(255, 248, 220, 240)); // Light cornsilk
                m_mainItems[i].buttonBackground.setOutlineColor(sf::Color(218, 165, 32)); // Goldenrod
                m_mainItems[i].buttonBackground.setOutlineThickness(2.0f);
                
                if (m_mainItems[i].displayText) {
                    m_mainItems[i].displayText->setFillColor(sf::Color(139, 69, 19)); // Saddle brown
                    m_mainItems[i].displayText->setStyle(sf::Text::Bold);
                }
            }
        }
    } else {
        for (size_t i = 0; i < m_playModeItems.size(); ++i) {
            if (i == m_selectedIndex) {
                // Enhanced highlight
                m_playModeItems[i].buttonBackground.setFillColor(sf::Color(255, 255, 0, 240)); // Bright yellow
                m_playModeItems[i].buttonBackground.setOutlineColor(sf::Color(255, 140, 0)); // Dark orange
                m_playModeItems[i].buttonBackground.setOutlineThickness(4.0f);
                m_playModeItems[i].explanationBox.setOutlineColor(sf::Color(255, 140, 0)); // Dark orange
                m_playModeItems[i].explanationBox.setOutlineThickness(3.0f);
                
                if (m_playModeItems[i].displayText) {
                    m_playModeItems[i].displayText->setFillColor(sf::Color(139, 69, 19)); // Saddle brown
                    m_playModeItems[i].displayText->setStyle(sf::Text::Bold);
                }
            } else {
                // Normal appearance
                m_playModeItems[i].buttonBackground.setFillColor(sf::Color(255, 255, 224, 220)); // Light yellow
                m_playModeItems[i].buttonBackground.setOutlineColor(sf::Color(218, 165, 32)); // Goldenrod
                m_playModeItems[i].buttonBackground.setOutlineThickness(3.0f);
                m_playModeItems[i].explanationBox.setOutlineColor(sf::Color(173, 216, 230)); // Light blue
                m_playModeItems[i].explanationBox.setOutlineThickness(2.0f);
                
                if (m_playModeItems[i].displayText) {
                    m_playModeItems[i].displayText->setFillColor(sf::Color(47, 79, 47)); // Dark green
                    m_playModeItems[i].displayText->setStyle(sf::Text::Bold);
                }
            }
        }
    }
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