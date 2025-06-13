#include "PauseMenu.hpp"
#include <spdlog/spdlog.h>

PauseMenu::PauseMenu() 
    : m_selectedIndex(0), m_currentSpeed(1.0f), m_currentScore(0), 
      m_currentEpisode(0), m_currentEpsilon(0.0f), m_speedEditMode(false) {}

void PauseMenu::initialize(sf::RenderWindow& window) {
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
            spdlog::info("PauseMenu: Font loaded from: {}", path);
            break;
        }
    }
    
    if (!fontLoaded) {
        spdlog::error("PauseMenu: Failed to load any font");
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Semi-transparent background
    m_background.setSize(sf::Vector2f(static_cast<float>(windowSize.x), 
                                     static_cast<float>(windowSize.y)));
    m_background.setFillColor(sf::Color(0, 0, 0, 150));
    
    // Main panel
    m_panel.setSize(sf::Vector2f(600.0f, 500.0f));
    m_panel.setPosition(sf::Vector2f(windowSize.x / 2.0f - 300.0f, 
                                    windowSize.y / 2.0f - 250.0f));
    m_panel.setFillColor(sf::Color(40, 40, 40));
    m_panel.setOutlineThickness(3.0f);
    m_panel.setOutlineColor(sf::Color(100, 100, 100));
    
    // Title
    m_title = std::make_unique<sf::Text>(m_font);
    m_title->setString("GAME PAUSED");
    m_title->setCharacterSize(36);
    m_title->setFillColor(sf::Color::Yellow);
    m_title->setPosition(sf::Vector2f(windowSize.x / 2.0f - 120.0f, 
                                     windowSize.y / 2.0f - 220.0f));
    
    // Menu items
    m_items.clear();
    m_items.emplace_back("Resume Game", PauseMenuOption::RESUME);
    m_items.emplace_back("Speed Settings", PauseMenuOption::SPEED_SETTINGS);
    m_items.emplace_back("Agent Info", PauseMenuOption::AGENT_INFO);
    m_items.emplace_back("Restart Episode", PauseMenuOption::RESTART);
    m_items.emplace_back("Main Menu", PauseMenuOption::MAIN_MENU);
    
    float startY = windowSize.y / 2.0f - 150.0f;
    for (size_t i = 0; i < m_items.size(); ++i) {
        m_items[i].displayText = std::make_unique<sf::Text>(m_font);
        m_items[i].displayText->setString(m_items[i].text);
        m_items[i].displayText->setCharacterSize(24);
        m_items[i].displayText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 120.0f, 
                                                        startY + i * 50.0f));
    }
    
    // Info displays
    m_speedText = std::make_unique<sf::Text>(m_font);
    m_speedText->setCharacterSize(18);
    m_speedText->setFillColor(sf::Color(200, 200, 200));
    m_speedText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 280.0f, 
                                         windowSize.y / 2.0f + 100.0f));
    
    m_agentInfoText = std::make_unique<sf::Text>(m_font);
    m_agentInfoText->setCharacterSize(16);
    m_agentInfoText->setFillColor(sf::Color(150, 200, 255));
    m_agentInfoText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 280.0f, 
                                             windowSize.y / 2.0f + 130.0f));
    
    m_statsText = std::make_unique<sf::Text>(m_font);
    m_statsText->setCharacterSize(16);
    m_statsText->setFillColor(sf::Color(200, 255, 200));
    m_statsText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 280.0f, 
                                         windowSize.y / 2.0f + 160.0f));
    
    updateSelection();
    updateSpeedDisplay();
    updateAgentInfo();
    updateStatsDisplay();
}

void PauseMenu::handleEvent(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if (m_speedEditMode) {
            switch (keyPressed->code) {
                case sf::Keyboard::Key::Equal: // + key
                    m_currentSpeed = std::min(3.0f, m_currentSpeed + 0.1f);
                    if (m_speedCallback) m_speedCallback(m_currentSpeed);
                    updateSpeedDisplay();
                    break;
                case sf::Keyboard::Key::Hyphen: // - key
                    m_currentSpeed = std::max(0.5f, m_currentSpeed - 0.1f);
                    if (m_speedCallback) m_speedCallback(m_currentSpeed);
                    updateSpeedDisplay();
                    break;
                case sf::Keyboard::Key::Enter:
                case sf::Keyboard::Key::Escape:
                    m_speedEditMode = false;
                    updateSelection();
                    break;
            }
        } else {
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
                    handleSelection();
                    break;
                case sf::Keyboard::Key::Escape:
                    if (m_resumeCallback) m_resumeCallback();
                    break;
            }
        }
    }
}

void PauseMenu::handleSelection() {
    switch (m_items[m_selectedIndex].option) {
        case PauseMenuOption::RESUME:
            if (m_resumeCallback) m_resumeCallback();
            break;
        case PauseMenuOption::SPEED_SETTINGS:
            m_speedEditMode = true;
            updateSelection();
            break;
        case PauseMenuOption::AGENT_INFO:
            // Already displayed, just for navigation
            break;
        case PauseMenuOption::RESTART:
            if (m_restartCallback) m_restartCallback();
            break;
        case PauseMenuOption::MAIN_MENU:
            if (m_mainMenuCallback) m_mainMenuCallback();
            break;
    }
}

void PauseMenu::update() {
    // Add any animations or dynamic updates here
}

void PauseMenu::render(sf::RenderWindow& window) {
    window.draw(m_background);
    window.draw(m_panel);
    
    if (m_title) window.draw(*m_title);
    
    for (const auto& item : m_items) {
        if (item.displayText) window.draw(*item.displayText);
    }
    
    if (m_speedText) window.draw(*m_speedText);
    if (m_agentInfoText) window.draw(*m_agentInfoText);
    if (m_statsText) window.draw(*m_statsText);
}

void PauseMenu::updateSelection() {
    for (size_t i = 0; i < m_items.size(); ++i) {
        if (m_items[i].displayText) {
            if (i == m_selectedIndex && !m_speedEditMode) {
                m_items[i].displayText->setFillColor(sf::Color::Cyan);
                m_items[i].displayText->setStyle(sf::Text::Bold);
            } else {
                m_items[i].displayText->setFillColor(sf::Color::White);
                m_items[i].displayText->setStyle(sf::Text::Regular);
            }
        }
    }
}

void PauseMenu::updateSpeedDisplay() {
    std::string speedStr = "Speed: " + std::to_string(m_currentSpeed).substr(0, 4) + " blocks/sec";
    if (m_speedEditMode) {
        speedStr += " (Use +/- to adjust, ENTER to confirm)";
        m_speedText->setFillColor(sf::Color::Yellow);
    } else {
        m_speedText->setFillColor(sf::Color(200, 200, 200));
    }
    m_speedText->setString(speedStr);
}

void PauseMenu::updateAgentInfo() {
    std::string info = "Agent: " + m_currentAgent.name;
    if (m_currentAgent.type != AgentType::HUMAN) {
        info += " | Type: " + m_currentAgent.getAgentTypeString();
    }
    m_agentInfoText->setString(info);
}

void PauseMenu::updateStatsDisplay() {
    std::string stats = "Score: " + std::to_string(m_currentScore) + 
                       " | Episode: " + std::to_string(m_currentEpisode);
    if (m_currentEpsilon > 0) {
        stats += " | Epsilon: " + std::to_string(m_currentEpsilon).substr(0, 5);
    }
    m_statsText->setString(stats);
}

void PauseMenu::setGameStats(int score, int episode, float epsilon) {
    m_currentScore = score;
    m_currentEpisode = episode;
    m_currentEpsilon = epsilon;
    updateStatsDisplay();
}