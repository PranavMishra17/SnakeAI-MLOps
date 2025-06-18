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
                // Bright selection colors
                m_items[i].displayText->setFillColor(sf::Color(255, 215, 0)); // Gold
                m_items[i].displayText->setStyle(sf::Text::Bold);
            } else {
                // Normal light colors
                m_items[i].displayText->setFillColor(sf::Color(47, 79, 47)); // Dark green
                m_items[i].displayText->setStyle(sf::Text::Regular);
            }
        }
    }
}

void PauseMenu::updateSpeedDisplay() {
    std::string speedStr = "Speed: " + std::to_string(m_currentSpeed).substr(0, 4) + " blocks/sec";
    if (m_speedEditMode) {
        speedStr += " (Use +/- to adjust, ENTER to confirm)";
        m_speedText->setFillColor(sf::Color(255, 140, 0)); // Dark orange for edit mode
    } else {
        m_speedText->setFillColor(sf::Color(47, 79, 47)); // Dark green for normal
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
                default:
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
                default:
                    break;
            }
        }
    }

}