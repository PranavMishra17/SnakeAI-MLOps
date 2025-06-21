// HowToPlay.cpp
#include "HowToPlay.hpp"
#include <spdlog/spdlog.h>

HowToPlay::HowToPlay() {}

void HowToPlay::initialize(sf::RenderWindow& window) {
    std::vector<std::string> fontPaths = {
        "assets/fonts/ARIAL.TTF", "assets/fonts/arial.ttf", 
        "assets/fonts/ArialCE.ttf", "assets/fonts/Roboto.ttf"
    };
    
    for (const auto& path : fontPaths) {
        if (m_font.openFromFile(path)) break;
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Yellow theme background
    m_background.setSize(sf::Vector2f(windowSize.x, windowSize.y));
    m_background.setFillColor(sf::Color(255, 253, 208));
    
    // Title panel
    m_titlePanel.setSize(sf::Vector2f(windowSize.x - 100.0f, 100.0f));
    m_titlePanel.setPosition(sf::Vector2f(50.0f, 20.0f));
    m_titlePanel.setFillColor(sf::Color(255, 248, 220, 150));
    m_titlePanel.setOutlineThickness(2.0f);
    m_titlePanel.setOutlineColor(sf::Color(218, 165, 32, 100));
    
    // Content panel
    m_contentPanel.setSize(sf::Vector2f(windowSize.x - 100.0f, windowSize.y - 200.0f));
    m_contentPanel.setPosition(sf::Vector2f(50.0f, 140.0f));
    m_contentPanel.setFillColor(sf::Color(255, 250, 205, 180));
    m_contentPanel.setOutlineThickness(2.0f);
    m_contentPanel.setOutlineColor(sf::Color(218, 165, 32, 150));
    
    // Title
    m_title = std::make_unique<sf::Text>(m_font);
    m_title->setString("HOW TO PLAY");
    m_title->setCharacterSize(48);
    m_title->setFillColor(sf::Color(139, 69, 19));
    m_title->setStyle(sf::Text::Bold);
    
    auto titleBounds = m_title->getLocalBounds();
    m_title->setPosition(sf::Vector2f((windowSize.x - titleBounds.size.x) / 2.0f, 35.0f));
    
    // Instructions
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setString("ESC: Back to Menu");
    m_instructions->setCharacterSize(18);
    m_instructions->setFillColor(sf::Color(101, 67, 33));
    m_instructions->setPosition(sf::Vector2f(70.0f, windowSize.y - 60.0f));
    
    createHelpContent();
}

void HowToPlay::createHelpContent() {
    std::vector<std::string> helpInfo = {
        "Game Modes:",
        "",
        "Single Player - Classic Snake",
        "  Control snake with ARROW KEYS or WASD",
        "  Eat red apples to grow and score points",
        "  Avoid hitting walls or your own body",
        "",
        "Agent vs Player - Test AI Strategies",
        "  AI controls the snake automatically",
        "  Click with mouse to place apples strategically",
        "  Observe how AI responds to different placements",
        "",
        "Agent vs System - Pure AI Training",
        "  AI controls snake, system spawns apples",
        "  Watch different algorithms learn and improve",
        "  Compare Q-Learning, DQN, PPO, and Actor-Critic",
        "",
        "Game Controls:",
        "ESC - Pause game / Open menus",
        "+/- - Adjust game speed (0.5x to 3.0x)",
        "F1 - Quick access to leaderboard",
        "F2 - Switch AI agent (in AI modes)",
        "",
        "AI Algorithms Available:",
        "Q-Learning - Tabular reinforcement learning",
        "DQN - Deep Q-Network with neural networks",
        "PPO - Proximal Policy Optimization",
        "Actor-Critic - Dual network architecture"
    };
    
    m_helpTexts.clear();
    float startY = 170.0f;
    
    for (size_t i = 0; i < helpInfo.size(); ++i) {
        auto text = std::make_unique<sf::Text>(m_font);
        text->setString(helpInfo[i]);
        
        if (helpInfo[i].find(":") != std::string::npos && !helpInfo[i].empty()) {
            text->setCharacterSize(20);
            text->setFillColor(sf::Color(160, 82, 45));
            text->setStyle(sf::Text::Bold);
        } else if (helpInfo[i].find(" - ") != std::string::npos) {
            text->setCharacterSize(16);
            text->setFillColor(sf::Color(139, 69, 19));
            text->setStyle(sf::Text::Bold);
        } else {
            text->setCharacterSize(16);
            text->setFillColor(sf::Color(101, 67, 33));
        }
        
        text->setPosition(sf::Vector2f(80.0f, startY + i * 22.0f));
        m_helpTexts.push_back(std::move(text));
    }
}

void HowToPlay::handleEvent(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if (keyPressed->code == sf::Keyboard::Key::Escape) {
            if (m_backCallback) m_backCallback();
        }
    }
}

void HowToPlay::update() {}

void HowToPlay::render(sf::RenderWindow& window) {
    window.draw(m_background);
    window.draw(m_titlePanel);
    window.draw(m_contentPanel);
    window.draw(*m_title);
    
    for (const auto& text : m_helpTexts) {
        window.draw(*text);
    }
    
    window.draw(*m_instructions);
}