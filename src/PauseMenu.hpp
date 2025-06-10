#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <functional>
#include "GameState.hpp"

enum class PauseMenuOption {
    RESUME,
    SPEED_SETTINGS,
    AGENT_INFO,
    RESTART,
    MAIN_MENU
};

class PauseMenu {
public:
    PauseMenu();
    void initialize(sf::RenderWindow& window);
    void handleEvent(const sf::Event& event);
    void update();
    void render(sf::RenderWindow& window);
    
    void setResumeCallback(std::function<void()> callback) { m_resumeCallback = callback; }
    void setRestartCallback(std::function<void()> callback) { m_restartCallback = callback; }
    void setMainMenuCallback(std::function<void()> callback) { m_mainMenuCallback = callback; }
    void setSpeedCallback(std::function<void(float)> callback) { m_speedCallback = callback; }
    
    void setCurrentSpeed(float speed) { m_currentSpeed = speed; }
    void setCurrentAgent(const AgentConfig& agent) { m_currentAgent = agent; }
    void setGameStats(int score, int episode, float epsilon);
    
private:
    struct MenuItem {
        std::string text;
        PauseMenuOption option;
        std::unique_ptr<sf::Text> displayText;
        
        MenuItem(const std::string& t, PauseMenuOption o) : text(t), option(o) {}
    };
    
    std::vector<MenuItem> m_items;
    int m_selectedIndex;
    sf::Font m_font;
    
    // UI Elements
    std::unique_ptr<sf::Text> m_title;
    std::unique_ptr<sf::Text> m_speedText;
    std::unique_ptr<sf::Text> m_agentInfoText;
    std::unique_ptr<sf::Text> m_statsText;
    sf::RectangleShape m_background;
    sf::RectangleShape m_panel;
    
    // Callbacks
    std::function<void()> m_resumeCallback;
    std::function<void()> m_restartCallback;
    std::function<void()> m_mainMenuCallback;
    std::function<void(float)> m_speedCallback;
    
    // State
    float m_currentSpeed;
    AgentConfig m_currentAgent;
    int m_currentScore;
    int m_currentEpisode;
    float m_currentEpsilon;
    bool m_speedEditMode;
    
    void updateSelection();
    void handleSelection();
    void updateSpeedDisplay();
    void updateAgentInfo();
    void updateStatsDisplay();
};