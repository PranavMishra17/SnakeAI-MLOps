// Leaderboard.hpp - Fixed version
#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <functional>
#include <fstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "GameState.hpp"
#include "MLAgents.hpp"

class Leaderboard {
public:
    Leaderboard();
    void initialize(sf::RenderWindow& window);
    void handleEvent(const sf::Event& event);
    void update();
    void render(sf::RenderWindow& window);
    
    void addScore(const std::string& playerName, AgentType agentType, int score, int episode);
    void promptForName(int score, AgentType agentType, int episode);
    void setBackCallback(std::function<void()> callback) { m_backCallback = callback; }
    
    void loadLeaderboard();
    void saveLeaderboard();
    void loadModelPerformanceData();
    
    const std::vector<LeaderboardEntry>& getEntries() const { return m_entries; }
    
private:
    enum class LeaderboardState {
        VIEWING,
        ENTERING_NAME
    };
    
    // Core data
    std::vector<LeaderboardEntry> m_entries;
    std::vector<LeaderboardEntry> m_modelEntries;
    LeaderboardState m_state;
    sf::Font m_font;
    
    // UI Elements
    std::unique_ptr<sf::Text> m_title;
    std::unique_ptr<sf::Text> m_instructions;
    std::unique_ptr<sf::Text> m_namePrompt;
    std::unique_ptr<sf::Text> m_inputText;
    std::vector<std::unique_ptr<sf::Text>> m_entryTexts;
    sf::RectangleShape m_background;
    sf::RectangleShape m_inputBox;
    sf::RectangleShape m_titlePanel;    // ADD THIS
    sf::RectangleShape m_contentPanel;  // ADD THIS
    
    // Model performance data
    TrainedModelManager m_modelManager;
    EvaluationReportData m_evaluationData;
    
    // Input state
    std::string m_currentInput;
    int m_pendingScore;
    AgentType m_pendingAgentType;
    int m_pendingEpisode;
    
    std::function<void()> m_backCallback;
    
    // Core functionality
    void updateDisplay();
    void handleNameInput(std::uint32_t unicode);
    void finalizeName();
    std::string getDefaultName(AgentType agentType);
    void sortEntries();
    void renderMainLeaderboard(sf::RenderWindow& window);
    
    static constexpr int MAX_ENTRIES = 15;
    static constexpr int MAX_NAME_LENGTH = 20;
    static const std::string LEADERBOARD_PATH;
};