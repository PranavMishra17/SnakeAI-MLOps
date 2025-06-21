#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <functional>
#include <fstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "GameState.hpp"
#include "MLAgents.hpp"

// Image viewer component for analysis charts
class ImageViewer {
public:
    ImageViewer();
    void initialize(sf::RenderWindow& window);
    void loadImage(const std::string& imagePath, const std::string& title);
    void handleEvent(const sf::Event& event);
    void render(sf::RenderWindow& window);
    bool isVisible() const { return m_visible; }
    void close() { m_visible = false; }
    
    void setCloseCallback(std::function<void()> callback) { m_closeCallback = callback; }
    ImageViewer(const sf::Texture& imageTexture);

private:
    sf::Font m_font;
    sf::Texture m_imageTexture;
    sf::Sprite m_imageSprite;
    std::unique_ptr<sf::Text> m_titleText;        // Changed to pointer
    std::unique_ptr<sf::Text> m_instructionText;  // Changed to pointer
    sf::RectangleShape m_background;
    sf::RectangleShape m_imageFrame;
    bool m_visible;
    std::string m_currentTitle;
    std::function<void()> m_closeCallback;
    
    void updateImageDisplay(sf::RenderWindow& window);
};

// Enhanced leaderboard with model stats
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
    void loadModelPerformanceData(); // NEW: Load model best scores
    
    const std::vector<LeaderboardEntry>& getEntries() const { return m_entries; }
    
private:
    enum class LeaderboardState {
        VIEWING,
        ENTERING_NAME,
        MODEL_STATS,    // NEW: Model statistics view
        IMAGE_VIEWING   // NEW: Image viewing mode
    };
    
    enum class StatsSection {
        MAIN_LEADERBOARD,
        MODEL_PERFORMANCE,
        ANALYSIS_CHARTS
    };
    
    // Core data
    std::vector<LeaderboardEntry> m_entries;
    std::vector<LeaderboardEntry> m_modelEntries; // NEW: Model best scores
    LeaderboardState m_state;
    StatsSection m_currentSection; // NEW: Current stats section
    sf::Font m_font;
    
    // UI Elements
    std::unique_ptr<sf::Text> m_title;
    std::unique_ptr<sf::Text> m_instructions;
    std::unique_ptr<sf::Text> m_namePrompt;
    std::unique_ptr<sf::Text> m_inputText;
    std::unique_ptr<sf::Text> m_sectionTitle; // NEW: Section navigation
    std::vector<std::unique_ptr<sf::Text>> m_entryTexts;
    std::vector<std::unique_ptr<sf::Text>> m_statsButtons; // NEW: Analysis image buttons
    sf::RectangleShape m_background;
    sf::RectangleShape m_inputBox;
    
    // NEW: Model performance data
    TrainedModelManager m_modelManager;
    EvaluationReportData m_evaluationData;
    
    // NEW: Image viewer for analysis charts
    std::unique_ptr<ImageViewer> m_imageViewer;
    
    // Navigation
    int m_selectedStatsButton; // NEW: For navigation in stats section
    
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
    
    // NEW: Enhanced functionality
    void setupStatsButtons();
    void renderMainLeaderboard(sf::RenderWindow& window);
    void renderModelPerformance(sf::RenderWindow& window);
    void renderAnalysisCharts(sf::RenderWindow& window);
    void handleStatsNavigation(const sf::Event& event);
    void handleImageViewing(const sf::Event& event);
    
    void loadModelBestScores();
    void addModelEntriesToMain();
    void updateSectionDisplay();
    
    // NEW: Analysis image management
    struct AnalysisImage {
        std::string name;
        std::string path;
        std::string description;
        sf::RectangleShape button;
        std::unique_ptr<sf::Text> buttonText;
    };
    std::vector<AnalysisImage> m_analysisImages;
    void initializeAnalysisImages();
    
    static constexpr int MAX_ENTRIES = 15; // Increased for model entries
    static constexpr int MAX_NAME_LENGTH = 20;
    static const std::string LEADERBOARD_PATH;
};