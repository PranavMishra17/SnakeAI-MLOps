#pragma once
#include <SFML/Graphics.hpp>
#include <memory>
#include <spdlog/spdlog.h>
#include "GameState.hpp"
#include "Grid.hpp"
#include "Snake.hpp"
#include "Apple.hpp"
#include "Menu.hpp"
#include "PauseMenu.hpp"
#include "AgentSelection.hpp"
#include "Leaderboard.hpp"
#include "MLAgents.hpp"
#include "UnifiedDataCollector.hpp"
#include "InputManager.hpp"
#include "StatsGallery.hpp"
#include "Settings.hpp"
#include "HowToPlay.hpp"

// Forward declarations
class PauseMenu;
class AgentSelection;
class Leaderboard;
class IAgent;

class Game {
public:
    Game();
    ~Game();
    void run();

private:
    // Event handling
    void processEvents();
    void setupCallbacks();
    void handleGlobalKeys(const sf::Event& event);
    void handleGameplayEvents(const sf::Event& event);
    void handleApplePlacement(const sf::Event& event);
    void handleGameOverEvents(const sf::Event& event);
    
    // Game logic
    void update(float deltaTime);
    void updateGame(float deltaTime);
    void handleSnakeDeath(const EnhancedState& currentState, Direction action);
    float calculateReward(bool ateFood, bool died) const;
    void spawnNewApple();
    
    // Rendering
    void render();
    void renderGame();
    void renderUI();
    void renderEnhancedModelInfo();     // NEW: Enhanced model performance display
    void renderPerformanceMetrics();    // NEW: Performance metrics panel
    void renderModelComparison();       // NEW: Compare current vs best scores
    void renderSettings();
    void renderHowToPlay();
    void renderGameOverScreen();
    
    // Menu handling
    void handleMenuSelection(GameMode mode);
    void selectAgent(const AgentConfig& config);
    void startGame();
    void resetGame();
    bool shouldAddToLeaderboard() const;
    
    // NEW: Enhanced model management
    void updateCurrentModelStats();
    void loadModelPerformanceData();
    std::string formatModelPerformance() const;
    sf::Color getPerformanceColor(float currentScore, float bestScore) const;
    
    // Utility methods
    std::string getGameModeString() const;
    
    // Game components
    std::unique_ptr<sf::RenderWindow> m_window;
    std::unique_ptr<Grid> m_grid;
    std::unique_ptr<Snake> m_snake;
    std::unique_ptr<Apple> m_apple;
    std::unique_ptr<Menu> m_menu;
    std::unique_ptr<PauseMenu> m_pauseMenu;
    std::unique_ptr<AgentSelection> m_agentSelection;
    std::unique_ptr<Leaderboard> m_leaderboard;
    std::unique_ptr<IAgent> m_currentAgent;
    std::unique_ptr<UnifiedDataCollector> m_dataCollector;
    std::unique_ptr<InputManager> m_inputManager;
    std::unique_ptr<StatsGallery> m_statsGallery;  // ADD THIS LINE
    
    // NEW: Enhanced model management
    std::unique_ptr<TrainedModelManager> m_modelManager;
    TrainedModelInfo m_currentModelInfo;
    ModelPerformanceData m_sessionPerformance;  // Track current session performance
    
    // Game state
    GameState m_currentState;
    GameMode m_gameMode;
    AgentType m_currentAgentType;
    AgentConfig m_currentAgentConfig;
    
    // Timing
    sf::Clock m_clock;
    sf::Clock m_gameClock;
    sf::Clock m_sessionClock;               // NEW: Track session time
    float m_moveTimer;
    float m_moveSpeed; // blocks per second
    
    // Game statistics
    int m_score;
    int m_episode;
    int m_sessionBestScore;                 // NEW: Best score this session
    int m_sessionEpisodes;                  // NEW: Episodes played this session
    float m_sessionAverageScore;            // NEW: Session average
    bool m_gameOver;
    mutable int m_previousDistance; // For reward calculation
    
    // NEW: Performance tracking
    std::vector<int> m_recentScores;        // Last 10 scores for trend analysis
    float m_currentEfficiency;              // Current score/episode efficiency
    bool m_beatModelBest;                   // Flag if we beat model's best score
    
    // Apple placement for agent vs player mode
    sf::Vector2i m_nextApplePos;
    bool m_hasNextApple;
    
    // Settings
    float m_minSpeed = 0.5f;
    float m_maxSpeed = 3.0f;
    
    // NEW: UI Enhancement flags
    bool m_showEnhancedStats = true;        // Show detailed model stats
    bool m_showPerformanceComparison = true; // Show vs model best
    sf::Font m_uiFont;                      // Dedicated UI font

        // Add to private members:
    std::unique_ptr<Settings> m_settings;
    std::unique_ptr<HowToPlay> m_howToPlay;
};