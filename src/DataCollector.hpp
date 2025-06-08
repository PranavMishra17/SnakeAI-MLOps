 #pragma once
#include <fstream>
#include <vector>
#include <chrono>
#include <nlohmann/json.hpp>
#include "GameState.hpp"

struct GameData {
    int episode;
    int steps;
    int score;
    float totalReward;
    bool died;
    std::chrono::milliseconds duration;
    float epsilon;
    std::vector<std::pair<AgentState, Direction>> stateActionPairs;
};

class DataCollector {
public:
    DataCollector(const std::string& dataPath = "data/");
    ~DataCollector();
    
    void startEpisode(int episode);
    void recordStep(const AgentState& state, Direction action, float reward);
    void endEpisode(int score, bool died, float epsilon);
    
    void saveTrainingData();
    void saveSummary();
    
    const std::vector<GameData>& getGameHistory() const { return m_gameHistory; }
    
private:
    std::string m_dataPath;
    std::vector<GameData> m_gameHistory;
    GameData m_currentGame;
    std::chrono::steady_clock::time_point m_episodeStart;
    
    void ensureDirectoryExists(const std::string& path);
};
