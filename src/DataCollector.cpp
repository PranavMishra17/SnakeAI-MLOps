 
#include "DataCollector.hpp"
#include <filesystem>
#include <spdlog/spdlog.h>

DataCollector::DataCollector(const std::string& dataPath) : m_dataPath(dataPath) {
    ensureDirectoryExists(m_dataPath);
}

DataCollector::~DataCollector() {
    saveTrainingData();
    saveSummary();
}

void DataCollector::startEpisode(int episode) {
    m_currentGame = GameData{};
    m_currentGame.episode = episode;
    m_currentGame.steps = 0;
    m_currentGame.score = 0;
    m_currentGame.totalReward = 0.0f;
    m_episodeStart = std::chrono::steady_clock::now();
}

void DataCollector::recordStep(const AgentState& state, Direction action, float reward) {
    m_currentGame.steps++;
    m_currentGame.totalReward += reward;
    m_currentGame.stateActionPairs.push_back({state, action});
}

void DataCollector::endEpisode(int score, bool died, float epsilon) {
    auto duration = std::chrono::steady_clock::now() - m_episodeStart;
    m_currentGame.duration = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    m_currentGame.score = score;
    m_currentGame.died = died;
    m_currentGame.epsilon = epsilon;
    
    m_gameHistory.push_back(m_currentGame);
    
    // Save every 10 episodes
    if (m_gameHistory.size() % 10 == 0) {
        saveTrainingData();
    }
}

void DataCollector::saveTrainingData() {
    nlohmann::json j;
    
    for (const auto& game : m_gameHistory) {
        nlohmann::json gameJson;
        gameJson["episode"] = game.episode;
        gameJson["steps"] = game.steps;
        gameJson["score"] = game.score;
        gameJson["total_reward"] = game.totalReward;
        gameJson["died"] = game.died;
        gameJson["duration_ms"] = game.duration.count();
        gameJson["epsilon"] = game.epsilon;
        
        j.push_back(gameJson);
    }
    
    std::string filename = m_dataPath + "training_data.json";
    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(4);
        spdlog::info("Training data saved to {}", filename);
    }
}

void DataCollector::saveSummary() {
    if (m_gameHistory.empty()) return;
    
    nlohmann::json summary;
    
    // Calculate statistics
    int totalEpisodes = m_gameHistory.size();
    float avgScore = 0, avgSteps = 0, avgReward = 0;
    int maxScore = 0, maxSteps = 0;
    
    for (const auto& game : m_gameHistory) {
        avgScore += game.score;
        avgSteps += game.steps;
        avgReward += game.totalReward;
        maxScore = std::max(maxScore, game.score);
        maxSteps = std::max(maxSteps, game.steps);
    }
    
    avgScore /= totalEpisodes;
    avgSteps /= totalEpisodes;
    avgReward /= totalEpisodes;
    
    summary["total_episodes"] = totalEpisodes;
    summary["average_score"] = avgScore;
    summary["average_steps"] = avgSteps;
    summary["average_reward"] = avgReward;
    summary["max_score"] = maxScore;
    summary["max_steps"] = maxSteps;
    
    std::string filename = m_dataPath + "training_summary.json";
    std::ofstream file(filename);
    if (file.is_open()) {
        file << summary.dump(4);
        spdlog::info("Summary saved to {}", filename);
    }
}

void DataCollector::ensureDirectoryExists(const std::string& path) {
    std::filesystem::create_directories(path);
}