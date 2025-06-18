#include "UnifiedDataCollector.hpp"
#include <filesystem>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>

UnifiedDataCollector::UnifiedDataCollector(const std::string& dataPath) 
    : m_dataPath(dataPath), m_currentEpisodeNumber(0) {
    ensureDirectoryExists(m_dataPath);
    m_trainingStart = std::chrono::steady_clock::now();
}

UnifiedDataCollector::~UnifiedDataCollector() {
    saveTrainingData();
}

void UnifiedDataCollector::setAgentType(AgentType agentType) {
    switch(agentType) {
        case AgentType::Q_LEARNING:
            m_dataType = AgentDataType::Q_LEARNING;
            break;
        case AgentType::DEEP_Q_NETWORK:
        case AgentType::PPO:
        case AgentType::ACTOR_CRITIC:
            m_dataType = AgentDataType::NEURAL_NETWORK;
            break;
        case AgentType::GENETIC_ALGORITHM:
            m_dataType = AgentDataType::GENETIC_ALGORITHM;
            break;
        default:
            m_dataType = AgentDataType::Q_LEARNING;
    }
}

void UnifiedDataCollector::startEpisode(int episode) {
    m_currentEpisodeNumber = episode;
    m_currentEpisodeData.clear();
    m_episodeStart = std::chrono::steady_clock::now();
}

void UnifiedDataCollector::recordTransition(const UnifiedTransition& transition) {
    UnifiedTransition trans = transition;
    trans.episode = m_currentEpisodeNumber;
    trans.step = m_currentEpisodeData.size();
    trans.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - m_episodeStart);
    
    m_currentEpisodeData.push_back(trans);
}

void UnifiedDataCollector::endEpisode(int finalScore, bool died, float epsilon) {
    if (!m_currentEpisodeData.empty()) {
        // Mark the last transition as terminal
        m_currentEpisodeData.back().terminal = true;
        
        // Calculate episode metrics
        float totalReward = 0.0f;
        for (const auto& transition : m_currentEpisodeData) {
            totalReward += transition.reward;
        }
        
        m_episodeScores.push_back(static_cast<float>(finalScore));
        m_episodeRewards.push_back(totalReward);
        m_episodeLengths.push_back(static_cast<int>(m_currentEpisodeData.size()));
        
        // Store episode
        m_episodeHistory.push_back(std::move(m_currentEpisodeData));
        m_currentEpisodeData.clear();
        
        // Auto-save every 10 episodes
        if (m_episodeHistory.size() % 10 == 0) {
            saveTrainingData();
        }
        
        spdlog::info("Episode {} completed: Score={}, Length={}, Reward={:.2f}, Died={}, Epsilon={:.3f}",
                     m_episodeHistory.size(), finalScore, m_episodeLengths.back(), totalReward, died, epsilon);
    }
}

std::vector<QLearningTransition> UnifiedDataCollector::getQLearningData(int maxTransitions) {
    std::vector<QLearningTransition> qData;
    int transitionCount = 0;
    
    for (const auto& episode : m_episodeHistory) {
        for (size_t i = 0; i < episode.size(); ++i) {
            if (maxTransitions > 0 && transitionCount >= maxTransitions) break;
            
            const auto& transition = episode[i];
            QLearningTransition qTrans;
            qTrans.stateKey = transition.basicState.toString();
            qTrans.action = transition.actionIndex;
            qTrans.reward = transition.reward;
            qTrans.terminal = transition.terminal;
            
            // Get next state
            if (i + 1 < episode.size()) {
                qTrans.nextStateKey = episode[i + 1].basicState.toString();
            } else {
                qTrans.nextStateKey = transition.basicState.toString(); // Terminal state
            }
            
            qData.push_back(qTrans);
            transitionCount++;
        }
        if (maxTransitions > 0 && transitionCount >= maxTransitions) break;
    }
    
    return qData;
}

ExperienceReplayBuffer UnifiedDataCollector::getDQNReplayBuffer() {
    ExperienceReplayBuffer buffer(50000); // 50K capacity for DQN
    
    for (const auto& episode : m_episodeHistory) {
        for (size_t i = 0; i < episode.size(); ++i) {
            const auto& transition = episode[i];
            DQNTransition dqnTrans;
            dqnTrans.state = transition.enhancedState.toVector();
            dqnTrans.action = transition.actionIndex;
            dqnTrans.reward = transition.reward;
            dqnTrans.terminal = transition.terminal;
            
            // Get next state
            if (i + 1 < episode.size()) {
                dqnTrans.nextState = episode[i + 1].enhancedState.toVector();
            } else {
                dqnTrans.nextState = transition.enhancedState.toVector(); // Terminal state
            }
            
            buffer.addTransition(dqnTrans);
        }
    }
    
    return buffer;
}

std::vector<std::vector<UnifiedTransition>> UnifiedDataCollector::getPolicyGradientEpisodes(int maxEpisodes) {
    if (maxEpisodes < 0 || maxEpisodes > static_cast<int>(m_episodeHistory.size())) {
        return m_episodeHistory;
    }
    
    std::vector<std::vector<UnifiedTransition>> result;
    int startIdx = std::max(0, static_cast<int>(m_episodeHistory.size()) - maxEpisodes);
    
    for (int i = startIdx; i < static_cast<int>(m_episodeHistory.size()); ++i) {
        result.push_back(m_episodeHistory[i]);
    }
    
    return result;
}

TrainingMetrics UnifiedDataCollector::calculateMetrics() const {
    TrainingMetrics metrics;
    
    if (m_episodeScores.empty()) return metrics;
    
    metrics.totalEpisodes = static_cast<int>(m_episodeScores.size());
    metrics.totalSteps = getTotalTransitions();
    
    // Calculate score metrics
    metrics.averageScore = std::accumulate(m_episodeScores.begin(), m_episodeScores.end(), 0.0f) / m_episodeScores.size();
    metrics.maxScore = *std::max_element(m_episodeScores.begin(), m_episodeScores.end());
    
    // Calculate standard deviation
    float scoreVariance = 0.0f;
    for (float score : m_episodeScores) {
        scoreVariance += (score - metrics.averageScore) * (score - metrics.averageScore);
    }
    metrics.scoreStandardDeviation = std::sqrt(scoreVariance / m_episodeScores.size());
    
    // Episode length metrics
    float avgLength = std::accumulate(m_episodeLengths.begin(), m_episodeLengths.end(), 0.0f) / m_episodeLengths.size();
    metrics.averageEpisodeLength = avgLength;
    
    // Reward metrics
    metrics.averageReward = std::accumulate(m_episodeRewards.begin(), m_episodeRewards.end(), 0.0f) / m_episodeRewards.size();
    
    // Time metrics
    auto currentTime = std::chrono::steady_clock::now();
    auto trainingDuration = std::chrono::duration_cast<std::chrono::seconds>(currentTime - m_trainingStart);
    metrics.wallClockTime = static_cast<float>(trainingDuration.count());
    
    if (metrics.wallClockTime > 0) {
        metrics.samplesPerSecond = static_cast<float>(metrics.totalSteps) / metrics.wallClockTime;
    }
    
    return metrics;
}

void UnifiedDataCollector::saveTrainingData(const std::string& filename) {
    std::string filepath = filename.empty() ? getDefaultFilename() : filename;
    
    try {
        nlohmann::json j;
        
        // Metadata
        auto metrics = calculateMetrics();
        j["metadata"] = {
            {"agentDataType", static_cast<int>(m_dataType)},
            {"totalEpisodes", metrics.totalEpisodes},
            {"totalSteps", metrics.totalSteps},
            {"trainingDate", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"averageScore", metrics.averageScore},
            {"maxScore", metrics.maxScore},
            {"averageReward", metrics.averageReward}
        };
        
        // Episodes (save recent episodes only to keep file size manageable)
        j["episodes"] = nlohmann::json::array();
        int startIdx = std::max(0, static_cast<int>(m_episodeHistory.size()) - 100); // Last 100 episodes
        
        for (int i = startIdx; i < static_cast<int>(m_episodeHistory.size()); ++i) {
            const auto& episode = m_episodeHistory[i];
            nlohmann::json episodeJson;
            episodeJson["episodeNumber"] = i;
            episodeJson["episodeLength"] = episode.size();
            
            if (!episode.empty()) {
                float episodeReward = 0.0f;
                for (const auto& transition : episode) {
                    episodeReward += transition.reward;
                }
                episodeJson["totalReward"] = episodeReward;
                episodeJson["died"] = episode.back().terminal;
                
                // Sample some transitions (not all to keep file size reasonable)
                episodeJson["sampleTransitions"] = nlohmann::json::array();
                int sampleRate = std::max(1, static_cast<int>(episode.size()) / 10); // Sample every N transitions
                
                for (size_t j = 0; j < episode.size(); j += sampleRate) {
                    const auto& trans = episode[j];
                    nlohmann::json transJson;
                    transJson["step"] = trans.step;
                    transJson["action"] = trans.actionIndex;
                    transJson["reward"] = trans.reward;
                    transJson["terminal"] = trans.terminal;
                    transJson["epsilon"] = trans.epsilon;
                    
                    // Basic state for Q-learning
                    transJson["basicState"] = {
                        {"dangerStraight", trans.basicState.dangerStraight},
                        {"dangerLeft", trans.basicState.dangerLeft},
                        {"dangerRight", trans.basicState.dangerRight},
                        {"currentDirection", static_cast<int>(trans.basicState.currentDirection)},
                        {"foodLeft", trans.basicState.foodLeft},
                        {"foodRight", trans.basicState.foodRight},
                        {"foodUp", trans.basicState.foodUp},
                        {"foodDown", trans.basicState.foodDown}
                    };
                    
                    episodeJson["sampleTransitions"].push_back(transJson);
                }
            }
            
            j["episodes"].push_back(episodeJson);
        }
        
        std::ofstream file(filepath);
        if (file.is_open()) {
            file << j.dump(2); // Pretty print with indentation
            spdlog::info("Training data saved to {}", filepath);
        } else {
            spdlog::error("Failed to open file for writing: {}", filepath);
        }
        
    } catch (const std::exception& e) {
        spdlog::error("Error saving training data: {}", e.what());
    }
}

void UnifiedDataCollector::loadTrainingData(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            spdlog::warn("Training data file not found: {}", filename);
            return;
        }
        
        nlohmann::json j;
        file >> j;
        
        // Load metadata
        if (j.contains("metadata")) {
            auto meta = j["metadata"];
            if (meta.contains("agentDataType")) {
                m_dataType = static_cast<AgentDataType>(meta["agentDataType"].get<int>());
            }
        }
        
        spdlog::info("Training data loaded from {}", filename);
        
    } catch (const std::exception& e) {
        spdlog::error("Error loading training data: {}", e.what());
    }
}

int UnifiedDataCollector::getTotalTransitions() const {
    int total = 0;
    for (const auto& episode : m_episodeHistory) {
        total += static_cast<int>(episode.size());
    }
    return total;
}

float UnifiedDataCollector::getAverageScore(int recentEpisodes) const {
    if (m_episodeScores.empty()) return 0.0f;
    
    int startIdx = std::max(0, static_cast<int>(m_episodeScores.size()) - recentEpisodes);
    float sum = 0.0f;
    int count = 0;
    
    for (int i = startIdx; i < static_cast<int>(m_episodeScores.size()); ++i) {
        sum += m_episodeScores[i];
        count++;
    }
    
    return count > 0 ? sum / count : 0.0f;
}

float UnifiedDataCollector::getAverageReward(int recentEpisodes) const {
    if (m_episodeRewards.empty()) return 0.0f;
    
    int startIdx = std::max(0, static_cast<int>(m_episodeRewards.size()) - recentEpisodes);
    float sum = 0.0f;
    int count = 0;
    
    for (int i = startIdx; i < static_cast<int>(m_episodeRewards.size()); ++i) {
        sum += m_episodeRewards[i];
        count++;
    }
    
    return count > 0 ? sum / count : 0.0f;
}

void UnifiedDataCollector::ensureDirectoryExists(const std::string& path) {
    std::filesystem::create_directories(path);
}

std::string UnifiedDataCollector::getDefaultFilename() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "training_data_%Y%m%d_%H%M%S.json", &tm);
    
    return m_dataPath + std::string(buffer);
}