#pragma once
#include <vector>
#include <deque>
#include <fstream>
#include <chrono>
#include <random>
#include <memory>
#include <nlohmann/json.hpp>
#include "GameState.hpp"

enum class AgentDataType {
    Q_LEARNING,
    NEURAL_NETWORK,
    GENETIC_ALGORITHM
};

struct UnifiedTransition {
    // Universal fields
    int episode;
    int step;
    float reward;
    bool terminal;
    std::chrono::milliseconds timestamp;
    
    // State representations
    AgentState basicState;
    EnhancedState enhancedState;
    
    // Action information
    int actionIndex;
    Direction actionDirection;
    
    // Neural network specific
    float logProbability = 0.0f;
    float valueEstimate = 0.0f;
    float advantage = 0.0f;
    
    // Meta information
    AgentType agentType;
    float epsilon;
    float learningRate;
};

struct QLearningTransition {
    std::string stateKey;
    int action;
    float reward;
    std::string nextStateKey;
    bool terminal;
};

struct DQNTransition {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> nextState;
    bool terminal;
    float priority = 1.0f;
};

struct TrainingMetrics {
    float averageScore = 0.0f;
    float maxScore = 0.0f;
    float scoreStandardDeviation = 0.0f;
    float averageEpisodeLength = 0.0f;
    float averageReward = 0.0f;
    float averageLoss = 0.0f;
    float explorationRate = 0.0f;
    float samplesPerSecond = 0.0f;
    float wallClockTime = 0.0f;
    int totalEpisodes = 0;
    int totalSteps = 0;
    std::map<std::string, float> agentSpecificMetrics;
};

template<typename T>
class RingBuffer {
private:
    std::vector<T> m_buffer;
    size_t m_capacity;
    size_t m_head;
    size_t m_size;
    
public:
    RingBuffer(size_t capacity) : m_capacity(capacity), m_head(0), m_size(0) {
        m_buffer.resize(capacity);
    }
    
    void push(const T& item) {
        m_buffer[m_head] = item;
        m_head = (m_head + 1) % m_capacity;
        if (m_size < m_capacity) m_size++;
    }
    
    T& operator[](size_t index) {
        return m_buffer[(m_head - m_size + index) % m_capacity];
    }
    
    std::vector<T> sample(size_t count) {
        if (count > m_size) count = m_size;
        std::vector<T> result;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, m_size - 1);
        
        for (size_t i = 0; i < count; ++i) {
            result.push_back((*this)[dist(gen)]);
        }
        return result;
    }
    
    void clear() { m_head = 0; m_size = 0; }
    size_t size() const { return m_size; }
    bool empty() const { return m_size == 0; }
    bool full() const { return m_size == m_capacity; }
};

class ExperienceReplayBuffer {
private:
    RingBuffer<DQNTransition> m_buffer;
    std::mt19937 m_rng;
    
public:
    ExperienceReplayBuffer(size_t capacity = 100000) 
        : m_buffer(capacity), m_rng(std::random_device{}()) {}
    
    void addTransition(const DQNTransition& transition) {
        m_buffer.push(transition);
    }
    
    std::vector<DQNTransition> sampleBatch(size_t batchSize) {
        return m_buffer.sample(batchSize);
    }
    
    size_t size() const { return m_buffer.size(); }
    bool canSample(size_t batchSize) const { return m_buffer.size() >= batchSize; }
};

class UnifiedDataCollector {
private:
    std::vector<UnifiedTransition> m_currentEpisode;
    std::vector<std::vector<UnifiedTransition>> m_episodeHistory;
    AgentDataType m_dataType;
    std::string m_dataPath;
    std::chrono::steady_clock::time_point m_episodeStart;
    std::chrono::steady_clock::time_point m_trainingStart;
    
    // Metrics tracking
    std::vector<float> m_episodeScores;
    std::vector<float> m_episodeRewards;
    std::vector<int> m_episodeLengths;
    int m_currentEpisode;
    
public:
    UnifiedDataCollector(const std::string& dataPath = "data/");
    ~UnifiedDataCollector();
    
    void setAgentType(AgentType agentType);
    void startEpisode(int episode);
    void recordTransition(const UnifiedTransition& transition);
    void endEpisode(int finalScore, bool died, float epsilon = 0.0f);
    
    // Data retrieval for different agent types
    std::vector<QLearningTransition> getQLearningData(int maxTransitions = -1);
    ExperienceReplayBuffer getDQNReplayBuffer();
    std::vector<std::vector<UnifiedTransition>> getPolicyGradientEpisodes(int maxEpisodes = -1);
    
    // Training metrics
    TrainingMetrics calculateMetrics() const;
    void saveTrainingData(const std::string& filename = "");
    void loadTrainingData(const std::string& filename);
    
    // Real-time access
    const std::vector<UnifiedTransition>& getCurrentEpisode() const { return m_currentEpisode; }
    const std::vector<std::vector<UnifiedTransition>>& getEpisodeHistory() const { return m_episodeHistory; }
    
    // Statistics
    int getTotalEpisodes() const { return m_episodeHistory.size(); }
    int getTotalTransitions() const;
    float getAverageScore(int recentEpisodes = 100) const;
    float getAverageReward(int recentEpisodes = 100) const;
    
private:
    void ensureDirectoryExists(const std::string& path);
    std::string getDefaultFilename() const;
};