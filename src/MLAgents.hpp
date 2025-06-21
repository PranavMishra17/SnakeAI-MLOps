#pragma once
#include "GameState.hpp"
#include "Snake.hpp"
#include "Apple.hpp"
#include "Grid.hpp"
#include <memory>
#include <random>
#include <vector>
#include <map>
#include <fstream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include "TorchInference.hpp"

// Base Agent Interface
class IAgent {
public:
    virtual ~IAgent() = default;
    virtual Direction getAction(const EnhancedState& state, bool training = true) = 0;
    virtual void updateAgent(const EnhancedState &state, Direction action, float reward, const EnhancedState &nextState) = 0;
    virtual void saveModel(const std::string& path) = 0;
    virtual bool loadModel(const std::string& path) = 0;
    virtual float getEpsilon() const = 0;
    virtual void decayEpsilon() = 0;
    virtual std::string getAgentInfo() const = 0;
    virtual std::string getModelInfo() const { return "No model info available"; }
};

// Enhanced Model Performance Data
struct ModelPerformanceData {
    float bestScore = 0.0f;
    float averageScore = 0.0f;
    float consistency = 0.0f;
    float efficiency = 0.0f;
    int totalEpisodes = 0;
    int trainingDuration = 0; // seconds
    float convergenceRate = 0.0f;
    float explorationDecay = 0.0f;
    std::string trainingProfile = "unknown";
    std::string analysisImagePath = "";
    
    // Detailed metrics
    float maxReward = 0.0f;
    float avgEpisodeLength = 0.0f;
    float successRate = 0.0f; // percentage of episodes with score > 5
    float improvementRate = 0.0f; // score improvement over time
};

// Enhanced Trained Model Information
struct TrainedModelInfo {
    std::string name;
    std::string profile;
    std::string modelPath;
    std::string description;
    std::string modelType;  // "qlearning", "dqn", "ppo", "actor_critic"
    bool isLoaded = false;
    
    // Enhanced performance data
    ModelPerformanceData performance;
    
    // Training metadata
    std::string trainingDate;
    std::string version;
    std::map<std::string, float> hyperparameters;
    
    static TrainedModelInfo fromFile(const std::string& infoPath);
    void saveToFile(const std::string& infoPath) const;
    
    // Display helpers
    std::string getFormattedPerformance() const {
        return "Best: " + std::to_string(static_cast<int>(performance.bestScore)) + 
               " | Avg: " + std::to_string(performance.averageScore).substr(0, 4) +
               " | Episodes: " + std::to_string(performance.totalEpisodes);
    }
    
    std::string getDetailedStats() const {
        return "Consistency: " + std::to_string(performance.consistency).substr(0, 4) +
               " | Success Rate: " + std::to_string(performance.successRate).substr(0, 4) + "%" +
               " | Efficiency: " + std::to_string(performance.efficiency).substr(0, 4);
    }
};

// Enhanced Q-Learning Agent with trained model support
class QLearningAgentEnhanced : public IAgent {
private:
    std::map<std::string, std::array<float, 4>> m_qTable;
    float m_learningRate;
    float m_discountFactor;
    float m_epsilon;
    mutable std::mt19937 m_rng;
    
    // Training state
    AgentState m_lastState;
    Direction m_lastAction;
    bool m_hasLastState;
    
    // Model information
    TrainedModelInfo m_modelInfo;
    bool m_isPreTrained;
    
public:
    QLearningAgentEnhanced(float lr = 0.1f, float gamma = 0.95f, float eps = 0.1f);
    QLearningAgentEnhanced(const TrainedModelInfo& modelInfo);
    
    Direction getAction(const EnhancedState& state, bool training = true) override;
    void updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) override;
    void saveModel(const std::string&) override {}  // Keep as empty inline
    bool loadModel(const std::string& path) override;
    float getEpsilon() const override;
    void decayEpsilon() override;
    std::string getAgentInfo() const override;
    std::string getModelInfo() const override;
    
    // Training methods
    void startEpisode();
    void endEpisode();
    
    // Model management
    bool loadTrainedModel(const std::string& modelPath);
    const TrainedModelInfo& getModelInformation() const { return m_modelInfo; }
    bool isPreTrained() const { return m_isPreTrained; }
    int getQTableSize() const { return m_qTable.size(); }
    
    // Performance info
    std::string getBestScoreInfo() const {
        if (m_isPreTrained) {
            return "Best Score: " + std::to_string(static_cast<int>(m_modelInfo.performance.bestScore));
        }
        return "Training from scratch";
    }
    
private:
    Direction getMaxQAction(const AgentState& state) const;
    void updateQValue(const AgentState& state, Direction action, float reward, const AgentState& nextState);
    std::string encodeState9Bit(const AgentState& state) const;
    AgentState decodeState9Bit(const std::string& stateStr) const;
};


// Updated DQN Agent with proper .pt loading
class DQNAgent : public IAgent {
private:
    float m_epsilon;
    mutable std::mt19937 m_rng;
    TrainedModelInfo m_modelInfo;
    std::unique_ptr<DQNInference> m_torchModel;
    
public:
    DQNAgent();
    DQNAgent(const TrainedModelInfo& modelInfo);
    
    Direction getAction(const EnhancedState& state, bool training = true) override;
    bool loadModel(const std::string& path) override;
    std::string getModelInfo() const override;
    std::string getAgentInfo() const override;
    
    // Required interface methods
    void updateAgent(const EnhancedState&, Direction, float, const EnhancedState&) override;
    void saveModel(const std::string&) override {}  // Keep as empty inline
    float getEpsilon() const override { return m_epsilon; }
    void decayEpsilon() override { m_epsilon *= 0.995f; }

private:
    Direction makeIntelligentDecision(const EnhancedState& state);
};

// Updated PPO Agent with proper .pt loading
class PPOAgent : public IAgent {
private:
    mutable std::mt19937 m_rng;
    TrainedModelInfo m_modelInfo;
    std::unique_ptr<PPOInference> m_torchModel;
    
public:
    PPOAgent();
    PPOAgent(const TrainedModelInfo& modelInfo);
    
    Direction getAction(const EnhancedState& state, bool training = true) override;
    bool loadModel(const std::string& path) override;
    std::string getModelInfo() const override;
    std::string getAgentInfo() const override;
    
    // Required interface methods
    void updateAgent(const EnhancedState&, Direction, float, const EnhancedState&) override;
   void saveModel(const std::string&) override {}  // Keep as empty inline
    float getEpsilon() const override { return 0.0f; }
    void decayEpsilon() override {}
};

// Updated Actor-Critic Agent with proper .pt loading
class ActorCriticAgent : public IAgent {
private:
    mutable std::mt19937 m_rng;
    TrainedModelInfo m_modelInfo;
    std::unique_ptr<ActorCriticInference> m_torchModel;
    
public:
    ActorCriticAgent();
    ActorCriticAgent(const TrainedModelInfo& modelInfo);
    
    Direction getAction(const EnhancedState& state, bool training = true) override;
    float getStateValue(const EnhancedState& state);
    bool loadModel(const std::string& path) override;
    std::string getModelInfo() const override;
    std::string getAgentInfo() const override;
    
    // Required interface methods
    void updateAgent(const EnhancedState&, Direction, float, const EnhancedState&) override;
    void saveModel(const std::string&) override {}  // Keep as empty inline
    float getEpsilon() const override { return 0.0f; }
    void decayEpsilon() override {}
};

// Enhanced Evaluation Report Data
struct EvaluationReportData {
    std::map<std::string, ModelPerformanceData> modelPerformance;
    std::string reportDate;
    std::string version;
    
    // Analysis images
    std::map<std::string, std::string> analysisImages; // model_name -> image_path
    std::string comparisonImagePath;
    
    // Summary statistics
    std::string bestOverallModel;
    std::string mostConsistentModel;
    std::string fastestLearner;
    
    static EvaluationReportData loadFromFile(const std::string& reportPath);
};

// Enhanced Trained Model Manager with Performance Data
class TrainedModelManager {
private:
    std::vector<TrainedModelInfo> m_availableModels;
    std::string m_modelsDirectory;
    EvaluationReportData m_evaluationData;
    
public:
    TrainedModelManager(const std::string& modelsDir = "models/");
    
    void scanForModels();
    void loadEvaluationReport(const std::string& reportPath = "models/enhanced_evaluation_report_fixed.json");
    void createModelInfoFiles();
    
    std::vector<TrainedModelInfo> getAvailableModels() const { return m_availableModels; }
    TrainedModelInfo* findModel(const std::string& modelName);
    const EvaluationReportData& getEvaluationData() const { return m_evaluationData; }
    
    // Enhanced model queries
    std::vector<TrainedModelInfo> getTopPerformingModels(int count = 5) const;
    std::vector<TrainedModelInfo> getModelsByType(const std::string& modelType) const;
    TrainedModelInfo* getBestModelOfType(const std::string& modelType);
    
    // Model validation
    bool validateQlearningModel(const std::string& modelPath) const;
    
    // Statistics
    size_t getModelCount() const { return m_availableModels.size(); }
    size_t getQlearningModelCount() const { return getModelsByType("qlearning").size(); }
    size_t getNeuralNetworkModelCount() const { 
        return getModelsByType("dqn").size() + 
               getModelsByType("ppo").size() +
               getModelsByType("actor_critic").size(); 
    }
    
    // Performance insights
    std::string getPerformanceSummary() const;
    std::vector<std::pair<std::string, float>> getLeaderboardData() const;
};

// Enhanced Agent Factory with Performance Data
class AgentFactory {
public:
    static std::unique_ptr<IAgent> createAgent(const AgentConfig& config);
    static std::unique_ptr<IAgent> createTrainedAgent(const std::string& modelName);
    static std::unique_ptr<IAgent> createAgentWithPerformanceData(const TrainedModelInfo& modelInfo);
    static std::vector<AgentConfig> getAvailableTrainedAgents();
    
    // Create agents by type with performance data
    static std::unique_ptr<IAgent> createQLearningAgent(const std::string& profile = "balanced");
    static std::unique_ptr<IAgent> createDQNAgent(const std::string& profile = "balanced");
    static std::unique_ptr<IAgent> createPPOAgent(const std::string& profile = "balanced");
    static std::unique_ptr<IAgent> createActorCriticAgent(const std::string& profile = "balanced");
    
    // Utility functions
    static bool isModelTypeSupported(const std::string& modelType) {
        return modelType == "qlearning" || modelType == "dqn" || 
               modelType == "ppo" || modelType == "actor_critic";
    }
    
    static bool isFullyImplemented(const std::string& modelType) {
        return modelType == "qlearning";
    }
    
    static std::string getSupportedTechniques() {
        return "Q-Learning (full C++ support), DQN/PPO/Actor-Critic (intelligent C++ placeholders)";
    }
};

// Enhanced State Generator (unchanged interface)
class StateGenerator {
public:
    static EnhancedState generateState(const Snake& snake, const Apple& apple, const Grid& grid);
    static EnhancedState generateEnhancedState(const Snake& snake, const Apple& apple, const Grid& grid, int episode = 0);
    
private:
    static AgentState generateBasicState(const Snake& snake, const Apple& apple, const Grid& grid);
    static void calculateBodyDensity(const Snake& snake, const Grid& grid, float density[4]);
    static float calculatePathToFood(const Snake& snake, const Apple& apple, const Grid& grid);
    static float calculateSnakeEfficiency(const Snake& snake, int episode);
    static float calculateEnvironmentComplexity(const Snake& snake, const Grid& grid);
};