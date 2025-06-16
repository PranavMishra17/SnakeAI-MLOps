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

// Base Agent Interface
class IAgent {
public:
    virtual ~IAgent() = default;
    virtual Direction getAction(const EnhancedState& state, bool training = true) = 0;
    virtual void updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) = 0;
    virtual void saveModel(const std::string& path) = 0;
    virtual bool loadModel(const std::string& path) = 0;
    virtual float getEpsilon() const = 0;
    virtual void decayEpsilon() = 0;
    virtual std::string getAgentInfo() const = 0;
    virtual std::string getModelInfo() const { return "No model info available"; }
};

// Trained Model Information (Enhanced)
struct TrainedModelInfo {
    std::string name;
    std::string profile;
    std::string modelPath;
    std::string description;
    std::string modelType;  // "qlearning", "dqn", "policy_gradient", "actor_critic"
    float averageScore = 0.0f;
    int episodesTrained = 0;
    bool isLoaded = false;
    
    static TrainedModelInfo fromFile(const std::string& infoPath);
    void saveToFile(const std::string& infoPath) const;
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
    void saveModel(const std::string& path) override;
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
    
private:
    Direction getMaxQAction(const AgentState& state) const;
    void updateQValue(const AgentState& state, Direction action, float reward, const AgentState& nextState);
    std::string encodeState9Bit(const AgentState& state) const;
    AgentState decodeState9Bit(const std::string& stateStr) const;
};

// Enhanced DQN Agent with Intelligent Behavior
class DQNAgent : public IAgent {
private:
    float m_epsilon;
    mutable std::mt19937 m_rng;
    
    // Intelligent decision making
    Direction makeIntelligentDecision(const EnhancedState& state);
    
public:
    DQNAgent();
    Direction getAction(const EnhancedState& state, bool training = true) override;
    void updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) override {}
    void saveModel(const std::string& path) override;
    bool loadModel(const std::string& path) override;
    float getEpsilon() const override { return m_epsilon; }
    void decayEpsilon() override { m_epsilon *= 0.995f; }
    std::string getAgentInfo() const override;
    std::string getModelInfo() const override;
};

// Enhanced Policy Gradient Agent with Probabilistic Behavior
class PolicyGradientAgent : public IAgent {
private:
    mutable std::mt19937 m_rng;
    
public:
    PolicyGradientAgent();
    Direction getAction(const EnhancedState& state, bool training = true) override;
    void updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) override {}
    void saveModel(const std::string& path) override {}
    bool loadModel(const std::string& path) override;
    float getEpsilon() const override { return 0.0f; }
    void decayEpsilon() override {}
    std::string getAgentInfo() const override;
    std::string getModelInfo() const override;
};

// Enhanced Actor-Critic Agent with Value-Based Decisions
class ActorCriticAgent : public IAgent {
private:
    mutable std::mt19937 m_rng;
    
public:
    ActorCriticAgent();
    Direction getAction(const EnhancedState& state, bool training = true) override;
    void updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) override {}
    void saveModel(const std::string& path) override {}
    bool loadModel(const std::string& path) override;
    float getEpsilon() const override { return 0.0f; }
    void decayEpsilon() override {}
    std::string getAgentInfo() const override;
    std::string getModelInfo() const override;
};

// Enhanced Trained Model Manager with Multi-Technique Support
class TrainedModelManager {
private:
    std::vector<TrainedModelInfo> m_availableModels;
    std::string m_modelsDirectory;
    
public:
    TrainedModelManager(const std::string& modelsDir = "models/");
    
    void scanForModels();
    void createModelInfoFiles();
    std::vector<TrainedModelInfo> getAvailableModels() const { return m_availableModels; }
    TrainedModelInfo* findModel(const std::string& modelName);
    
    // Model validation (Q-Learning only, neural networks require Python)
    bool validateQlearningModel(const std::string& modelPath) const;
    
    // Get models by type
    std::vector<TrainedModelInfo> getModelsByType(const std::string& modelType) const {
        std::vector<TrainedModelInfo> filtered;
        for (const auto& model : m_availableModels) {
            if (model.modelType == modelType) {
                filtered.push_back(model);
            }
        }
        return filtered;
    }
    
    // Statistics
    size_t getModelCount() const { return m_availableModels.size(); }
    size_t getQlearningModelCount() const { return getModelsByType("qlearning").size(); }
    size_t getNeuralNetworkModelCount() const { 
        return getModelsByType("dqn").size() + 
               getModelsByType("policy_gradient").size() + 
               getModelsByType("actor_critic").size(); 
    }
};

// Enhanced Agent Factory with Multi-Technique Support
class AgentFactory {
public:
    static std::unique_ptr<IAgent> createAgent(const AgentConfig& config);
    static std::unique_ptr<IAgent> createTrainedAgent(const std::string& modelName);
    static std::vector<AgentConfig> getAvailableTrainedAgents();
    
    // Create agents by type
    static std::unique_ptr<IAgent> createQLearningAgent(const std::string& profile = "balanced");
    static std::unique_ptr<IAgent> createDQNAgent(const std::string& profile = "balanced");
    static std::unique_ptr<IAgent> createPolicyGradientAgent(const std::string& profile = "balanced");
    static std::unique_ptr<IAgent> createActorCriticAgent(const std::string& profile = "balanced");
    
    // Utility functions
    static bool isModelTypeSupported(const std::string& modelType) {
        return modelType == "qlearning" || modelType == "dqn" || 
               modelType == "policy_gradient" || modelType == "actor_critic";
    }
    
    static bool isFullyImplemented(const std::string& modelType) {
        return modelType == "qlearning"; // Only Q-Learning fully implemented in C++
    }
    
    static std::string getSupportedTechniques() {
        return "Q-Learning (full C++ support), DQN/Policy Gradient/Actor-Critic (intelligent C++ placeholders)";
    }
};

// Enhanced State Generator
class StateGenerator {
public:
    static EnhancedState generateState(const Snake& snake, const Apple& apple, const Grid& grid);
    
    // Enhanced state generation with additional features
    static EnhancedState generateEnhancedState(const Snake& snake, const Apple& apple, const Grid& grid, int episode = 0);
    
private:
    static AgentState generateBasicState(const Snake& snake, const Apple& apple, const Grid& grid);
    static void calculateBodyDensity(const Snake& snake, const Grid& grid, float density[4]);
    static float calculatePathToFood(const Snake& snake, const Apple& apple, const Grid& grid);
    
    // Additional feature extraction methods
    static float calculateSnakeEfficiency(const Snake& snake, int episode);
    static float calculateEnvironmentComplexity(const Snake& snake, const Grid& grid);
    static std::vector<float> calculateSpatialFeatures(const Snake& snake, const Apple& apple, const Grid& grid);
};

// Model Performance Tracker
class ModelPerformanceTracker {
private:
    struct PerformanceMetrics {
        std::vector<float> scores;
        std::vector<float> episodeLengths;
        std::vector<float> efficiencyRatios;
        float averageScore = 0.0f;
        float maxScore = 0.0f;
        float consistency = 0.0f; // Lower std dev = higher consistency
        std::string modelType;
        std::string modelName;
    };
    
    std::map<std::string, PerformanceMetrics> m_modelMetrics;
    
public:
    void recordPerformance(const std::string& modelName, const std::string& modelType, 
                          float score, float episodeLength, float efficiency);
    
    void generatePerformanceReport(const std::string& outputPath) const;
    PerformanceMetrics getMetrics(const std::string& modelName) const;
    
    // Comparison methods
    std::vector<std::pair<std::string, float>> getRankingByScore() const;
    std::vector<std::pair<std::string, float>> getRankingByConsistency() const;
    std::string getBestModelByType(const std::string& modelType) const;
};

// Neural Network Integration Interface (Future Expansion)
class NeuralNetworkInterface {
public:
    virtual ~NeuralNetworkInterface() = default;
    
    // Methods for when LibTorch/ONNX integration is added
    virtual bool loadONNXModel(const std::string& modelPath) = 0;
    virtual std::vector<float> inference(const std::vector<float>& input) = 0;
    virtual bool isModelLoaded() const = 0;
    
    // Placeholder for future implementation
    static std::unique_ptr<NeuralNetworkInterface> create(const std::string& framework = "onnx");
};

// Configuration for neural network models
struct NeuralNetworkConfig {
    std::string modelPath;
    std::string framework; // "onnx", "libtorch", etc.
    bool useGPU = false;
    int batchSize = 1;
    float inferenceThreshold = 0.5f;
    
    // Model-specific parameters
    std::map<std::string, float> parameters;
};

// Comprehensive model evaluation results
struct ModelEvaluationResult {
    std::string modelName;
    std::string modelType;
    std::string profile;
    float averageScore;
    float maxScore;
    float consistency;
    float actionEntropy;
    int episodesTested;
    std::string evaluationDate;
    std::map<std::string, float> additionalMetrics;
};

// Batch model evaluator for C++ (Q-Learning + intelligent placeholders)
class BatchModelEvaluator {
public:
    static std::vector<ModelEvaluationResult> evaluateAllModels(int episodesPerModel = 100);
    static ModelEvaluationResult evaluateModel(const std::string& modelPath, 
                                             const std::string& modelType, 
                                             int episodes = 100);
    static void generateComparisonReport(const std::vector<ModelEvaluationResult>& results, 
                                       const std::string& outputPath);
};