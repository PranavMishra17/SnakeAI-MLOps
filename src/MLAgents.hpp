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

// Base Agent Interface
class IAgent {
public:
    virtual ~IAgent() = default;
    virtual Direction getAction(const EnhancedState& state, bool training = true) = 0;
    virtual void updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) = 0;
    virtual void saveModel(const std::string& path) = 0;
    virtual void loadModel(const std::string& path) = 0;
    virtual float getEpsilon() const = 0;
    virtual void decayEpsilon() = 0;
    virtual std::string getAgentInfo() const = 0;
};

// Enhanced Q-Learning Agent
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
    
public:
    QLearningAgentEnhanced(float lr = 0.1f, float gamma = 0.95f, float eps = 0.1f);
    Direction getAction(const EnhancedState& state, bool training = true) override;
    void updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) override;
    void saveModel(const std::string& path) override;
    void loadModel(const std::string& path) override;
    float getEpsilon() const override;
    void decayEpsilon() override;
    std::string getAgentInfo() const override;
    
    // Training methods
    void startEpisode();
    void endEpisode();
    
private:
    Direction getMaxQAction(const AgentState& state) const;
    void updateQValue(const AgentState& state, Direction action, float reward, const AgentState& nextState);
};

// DQN Agent (Placeholder)
class DQNAgent : public IAgent {
private:
    struct NeuralNetwork {
        std::vector<std::vector<float>> weights;
        std::vector<float> biases;
        int inputSize = 20;
        int hiddenSize = 128;
        int outputSize = 4;
        
        NeuralNetwork();
        std::vector<float> forward(const std::vector<float>& input);
    };
    
    NeuralNetwork m_network;
    float m_epsilon;
    mutable std::mt19937 m_rng;
    
public:
    DQNAgent();
    Direction getAction(const EnhancedState& state, bool training = true) override;
    void updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) override {}
    void saveModel(const std::string& path) override;
    void loadModel(const std::string& path) override;
    float getEpsilon() const override;
    void decayEpsilon() override;
    std::string getAgentInfo() const override;
};

// Policy Gradient Agent (Placeholder)
class PolicyGradientAgent : public IAgent {
private:
    struct PolicyNetwork {
        std::vector<std::vector<float>> weights;
        int inputSize = 20;
        int hiddenSize = 64;
        int outputSize = 4;
        
        std::vector<float> forward(const std::vector<float>& input);
    };
    
    PolicyNetwork m_network;
    std::vector<float> m_episodeRewards;
    mutable std::mt19937 m_rng;
    
public:
    PolicyGradientAgent();
    Direction getAction(const EnhancedState& state, bool training = true) override;
    void updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) override;
    void saveModel(const std::string& path) override {}
    void loadModel(const std::string& path) override {}
    float getEpsilon() const override { return 0.0f; }
    void decayEpsilon() override {}
    std::string getAgentInfo() const override { return "Policy Gradient (Placeholder)"; }
};

// Agent Factory
class AgentFactory {
public:
    static std::unique_ptr<IAgent> createAgent(const AgentConfig& config);
};

// Enhanced State Generator
class StateGenerator {
public:
    static EnhancedState generateState(const Snake& snake, const Apple& apple, const Grid& grid);
    
private:
    static AgentState generateBasicState(const Snake& snake, const Apple& apple, const Grid& grid);
    static void calculateBodyDensity(const Snake& snake, const Grid& grid, float density[4]);
    static float calculatePathToFood(const Snake& snake, const Apple& apple, const Grid& grid);
};