#include "MLAgents.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <algorithm>

// TrainedModelInfo Implementation
TrainedModelInfo TrainedModelInfo::fromFile(const std::string& infoPath) {
    TrainedModelInfo info;
    
    try {
        std::ifstream file(infoPath);
        if (!file.is_open()) {
            spdlog::warn("TrainedModelInfo: Could not load info file: {}", infoPath);
            return info;
        }
        
        nlohmann::json j;
        file >> j;
        
        info.name = j.value("name", "Unknown Model");
        info.profile = j.value("profile", "unknown");
        info.modelPath = j.value("modelPath", "");
        info.description = j.value("description", "No description available");
        info.averageScore = j.value("averageScore", 0.0f);
        info.episodesTrained = j.value("episodesTrained", 0);
        info.modelType = j.value("modelType", "qlearning");
        info.isLoaded = false;
        
        spdlog::info("TrainedModelInfo: Loaded model info: {} (type: {}, avg score: {:.2f})", 
                     info.name, info.modelType, info.averageScore);
        
    } catch (const std::exception& e) {
        spdlog::error("TrainedModelInfo: Error loading info file {}: {}", infoPath, e.what());
    }
    
    return info;
}

void TrainedModelInfo::saveToFile(const std::string& infoPath) const {
    try {
        nlohmann::json j;
        j["name"] = name;
        j["profile"] = profile;
        j["modelPath"] = modelPath;
        j["description"] = description;
        j["averageScore"] = averageScore;
        j["episodesTrained"] = episodesTrained;
        j["modelType"] = modelType;
        j["createdDate"] = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        std::ofstream file(infoPath);
        if (file.is_open()) {
            file << j.dump(4);
            spdlog::info("TrainedModelInfo: Saved model info: {}", infoPath);
        }
    } catch (const std::exception& e) {
        spdlog::error("TrainedModelInfo: Error saving info file {}: {}", infoPath, e.what());
    }
}

// Enhanced Q-Learning Agent Implementation
QLearningAgentEnhanced::QLearningAgentEnhanced(float lr, float gamma, float eps)
    : m_learningRate(lr), m_discountFactor(gamma), m_epsilon(eps), 
      m_rng(std::random_device{}()), m_hasLastState(false), m_isPreTrained(false) {
    spdlog::info("QLearningAgentEnhanced: Initialized with lr={}, gamma={}, epsilon={}", lr, gamma, eps);
    
    m_modelInfo.name = "Q-Learning Agent";
    m_modelInfo.profile = "scratch";
    m_modelInfo.description = "Fresh Q-Learning agent training from scratch";
    m_modelInfo.modelType = "qlearning";
}

QLearningAgentEnhanced::QLearningAgentEnhanced(const TrainedModelInfo& modelInfo)
    : m_learningRate(0.1f), m_discountFactor(0.95f), m_epsilon(0.02f),
      m_rng(std::random_device{}()), m_hasLastState(false), m_modelInfo(modelInfo), m_isPreTrained(false) {
    
    spdlog::info("QLearningAgentEnhanced: Creating agent with pre-trained model: {}", modelInfo.name);
    
    if (loadTrainedModel(modelInfo.modelPath)) {
        m_isPreTrained = true;
        spdlog::info("QLearningAgentEnhanced: Successfully loaded pre-trained model with {} states", 
                     m_qTable.size());
    } else {
        spdlog::error("QLearningAgentEnhanced: Failed to load pre-trained model: {}", modelInfo.modelPath);
    }
}

std::string QLearningAgentEnhanced::encodeState9Bit(const AgentState& state) const {
    std::string binary;
    
    binary += state.dangerStraight ? "1" : "0";
    binary += state.dangerLeft ? "1" : "0"; 
    binary += state.dangerRight ? "1" : "0";
    
    switch (state.currentDirection) {
        case Direction::UP:    binary += "00"; break;
        case Direction::DOWN:  binary += "01"; break;
        case Direction::LEFT:  binary += "10"; break;
        case Direction::RIGHT: binary += "11"; break;
    }
    
    binary += state.foodLeft ? "1" : "0";
    binary += state.foodRight ? "1" : "0";
    binary += state.foodUp ? "1" : "0";
    binary += state.foodDown ? "1" : "0";
    
    return binary;
}

Direction QLearningAgentEnhanced::getAction(const EnhancedState& state, bool training) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    if (training && dist(m_rng) < m_epsilon) {
        std::uniform_int_distribution<int> actionDist(0, 3);
        Direction randomAction = static_cast<Direction>(actionDist(m_rng));
        spdlog::debug("QLearningAgent: Random action (epsilon={:.3f}): {}", m_epsilon, static_cast<int>(randomAction));
        return randomAction;
    }
    
    Direction greedyAction = getMaxQAction(state.basic);
    spdlog::debug("QLearningAgent: Greedy action: {}", static_cast<int>(greedyAction));
    return greedyAction;
}

Direction QLearningAgentEnhanced::getMaxQAction(const AgentState& state) const {
    std::string stateKey = encodeState9Bit(state);
    
    if (m_qTable.find(stateKey) == m_qTable.end()) {
        std::uniform_int_distribution<int> dist(0, 3);
        Direction randomAction = static_cast<Direction>(dist(m_rng));
        spdlog::debug("QLearningAgent: Unseen state, random action: {}", static_cast<int>(randomAction));
        return randomAction;
    }
    
    const auto& actions = m_qTable.at(stateKey);
    int maxIdx = std::distance(actions.begin(), std::max_element(actions.begin(), actions.end()));
    
    spdlog::debug("QLearningAgent: Max Q action for state {}: {} (Q={:.3f})", 
                 stateKey, maxIdx, actions[maxIdx]);
    
    return static_cast<Direction>(maxIdx);
}

bool QLearningAgentEnhanced::loadTrainedModel(const std::string& modelPath) {
    try {
        std::ifstream file(modelPath);
        if (!file.is_open()) {
            spdlog::error("QLearningAgent: Cannot open model file: {}", modelPath);
            return false;
        }
        
        nlohmann::json j;
        file >> j;
        
        m_qTable.clear();
        
        if (j.contains("qTable")) {
            for (auto& [state, actions] : j["qTable"].items()) {
                if (state.length() != 9 || 
                    std::any_of(state.begin(), state.end(), [](char c) { return c != '0' && c != '1'; })) {
                    spdlog::warn("QLearningAgent: Invalid state key format: {}", state);
                    continue;
                }
                
                std::array<float, 4> actionValues;
                for (size_t i = 0; i < 4 && i < actions.size(); ++i) {
                    actionValues[i] = actions[i];
                }
                m_qTable[state] = actionValues;
            }
        }
        
        if (j.contains("hyperparameters")) {
            auto params = j["hyperparameters"];
            if (params.contains("learningRate")) {
                m_learningRate = params["learningRate"];
            }
            if (params.contains("discountFactor")) {
                m_discountFactor = params["discountFactor"];
            }
            if (params.contains("epsilon")) {
                m_epsilon = params["epsilon"];
            }
        }
        
        m_modelInfo.isLoaded = true;
        spdlog::info("QLearningAgent: Successfully loaded model from {} with {} states", 
                     modelPath, m_qTable.size());
        
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("QLearningAgent: Error loading model from {}: {}", modelPath, e.what());
        return false;
    }
}

bool QLearningAgentEnhanced::loadModel(const std::string& path) {
    return loadTrainedModel(path);
}

float QLearningAgentEnhanced::getEpsilon() const {
    return m_epsilon;
}

void QLearningAgentEnhanced::decayEpsilon() {
    m_epsilon = std::max(0.02f, m_epsilon * 0.995f);
}

std::string QLearningAgentEnhanced::getAgentInfo() const {
    std::ostringstream oss;
    oss << "Q-Learning | States: " << getQTableSize() 
        << " | Epsilon: " << std::fixed << std::setprecision(3) << m_epsilon;
    return oss.str();
}

std::string QLearningAgentEnhanced::getModelInfo() const {
    if (m_isPreTrained) {
        return "Pre-trained: " + m_modelInfo.name + " (" + m_modelInfo.profile + ")";
    }
    return "Fresh Q-Learning agent (training from scratch)";
}

void QLearningAgentEnhanced::startEpisode() {
    m_hasLastState = false;
    spdlog::debug("QLearningAgentEnhanced: Starting new episode");
}

void QLearningAgentEnhanced::updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) {
    if (m_hasLastState) {
        updateQValue(m_lastState, m_lastAction, reward, state.basic);
    }
    m_lastState = state.basic;
    m_lastAction = action;
    m_hasLastState = true;
}

void QLearningAgentEnhanced::saveModel(const std::string& path) {
    try {
        nlohmann::json j;
        j["qTable"] = nlohmann::json::object();
        
        // Save Q-table entries
        for (const auto& [state, actions] : m_qTable) {
            j["qTable"][state] = actions;
        }
        
        // Save hyperparameters
        j["hyperparameters"] = {
            {"learningRate", m_learningRate},
            {"discountFactor", m_discountFactor}, 
            {"epsilon", m_epsilon}
        };
        
        std::ofstream file(path);
        if (file.is_open()) {
            file << j.dump(4);
            spdlog::info("QLearningAgent: Model saved to {}", path);
        }
    } catch (const std::exception& e) {
        spdlog::error("QLearningAgent: Failed to save model: {}", e.what());
    }
}

void QLearningAgentEnhanced::updateQValue(const AgentState& state, Direction action, float reward, const AgentState& nextState) {
    std::string stateKey = encodeState9Bit(state);
    std::string nextStateKey = encodeState9Bit(nextState);
    
    // Initialize if needed
    if (m_qTable.find(stateKey) == m_qTable.end()) {
        m_qTable[stateKey] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    if (m_qTable.find(nextStateKey) == m_qTable.end()) {
        m_qTable[nextStateKey] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    
    int actionIdx = static_cast<int>(action);
    float currentQ = m_qTable[stateKey][actionIdx];
    float maxNextQ = *std::max_element(m_qTable[nextStateKey].begin(), m_qTable[nextStateKey].end());
    
    // Q-learning update
    float newQ = currentQ + m_learningRate * (reward + m_discountFactor * maxNextQ - currentQ);
    m_qTable[stateKey][actionIdx] = newQ;
}

// Enhanced DQN Agent with Intelligent Behavior
DQNAgent::DQNAgent() : m_epsilon(0.05f), m_rng(std::random_device{}()) {
    spdlog::info("DQNAgent: Initialized (enhanced intelligent placeholder)");
    spdlog::info("DQNAgent: Using sophisticated heuristics for gameplay");
}

Direction DQNAgent::getAction(const EnhancedState& state, bool training) {
    // Enhanced intelligent behavior - not just random
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    // Small chance of exploration
    if (training && dist(m_rng) < m_epsilon) {
        std::uniform_int_distribution<int> actionDist(0, 3);
        return static_cast<Direction>(actionDist(m_rng));
    }
    
    // Intelligent decision making based on state
    return makeIntelligentDecision(state);
}

Direction DQNAgent::makeIntelligentDecision(const EnhancedState& state) {
    const auto& basic = state.basic;
    
    // Priority 1: Avoid immediate danger
    std::vector<Direction> safeActions;
    if (!basic.dangerStraight) safeActions.push_back(basic.currentDirection);
    if (!basic.dangerLeft) {
        // Calculate left direction
        Direction leftDir = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 3) % 4);
        safeActions.push_back(leftDir);
    }
    if (!basic.dangerRight) {
        // Calculate right direction
        Direction rightDir = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 1) % 4);
        safeActions.push_back(rightDir);
    }
    
    // If no safe actions, try opposite direction as last resort
    if (safeActions.empty()) {
        Direction opposite = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 2) % 4);
        safeActions.push_back(opposite);
    }
    
    // Priority 2: Among safe actions, choose one that moves toward food
    Direction bestAction = safeActions[0];
    
    for (Direction action : safeActions) {
        bool movesTowardFood = false;
        
        switch (action) {
            case Direction::UP:
                movesTowardFood = basic.foodUp;
                break;
            case Direction::DOWN:
                movesTowardFood = basic.foodDown;
                break;
            case Direction::LEFT:
                movesTowardFood = basic.foodLeft;
                break;
            case Direction::RIGHT:
                movesTowardFood = basic.foodRight;
                break;
        }
        
        if (movesTowardFood) {
            bestAction = action;
            break;
        }
    }
    
    spdlog::debug("DQNAgent: Intelligent decision - action: {}", static_cast<int>(bestAction));
    return bestAction;
}

bool DQNAgent::loadModel(const std::string& path) {
    spdlog::info("DQNAgent: Model file detected: {}", path);
    spdlog::info("DQNAgent: Using enhanced heuristics instead of neural network inference");
    return true; // Always return true for intelligent placeholder
}

void DQNAgent::saveModel(const std::string& path) {
    spdlog::info("DQNAgent: Intelligent placeholder - no model to save");
}

std::string DQNAgent::getAgentInfo() const {
    return "DQN (Intelligent Heuristics) | Safety-first with food seeking";
}

std::string DQNAgent::getModelInfo() const {
    return "Enhanced DQN placeholder - uses intelligent decision making";
}

// Enhanced PPO Agent Implementation (replaces PolicyGradientAgent)
PPOAgent::PPOAgent() : m_rng(std::random_device{}()) {
    spdlog::info("PPOAgent: Initialized (enhanced intelligent placeholder)");
    spdlog::info("PPOAgent: Using probabilistic decision making with PPO-style exploration");
}

Direction PPOAgent::getAction(const EnhancedState& state, bool training) {
    // PPO behavior - more exploratory but still intelligent
    const auto& basic = state.basic;
    
    // Calculate action probabilities based on state (PPO-style)
    std::array<float, 4> actionProbs = {0.25f, 0.25f, 0.25f, 0.25f}; // Base equal probability
    
    // Reduce probability of dangerous actions
    if (basic.dangerStraight) {
        actionProbs[static_cast<int>(basic.currentDirection)] *= 0.1f;
    }
    if (basic.dangerLeft) {
        int leftIdx = (static_cast<int>(basic.currentDirection) + 3) % 4;
        actionProbs[leftIdx] *= 0.1f;
    }
    if (basic.dangerRight) {
        int rightIdx = (static_cast<int>(basic.currentDirection) + 1) % 4;
        actionProbs[rightIdx] *= 0.1f;
    }
    
    // Increase probability of food-seeking actions
    if (basic.foodUp) actionProbs[0] *= 2.0f;
    if (basic.foodDown) actionProbs[1] *= 2.0f;
    if (basic.foodLeft) actionProbs[2] *= 2.0f;
    if (basic.foodRight) actionProbs[3] *= 2.0f;
    
    // Normalize probabilities
    float sum = actionProbs[0] + actionProbs[1] + actionProbs[2] + actionProbs[3];
    for (auto& prob : actionProbs) {
        prob /= sum;
    }
    
    // Sample action based on probabilities
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sample = dist(m_rng);
    float cumulative = 0.0f;
    
    for (int i = 0; i < 4; ++i) {
        cumulative += actionProbs[i];
        if (sample <= cumulative) {
            return static_cast<Direction>(i);
        }
    }
    
    return static_cast<Direction>(0); // Fallback
}

bool PPOAgent::loadModel(const std::string& path) {
    spdlog::info("PPOAgent: Model file detected: {}", path);
    spdlog::info("PPOAgent: Using probabilistic decision making");
    return true;
}

std::string PPOAgent::getAgentInfo() const {
    return "PPO (Probabilistic) | Proximal Policy Optimization style decisions";
}

std::string PPOAgent::getModelInfo() const {
    return "Enhanced PPO placeholder - probabilistic action selection";
}

// Enhanced Actor-Critic Agent
ActorCriticAgent::ActorCriticAgent() : m_rng(std::random_device{}()) {
    spdlog::info("ActorCriticAgent: Initialized (enhanced intelligent placeholder)");
}

Direction ActorCriticAgent::getAction(const EnhancedState& state, bool training) {
    // Actor-critic behavior - balanced exploration and exploitation
    const auto& basic = state.basic;
    
    // Calculate value estimates for each action (critic simulation)
    std::array<float, 4> actionValues = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Base values
    for (int i = 0; i < 4; ++i) {
        actionValues[i] = 0.5f; // Neutral value
    }
    
    // Penalize dangerous actions heavily
    if (basic.dangerStraight) {
        actionValues[static_cast<int>(basic.currentDirection)] = -1.0f;
    }
    if (basic.dangerLeft) {
        int leftIdx = (static_cast<int>(basic.currentDirection) + 3) % 4;
        actionValues[leftIdx] = -1.0f;
    }
    if (basic.dangerRight) {
        int rightIdx = (static_cast<int>(basic.currentDirection) + 1) % 4;
        actionValues[rightIdx] = -1.0f;
    }
    
    // Reward food-seeking actions
    if (basic.foodUp) actionValues[0] += 0.8f;
    if (basic.foodDown) actionValues[1] += 0.8f;
    if (basic.foodLeft) actionValues[2] += 0.8f;
    if (basic.foodRight) actionValues[3] += 0.8f;
    
    // Add some exploration noise
    std::normal_distribution<float> noise(0.0f, 0.1f);
    for (auto& value : actionValues) {
        value += noise(m_rng);
    }
    
    // Select action with highest estimated value
    int bestAction = std::distance(actionValues.begin(), 
                                 std::max_element(actionValues.begin(), actionValues.end()));
    
    return static_cast<Direction>(bestAction);
}

bool ActorCriticAgent::loadModel(const std::string& path) {
    spdlog::info("ActorCriticAgent: Model file detected: {}", path);
    spdlog::info("ActorCriticAgent: Using value-based decision making");
    return true;
}

std::string ActorCriticAgent::getAgentInfo() const {
    return "Actor-Critic (Value-based) | Balanced exploration-exploitation";
}

std::string ActorCriticAgent::getModelInfo() const {
    return "Enhanced Actor-Critic placeholder - value-based action selection";
}

// Enhanced TrainedModelManager Implementation
TrainedModelManager::TrainedModelManager(const std::string& modelsDir) 
    : m_modelsDirectory(modelsDir) {
    spdlog::info("TrainedModelManager: Initialized with directory: {}", modelsDir);
    
    if (!std::filesystem::exists(modelsDir)) {
        std::filesystem::create_directories(modelsDir);
        spdlog::info("TrainedModelManager: Created models directory: {}", modelsDir);
    }
    
    scanForModels();
}

// Replace the scanForModels() method in MLAgents.cpp:

void TrainedModelManager::scanForModels() {
    m_availableModels.clear();
    
    if (!std::filesystem::exists(m_modelsDirectory)) {
        spdlog::warn("TrainedModelManager: Models directory does not exist: {}", m_modelsDirectory);
        return;
    }
    
    // Scan for Q-Learning models
    auto qlearningDir = std::filesystem::path(m_modelsDirectory) / "qlearning";
    if (std::filesystem::exists(qlearningDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(qlearningDir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                
                // Look for qtable_*.json files (skip session files and reports)
                if (filename.substr(0, 7) == "qtable_" && 
                    filename.length() > 12 && 
                    filename.substr(filename.length() - 5) == ".json" &&
                    filename.find("session") == std::string::npos &&
                    filename.find("report") == std::string::npos) {
                    
                    // Extract profile: qtable_balanced.json -> "balanced"
                    std::string profile = filename.substr(7, filename.length() - 12);
                    
                    TrainedModelInfo info;
                    info.name = "Q-Learning " + profile;
                    info.profile = profile;
                    info.modelPath = entry.path().string();
                    info.modelType = "qlearning";
                    info.description = "Q-Learning model with " + profile + " training profile";
                    info.isLoaded = false;
                    
                    if (validateQlearningModel(info.modelPath)) {
                        m_availableModels.push_back(info);
                        spdlog::info("TrainedModelManager: Found Q-Learning model: {} ({})", 
                                     profile, info.modelPath);
                    }
                }
            }
        }
    }
    
    // Scan for DQN models
    auto dqnDir = std::filesystem::path(m_modelsDirectory) / "dqn";
    if (std::filesystem::exists(dqnDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(dqnDir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                
                // Look for dqn_*.pth files (skip checkpoint files)
                if (filename.substr(0, 4) == "dqn_" && 
                    filename.length() > 8 && 
                    filename.substr(filename.length() - 4) == ".pth" &&
                    filename.find("best") == std::string::npos &&
                    filename.find("checkpoint") == std::string::npos) {
                    
                    // Extract profile: dqn_balanced.pth -> "balanced"
                    std::string profile = filename.substr(4, filename.length() - 8);
                    
                    TrainedModelInfo info;
                    info.name = "DQN " + profile;
                    info.profile = profile;
                    info.modelPath = entry.path().string();
                    info.modelType = "dqn";
                    info.description = "DQN neural network model (" + profile + " profile)";
                    info.isLoaded = false;
                    
                    m_availableModels.push_back(info);
                    spdlog::info("TrainedModelManager: Found dqn model: {} ({})", 
                                 profile, info.modelPath);
                }
            }
        }
    }
    
    // Scan for PPO models (CHANGED: from policy_gradient to ppo)
    auto ppoDir = std::filesystem::path(m_modelsDirectory) / "ppo";
    if (std::filesystem::exists(ppoDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(ppoDir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                
                if (filename.substr(0, 4) == "ppo_" && 
                    filename.length() > 8 && 
                    filename.substr(filename.length() - 4) == ".pth" &&
                    filename.find("best") == std::string::npos &&
                    filename.find("checkpoint") == std::string::npos) {
                    
                    std::string profile = filename.substr(4, filename.length() - 8);
                    
                    TrainedModelInfo info;
                    info.name = "PPO " + profile;
                    info.profile = profile;
                    info.modelPath = entry.path().string();
                    info.modelType = "ppo";
                    info.description = "PPO model (" + profile + " profile)";
                    info.isLoaded = false;
                    
                    m_availableModels.push_back(info);
                    spdlog::info("TrainedModelManager: Found PPO model: {} ({})", 
                                 profile, info.modelPath);
                }
            }
        }
    }
    
    // Scan for Actor-Critic models
    auto acDir = std::filesystem::path(m_modelsDirectory) / "actor_critic";
    if (std::filesystem::exists(acDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(acDir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                
                // Look for ac_*.pth files
                if (filename.substr(0, 3) == "ac_" && 
                    filename.length() > 7 && 
                    filename.substr(filename.length() - 4) == ".pth" &&
                    filename.find("best") == std::string::npos &&
                    filename.find("checkpoint") == std::string::npos) {
                    
                    // Extract profile: ac_balanced.pth -> "balanced"
                    std::string profile = filename.substr(3, filename.length() - 7);
                    
                    TrainedModelInfo info;
                    info.name = "Actor-Critic " + profile;
                    info.profile = profile;
                    info.modelPath = entry.path().string();
                    info.modelType = "actor_critic";
                    info.description = "Actor-critic model (" + profile + " profile)";
                    info.isLoaded = false;
                    
                    m_availableModels.push_back(info);
                    spdlog::info("TrainedModelManager: Found actor_critic model: {} ({})", 
                                 profile, info.modelPath);
                }
            }
        }
    }
    
    spdlog::info("TrainedModelManager: Found {} total models", m_availableModels.size());
}

bool TrainedModelManager::validateQlearningModel(const std::string& modelPath) const {
    try {
        std::ifstream file(modelPath);
        if (!file.is_open()) return false;
        
        nlohmann::json j;
        file >> j;
        
        if (!j.contains("qTable") || !j.contains("hyperparameters")) {
            spdlog::warn("TrainedModelManager: Q-Learning model missing required structure: {}", modelPath);
            return false;
        }
        
        auto& qTable = j["qTable"];
        if (qTable.empty()) {
            spdlog::warn("TrainedModelManager: Empty Q-table in model: {}", modelPath);
            return false;
        }
        
        int validEntries = 0;
        for (auto& [state, actions] : qTable.items()) {
            if (state.length() == 9 && actions.is_array() && actions.size() == 4) {
                validEntries++;
                if (validEntries >= 5) break;
            }
        }
        
        if (validEntries == 0) {
            spdlog::warn("TrainedModelManager: No valid Q-table entries in model: {}", modelPath);
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("TrainedModelManager: Error validating Q-Learning model {}: {}", modelPath, e.what());
        return false;
    }
}

TrainedModelInfo* TrainedModelManager::findModel(const std::string& modelName) {
    for (auto& model : m_availableModels) {
        if (model.name == modelName) {
            return &model;
        }
    }
    return nullptr;
}

void TrainedModelManager::createModelInfoFiles() {
    // Optional: Create .info files for models that don't have them
    spdlog::debug("TrainedModelManager: Model info files creation not implemented");
}

// Updated Agent Factory
std::unique_ptr<IAgent> AgentFactory::createAgent(const AgentConfig& config) {
    spdlog::info("AgentFactory: Creating agent of type: {} ({})", 
                 static_cast<int>(config.type), config.getAgentTypeString());
    
    switch (config.type) {
        case AgentType::Q_LEARNING:
            return std::make_unique<QLearningAgentEnhanced>(config.learningRate, config.discountFactor, config.epsilon);
        case AgentType::DEEP_Q_NETWORK:
            return std::make_unique<DQNAgent>();
        case AgentType::PPO:  // CHANGED: POLICY_GRADIENT -> PPO
            return std::make_unique<PPOAgent>();
        case AgentType::ACTOR_CRITIC:
            return std::make_unique<ActorCriticAgent>();
        default:
            spdlog::warn("AgentFactory: Unknown agent type, defaulting to Q-Learning");
            return std::make_unique<QLearningAgentEnhanced>();
    }
}

std::unique_ptr<IAgent> AgentFactory::createTrainedAgent(const std::string& modelName) {
    TrainedModelManager manager;
    
    auto* modelInfo = manager.findModel(modelName);
    
    if (!modelInfo) {
        spdlog::error("AgentFactory: Trained model not found: {}", modelName);
        return nullptr;
    }
    
    spdlog::info("AgentFactory: Creating trained agent: {} (type: {})", 
                 modelInfo->name, modelInfo->modelType);
    
    if (modelInfo->modelType == "qlearning") {
        return std::make_unique<QLearningAgentEnhanced>(*modelInfo);
    } else {
        spdlog::info("AgentFactory: Creating enhanced intelligent placeholder for: {}", modelInfo->name);
        
        // Return appropriate enhanced placeholder based on model type
        if (modelInfo->modelType == "dqn") {
            auto agent = std::make_unique<DQNAgent>();
            agent->loadModel(modelInfo->modelPath); // This just logs that model was detected
            return agent;
        } else if (modelInfo->modelType == "ppo") {
            auto agent = std::make_unique<PPOAgent>();
            agent->loadModel(modelInfo->modelPath);
            return agent;
        } else if (modelInfo->modelType == "actor_critic") {
            auto agent = std::make_unique<ActorCriticAgent>();
            agent->loadModel(modelInfo->modelPath);
            return agent;
        }
    }
    
    return nullptr;
}



// State Generator Implementation (unchanged)
// Add this to the end of MLAgents.cpp

EnhancedState StateGenerator::generateState(const Snake& snake, const Apple& apple, const Grid& grid) {
    EnhancedState state;
    state.basic = generateBasicState(snake, apple, grid);
    
    auto head = snake.getHeadPosition();
    auto food = apple.getPosition();
    
    state.distanceToFood = static_cast<float>(std::abs(head.x - food.x) + std::abs(head.y - food.y));
    
    state.distanceToWall[0] = static_cast<float>(head.y);
    state.distanceToWall[1] = static_cast<float>(grid.getSize() - 1 - head.y);
    state.distanceToWall[2] = static_cast<float>(head.x);
    state.distanceToWall[3] = static_cast<float>(grid.getSize() - 1 - head.x);
    
    calculateBodyDensity(snake, grid, state.bodyDensity);
    
    state.snakeLength = snake.getLength();
    state.emptySpaces = grid.getSize() * grid.getSize() - snake.getLength() - 1;
    state.pathToFood = calculatePathToFood(snake, apple, grid);
    
    return state;
}

EnhancedState StateGenerator::generateEnhancedState(const Snake& snake, const Apple& apple, const Grid& grid, int episode) {
    EnhancedState state = generateState(snake, apple, grid);
    state.pathToFood += calculateSnakeEfficiency(snake, episode) * 0.1f;
    
    float complexity = calculateEnvironmentComplexity(snake, grid);
    for (int i = 0; i < 4; ++i) {
        state.distanceToWall[i] *= (1.0f + complexity * 0.2f);
    }
    
    return state;
}

AgentState StateGenerator::generateBasicState(const Snake& snake, const Apple& apple, const Grid& grid) {
    AgentState state;
    auto head = snake.getHeadPosition();
    auto food = apple.getPosition();
    auto direction = snake.getDirection();
    
    sf::Vector2i straight = head, left = head, right = head;
    
    switch (direction) {
        case Direction::UP:
            straight.y--; left.x--; right.x++;
            break;
        case Direction::DOWN:
            straight.y++; left.x++; right.x--;
            break;
        case Direction::LEFT:
            straight.x--; left.y++; right.y--;
            break;
        case Direction::RIGHT:
            straight.x++; left.y--; right.y++;
            break;
    }
    
    state.dangerStraight = !grid.isValidPosition(straight) || snake.isPositionOnSnake(straight);
    state.dangerLeft = !grid.isValidPosition(left) || snake.isPositionOnSnake(left);
    state.dangerRight = !grid.isValidPosition(right) || snake.isPositionOnSnake(right);
    state.currentDirection = direction;
    
    state.foodLeft = food.x < head.x;
    state.foodRight = food.x > head.x;
    state.foodUp = food.y < head.y;
    state.foodDown = food.y > head.y;
    
    return state;
}

void StateGenerator::calculateBodyDensity(const Snake& snake, const Grid& grid, float density[4]) {
    int gridSize = grid.getSize();
    int quadrantCounts[4] = {0, 0, 0, 0};
    int halfSize = gridSize / 2;
    
    for (const auto& segment : snake.getBody()) {
        int quadrant = 0;
        if (segment.x >= halfSize) quadrant += 1;
        if (segment.y >= halfSize) quadrant += 2;
        quadrantCounts[quadrant]++;
    }
    
    int quadrantSize = halfSize * halfSize;
    for (int i = 0; i < 4; ++i) {
        density[i] = static_cast<float>(quadrantCounts[i]) / static_cast<float>(quadrantSize);
    }
}

float StateGenerator::calculatePathToFood(const Snake& snake, const Apple& apple, const Grid& grid) {
    auto head = snake.getHeadPosition();
    auto food = apple.getPosition();
    
    float baseDistance = static_cast<float>(std::abs(head.x - food.x) + std::abs(head.y - food.y));
    
    sf::Vector2i current = head;
    sf::Vector2i direction = {
        (food.x > head.x) ? 1 : (food.x < head.x) ? -1 : 0,
        (food.y > head.y) ? 1 : (food.y < head.y) ? -1 : 0
    };
    
    float penalty = 0.0f;
    int steps = 0;
    const int maxSteps = 10;
    
    while (current != food && steps < maxSteps) {
        current.x += direction.x;
        current.y += direction.y;
        
        if (snake.isPositionOnSnake(current)) {
            penalty += 2.0f;
        }
        steps++;
    }
    
    return baseDistance + penalty;
}

float StateGenerator::calculateSnakeEfficiency(const Snake& snake, int episode) {
    float lengthRatio = static_cast<float>(snake.getLength()) / 10.0f;
    float episodeProgress = std::min(1.0f, static_cast<float>(episode) / 1000.0f);
    return lengthRatio * (1.0f + episodeProgress);
}

float StateGenerator::calculateEnvironmentComplexity(const Snake& snake, const Grid& grid) {
    int totalCells = grid.getSize() * grid.getSize();
    int occupiedCells = snake.getLength();
    float occupancyRatio = static_cast<float>(occupiedCells) / static_cast<float>(totalCells);
    
    int directionChanges = 0;
    const auto& body = snake.getBody();
    
    if (body.size() > 2) {
        for (size_t i = 1; i < body.size() - 1; ++i) {
            sf::Vector2i prev = body[i-1] - body[i];
            sf::Vector2i next = body[i] - body[i+1];
            
            if (prev.x != next.x || prev.y != next.y) {
                directionChanges++;
            }
        }
    }
    
    float shapeComplexity = static_cast<float>(directionChanges) / static_cast<float>(std::max(1, static_cast<int>(body.size())));
    return occupancyRatio + shapeComplexity * 0.5f;
}