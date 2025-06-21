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
        info.modelType = j.value("modelType", "qlearning");
        info.isLoaded = false;
        
        // Load performance data if available
        if (j.contains("performance")) {
            auto perf = j["performance"];
            info.performance.bestScore = perf.value("bestScore", 0.0f);
            info.performance.averageScore = perf.value("averageScore", 0.0f);
            info.performance.consistency = perf.value("consistency", 0.0f);
            info.performance.efficiency = perf.value("efficiency", 0.0f);
            info.performance.totalEpisodes = perf.value("totalEpisodes", 0);
            info.performance.trainingDuration = perf.value("trainingDuration", 0);
        }
        
        spdlog::info("TrainedModelInfo: Loaded model info: {} (type: {}, best score: {:.1f})", 
                     info.name, info.modelType, info.performance.bestScore);
        
    } catch (const std::exception& e) {
        spdlog::error("TrainedModelInfo: Error loading info file {}: {}", infoPath, e.what());
    }
    
    return info;
}

// Enhanced EvaluationReportData Implementation
EvaluationReportData EvaluationReportData::loadFromFile(const std::string& reportPath) {
    EvaluationReportData data;
    
    try {
        std::ifstream file(reportPath);
        if (!file.is_open()) {
            spdlog::warn("EvaluationReportData: Could not load report: {}", reportPath);
            return data;
        }
        
        nlohmann::json j;
        file >> j;
        
        data.reportDate = j.value("report_date", "Unknown");
        data.version = j.value("version", "1.0");
        
        // Load model performance data
        if (j.contains("models")) {
            for (auto& [modelName, modelData] : j["models"].items()) {
                ModelPerformanceData perf;
                perf.bestScore = modelData.value("best_score", 0.0f);
                perf.averageScore = modelData.value("average_score", 0.0f);
                perf.consistency = modelData.value("consistency", 0.0f);
                perf.efficiency = modelData.value("efficiency", 0.0f);
                perf.totalEpisodes = modelData.value("total_episodes", 0);
                perf.successRate = modelData.value("success_rate", 0.0f);
                perf.improvementRate = modelData.value("improvement_rate", 0.0f);
                perf.trainingProfile = modelData.value("training_profile", "balanced");
                
                if (modelData.contains("analysis_image")) {
                    perf.analysisImagePath = modelData["analysis_image"];
                }
                
                data.modelPerformance[modelName] = perf;
            }
        }
        
        // Load analysis images
        if (j.contains("analysis_images")) {
            auto images = j["analysis_images"];
            data.analysisImages["qtable_balanced"] = images.value("qtable_balanced", "models/analysis_qtable_balanced.png");
            data.analysisImages["dqn_balanced"] = images.value("dqn_balanced", "models/analysis_dqn_balanced_best_fixed.png");
            data.analysisImages["ppo_balanced"] = images.value("ppo_balanced", "models/analysis_ppo_balanced_best_fixed.png");
            data.analysisImages["ac_balanced"] = images.value("ac_balanced", "models/analysis_ac_balanced_best_fixed.png");
            data.comparisonImagePath = images.value("comparison", "models/enhanced_comparison_fixed.png");
        } else {
            // Default paths if not in JSON
            data.analysisImages["qtable_balanced"] = "models/analysis_qtable_balanced.png";
            data.analysisImages["dqn_balanced"] = "models/analysis_dqn_balanced_best_fixed.png";
            data.analysisImages["ppo_balanced"] = "models/analysis_ppo_balanced_best_fixed.png";
            data.analysisImages["ac_balanced"] = "models/analysis_ac_balanced_best_fixed.png";
            data.comparisonImagePath = "models/enhanced_comparison_fixed.png";
        }
        
        // Load summary statistics
        if (j.contains("summary")) {
            auto summary = j["summary"];
            data.bestOverallModel = summary.value("best_overall", "Unknown");
            data.mostConsistentModel = summary.value("most_consistent", "Unknown");
            data.fastestLearner = summary.value("fastest_learner", "Unknown");
        }
        
        spdlog::info("EvaluationReportData: Loaded {} model performance entries", data.modelPerformance.size());
        
    } catch (const std::exception& e) {
        spdlog::error("EvaluationReportData: Error loading report {}: {}", reportPath, e.what());
    }
    
    return data;
}

void TrainedModelInfo::saveToFile(const std::string& infoPath) const {
    try {
        nlohmann::json j;
        j["name"] = name;
        j["profile"] = profile;
        j["modelPath"] = modelPath;
        j["description"] = description;
        j["averageScore"] = performance.averageScore;
        j["episodesTrained"] = performance.totalEpisodes;
        
        j["performance"] = {
            {"bestScore", performance.bestScore},
            {"averageScore", performance.averageScore},
            {"consistency", performance.consistency},
            {"efficiency", performance.efficiency},
            {"totalEpisodes", performance.totalEpisodes},
            {"trainingDuration", performance.trainingDuration}
        };
        j["episodesTrained"] = performance.totalEpisodes;
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

// Enhanced DQN Agent Implementation with LibTorch Integration
DQNAgent::DQNAgent() : m_epsilon(0.05f), m_rng(std::random_device{}()), 
                       m_modelLoadAttempted(false), m_isUsingTorch(false) {
    spdlog::info("DQNAgent: Initialized - attempting LibTorch integration");
    m_modelInfo.name = "DQN Agent";
    m_modelInfo.modelType = "dqn";
    m_modelInfo.description = "Deep Q-Network with LibTorch inference";
    
    m_torchModel = std::make_unique<DQNInference>();
}

DQNAgent::DQNAgent(const TrainedModelInfo& modelInfo) 
    : m_epsilon(0.05f), m_rng(std::random_device{}()), m_modelInfo(modelInfo),
      m_modelLoadAttempted(false), m_isUsingTorch(false) {
    spdlog::info("DQNAgent: Initialized with model info: {}", modelInfo.name);
    m_torchModel = std::make_unique<DQNInference>();
    
    // Attempt to load the model immediately
    if (!modelInfo.modelPath.empty()) {
        tryLoadTorchModel(modelInfo.modelPath);
    }
}

std::vector<std::string> DQNAgent::getPossibleModelPaths(const std::string& basePath) {
    std::vector<std::string> paths;
    
    // Original path
    paths.push_back(basePath);
    
    // Convert .pth to .pt
    if (basePath.find(".pth") != std::string::npos) {
        std::string ptPath = basePath;
        size_t pos = ptPath.find(".pth");
        ptPath.replace(pos, 4, ".pt");
        paths.push_back(ptPath);
    }
    
    // Convert .pt to .pth
    if (basePath.find(".pt") != std::string::npos) {
        std::string pthPath = basePath;
        size_t pos = pthPath.find(".pt");
        pthPath.replace(pos, 3, ".pth");
        paths.push_back(pthPath);
    }
    
    return paths;
}

bool DQNAgent::tryLoadTorchModel(const std::string& path) {
    if (m_modelLoadAttempted && m_lastModelPath == path) {
        return m_isUsingTorch;
    }
    
    m_modelLoadAttempted = true;
    m_lastModelPath = path;
    m_isUsingTorch = false;
    
    spdlog::info("DQNAgent: Attempting to load LibTorch model from: {}", path);
    
    auto possiblePaths = getPossibleModelPaths(path);
    
    for (const auto& modelPath : possiblePaths) {
        spdlog::debug("DQNAgent: Trying model path: {}", modelPath);
        
        if (!std::filesystem::exists(modelPath)) {
            spdlog::debug("DQNAgent: Model file does not exist: {}", modelPath);
            continue;
        }
        
        spdlog::info("DQNAgent: Found model file: {}", modelPath);
        
        try {
            if (m_torchModel->loadModel(modelPath)) {
                m_isUsingTorch = true;
                spdlog::info("DQNAgent: Successfully loaded LibTorch model: {}", modelPath);
                
                // Test the model with a sample input
                EnhancedState testState;
                testState.basic.dangerStraight = false;
                testState.basic.dangerLeft = false;
                testState.basic.dangerRight = false;
                testState.basic.currentDirection = Direction::RIGHT;
                testState.basic.foodLeft = true;
                testState.basic.foodRight = false;
                testState.basic.foodUp = false;
                testState.basic.foodDown = false;
                
                Direction testAction = m_torchModel->getAction(testState, 0.0f);
                spdlog::info("DQNAgent: Model test successful - predicted action: {}", static_cast<int>(testAction));
                
                return true;
            } else {
                spdlog::warn("DQNAgent: LibTorch failed to load model: {}", modelPath);
            }
        } catch (const std::exception& e) {
            spdlog::error("DQNAgent: Exception loading model {}: {}", modelPath, e.what());
        }
    }
    
    spdlog::warn("DQNAgent: All model loading attempts failed, falling back to heuristics");
    return false;
}

Direction DQNAgent::getAction(const EnhancedState& state, bool training) {
    if (m_isUsingTorch && m_torchModel && m_torchModel->isLoaded()) {
        try {
            float epsilon = training ? m_epsilon : 0.0f;
            Direction action = m_torchModel->getAction(state, epsilon);
            spdlog::debug("DQNAgent: LibTorch action: {}", static_cast<int>(action));
            return action;
        } catch (const std::exception& e) {
            spdlog::error("DQNAgent: LibTorch inference failed: {}, falling back to heuristics", e.what());
            m_isUsingTorch = false;
        }
    }
    
    // Fallback to intelligent heuristics
    spdlog::debug("DQNAgent: Using heuristic fallback");
    return makeIntelligentDecision(state);
}

Direction DQNAgent::makeIntelligentDecision(const EnhancedState& state) {
    const auto& basic = state.basic;
    
    // Priority 1: Avoid immediate danger
    std::vector<Direction> safeActions;
    if (!basic.dangerStraight) safeActions.push_back(basic.currentDirection);
    if (!basic.dangerLeft) {
        Direction leftDir = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 3) % 4);
        safeActions.push_back(leftDir);
    }
    if (!basic.dangerRight) {
        Direction rightDir = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 1) % 4);
        safeActions.push_back(rightDir);
    }
    
    if (safeActions.empty()) {
        Direction opposite = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 2) % 4);
        safeActions.push_back(opposite);
    }
    
    // Priority 2: Among safe actions, choose one that moves toward food
    for (Direction action : safeActions) {
        bool movesTowardFood = false;
        switch (action) {
            case Direction::UP: movesTowardFood = basic.foodUp; break;
            case Direction::DOWN: movesTowardFood = basic.foodDown; break;
            case Direction::LEFT: movesTowardFood = basic.foodLeft; break;
            case Direction::RIGHT: movesTowardFood = basic.foodRight; break;
        }
        if (movesTowardFood) return action;
    }
    
    return safeActions[0];
}

bool DQNAgent::loadModel(const std::string& path) {
    return tryLoadTorchModel(path);
}

std::string DQNAgent::getAgentInfo() const {
    if (m_isUsingTorch) {
        return "DQN (LibTorch) | Model loaded successfully";
    }
    return "DQN (Heuristic Fallback) | Model loading failed";
}

std::string DQNAgent::getModelInfo() const {
    if (m_isUsingTorch && m_torchModel && m_torchModel->isLoaded()) {
        return "LibTorch DQN - Path: " + m_torchModel->getModelPath() + 
               " | Best: " + std::to_string(static_cast<int>(m_modelInfo.performance.bestScore));
    }
    if (m_modelInfo.performance.bestScore > 0) {
        return "DQN Model (Fallback) - Best: " + std::to_string(static_cast<int>(m_modelInfo.performance.bestScore)) +
               " | Avg: " + std::to_string(m_modelInfo.performance.averageScore).substr(0, 4);
    }
    return "DQN fallback - LibTorch model not loaded";
}

void DQNAgent::updateAgent(const EnhancedState&, Direction, float, const EnhancedState&) {}

// Enhanced PPO Agent Implementation with LibTorch Integration
PPOAgent::PPOAgent() : m_rng(std::random_device{}()), 
                       m_modelLoadAttempted(false), m_isUsingTorch(false) {
    spdlog::info("PPOAgent: Initialized - attempting LibTorch integration");
    m_modelInfo.name = "PPO Agent";
    m_modelInfo.modelType = "ppo";
    m_modelInfo.description = "PPO with LibTorch inference";
    
    m_torchModel = std::make_unique<PPOInference>();
}

PPOAgent::PPOAgent(const TrainedModelInfo& modelInfo) 
    : m_rng(std::random_device{}()), m_modelInfo(modelInfo),
      m_modelLoadAttempted(false), m_isUsingTorch(false) {
    spdlog::info("PPOAgent: Initialized with model info: {}", modelInfo.name);
    m_torchModel = std::make_unique<PPOInference>();
    
    if (!modelInfo.modelPath.empty()) {
        tryLoadTorchModel(modelInfo.modelPath);
    }
}

std::vector<std::string> PPOAgent::getPossibleModelPaths(const std::string& basePath) {
    std::vector<std::string> paths;
    
    // Original path
    paths.push_back(basePath);
    
    // Convert .pth to .pt
    if (basePath.find(".pth") != std::string::npos) {
        std::string ptPath = basePath;
        size_t pos = ptPath.find(".pth");
        ptPath.replace(pos, 4, ".pt");
        paths.push_back(ptPath);
    }
    
    // Convert .pt to .pth
    if (basePath.find(".pt") != std::string::npos) {
        std::string pthPath = basePath;
        size_t pos = pthPath.find(".pt");
        pthPath.replace(pos, 3, ".pth");
        paths.push_back(pthPath);
    }
    
    return paths;
}

bool PPOAgent::tryLoadTorchModel(const std::string& path) {
    if (m_modelLoadAttempted && m_lastModelPath == path) {
        return m_isUsingTorch;
    }
    
    m_modelLoadAttempted = true;
    m_lastModelPath = path;
    m_isUsingTorch = false;
    
    spdlog::info("PPOAgent: Attempting to load LibTorch model from: {}", path);
    
    auto possiblePaths = getPossibleModelPaths(path);
    
    for (const auto& modelPath : possiblePaths) {
        spdlog::debug("PPOAgent: Trying model path: {}", modelPath);
        
        if (!std::filesystem::exists(modelPath)) {
            spdlog::debug("PPOAgent: Model file does not exist: {}", modelPath);
            continue;
        }
        
        spdlog::info("PPOAgent: Found model file: {}", modelPath);
        
        try {
            if (m_torchModel->loadModel(modelPath)) {
                m_isUsingTorch = true;
                spdlog::info("PPOAgent: Successfully loaded LibTorch model: {}", modelPath);
                
                // Test the model
                EnhancedState testState;
                testState.basic.dangerStraight = false;
                testState.basic.dangerLeft = true;
                testState.basic.dangerRight = false;
                testState.basic.currentDirection = Direction::UP;
                testState.basic.foodLeft = false;
                testState.basic.foodRight = true;
                testState.basic.foodUp = false;
                testState.basic.foodDown = true;
                
                Direction testAction = m_torchModel->getAction(testState);
                spdlog::info("PPOAgent: Model test successful - predicted action: {}", static_cast<int>(testAction));
                
                return true;
            } else {
                spdlog::warn("PPOAgent: LibTorch failed to load model: {}", modelPath);
            }
        } catch (const std::exception& e) {
            spdlog::error("PPOAgent: Exception loading model {}: {}", modelPath, e.what());
        }
    }
    
    spdlog::warn("PPOAgent: All model loading attempts failed, falling back to heuristics");
    return false;
}

Direction PPOAgent::getAction(const EnhancedState& state, bool training) {
    if (m_isUsingTorch && m_torchModel && m_torchModel->isLoaded()) {
        try {
            Direction action = m_torchModel->getAction(state);
            spdlog::debug("PPOAgent: LibTorch action: {}", static_cast<int>(action));
            return action;
        } catch (const std::exception& e) {
            spdlog::error("PPOAgent: LibTorch inference failed: {}, falling back to heuristics", e.what());
            m_isUsingTorch = false;
        }
    }
    
    spdlog::debug("PPOAgent: Using probabilistic fallback");
    return makeProbabilisticDecision(state);
}

Direction PPOAgent::makeProbabilisticDecision(const EnhancedState& state) {
    const auto& basic = state.basic;
    std::array<float, 4> actionProbs = {0.25f, 0.25f, 0.25f, 0.25f};
    
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
    
    return static_cast<Direction>(0);
}

bool PPOAgent::loadModel(const std::string& path) {
    return tryLoadTorchModel(path);
}

std::string PPOAgent::getAgentInfo() const {
    if (m_isUsingTorch) {
        return "PPO (LibTorch) | Model loaded successfully";
    }
    return "PPO (Probabilistic Fallback) | Model loading failed";
}

std::string PPOAgent::getModelInfo() const {
    if (m_isUsingTorch && m_torchModel && m_torchModel->isLoaded()) {
        return "LibTorch PPO - Path: " + m_torchModel->getModelPath() + 
               " | Best: " + std::to_string(static_cast<int>(m_modelInfo.performance.bestScore));
    }
    if (m_modelInfo.performance.bestScore > 0) {
        return "PPO Model (Fallback) - Best: " + std::to_string(static_cast<int>(m_modelInfo.performance.bestScore)) +
               " | Success Rate: " + std::to_string(m_modelInfo.performance.successRate).substr(0, 4) + "%";
    }
    return "PPO fallback - LibTorch model not loaded";
}

void PPOAgent::updateAgent(const EnhancedState&, Direction, float, const EnhancedState&) {}

// Enhanced Actor-Critic Agent Implementation with LibTorch Integration
ActorCriticAgent::ActorCriticAgent() : m_rng(std::random_device{}()), 
                                       m_modelLoadAttempted(false), m_isUsingTorch(false) {
    spdlog::info("ActorCriticAgent: Initialized - attempting LibTorch integration");
    m_modelInfo.name = "Actor-Critic Agent";
    m_modelInfo.modelType = "actor_critic";
    m_modelInfo.description = "Actor-critic with LibTorch inference";
    
    m_torchModel = std::make_unique<ActorCriticInference>();
}

ActorCriticAgent::ActorCriticAgent(const TrainedModelInfo& modelInfo) 
    : m_rng(std::random_device{}()), m_modelInfo(modelInfo),
      m_modelLoadAttempted(false), m_isUsingTorch(false) {
    spdlog::info("ActorCriticAgent: Initialized with model info: {}", modelInfo.name);
    m_torchModel = std::make_unique<ActorCriticInference>();
    
    if (!modelInfo.modelPath.empty()) {
        tryLoadTorchModel(modelInfo.modelPath);
    }
}

std::vector<std::string> ActorCriticAgent::getPossibleModelPaths(const std::string& basePath) {
    std::vector<std::string> paths;
    
    // Original path
    paths.push_back(basePath);
    
    // Convert .pth to .pt
    if (basePath.find(".pth") != std::string::npos) {
        std::string ptPath = basePath;
        size_t pos = ptPath.find(".pth");
        ptPath.replace(pos, 4, ".pt");
        paths.push_back(ptPath);
    }
    
    // Convert .pt to .pth
    if (basePath.find(".pt") != std::string::npos) {
        std::string pthPath = basePath;
        size_t pos = pthPath.find(".pt");
        pthPath.replace(pos, 3, ".pth");
        paths.push_back(pthPath);
    }
    
    return paths;
}

bool ActorCriticAgent::tryLoadTorchModel(const std::string& path) {
    if (m_modelLoadAttempted && m_lastModelPath == path) {
        return m_isUsingTorch;
    }
    
    m_modelLoadAttempted = true;
    m_lastModelPath = path;
    m_isUsingTorch = false;
    
    spdlog::info("ActorCriticAgent: Attempting to load LibTorch model from: {}", path);
    
    auto possiblePaths = getPossibleModelPaths(path);
    
    for (const auto& modelPath : possiblePaths) {
        spdlog::debug("ActorCriticAgent: Trying model path: {}", modelPath);
        
        if (!std::filesystem::exists(modelPath)) {
            spdlog::debug("ActorCriticAgent: Model file does not exist: {}", modelPath);
            continue;
        }
        
        spdlog::info("ActorCriticAgent: Found model file: {}", modelPath);
        
        try {
            if (m_torchModel->loadModel(modelPath)) {
                m_isUsingTorch = true;
                spdlog::info("ActorCriticAgent: Successfully loaded LibTorch model: {}", modelPath);
                
                // Test the model
                EnhancedState testState;
                testState.basic.dangerStraight = false;
                testState.basic.dangerLeft = false;
                testState.basic.dangerRight = true;
                testState.basic.currentDirection = Direction::RIGHT;
                testState.basic.foodLeft = true;
                testState.basic.foodRight = false;
                testState.basic.foodUp = false;
                testState.basic.foodDown = false;
                
                Direction testAction = m_torchModel->getAction(testState);
                spdlog::info("ActorCriticAgent: Model test successful - predicted action: {}", static_cast<int>(testAction));
                
                // Test value function if available
                float testValue = m_torchModel->getValue(testState);
                spdlog::info("ActorCriticAgent: State value estimate: {:.3f}", testValue);
                
                return true;
            } else {
                spdlog::warn("ActorCriticAgent: LibTorch failed to load model: {}", modelPath);
            }
        } catch (const std::exception& e) {
            spdlog::error("ActorCriticAgent: Exception loading model {}: {}", modelPath, e.what());
        }
    }
    
    spdlog::warn("ActorCriticAgent: All model loading attempts failed, falling back to heuristics");
    return false;
}

Direction ActorCriticAgent::getAction(const EnhancedState& state, bool training) {
    if (m_isUsingTorch && m_torchModel && m_torchModel->isLoaded()) {
        try {
            Direction action = m_torchModel->getAction(state);
            spdlog::debug("ActorCriticAgent: LibTorch action: {}", static_cast<int>(action));
            return action;
        } catch (const std::exception& e) {
            spdlog::error("ActorCriticAgent: LibTorch inference failed: {}, falling back to heuristics", e.what());
            m_isUsingTorch = false;
        }
    }
    
    spdlog::debug("ActorCriticAgent: Using value-based fallback");
    return makeValueBasedDecision(state);
}

float ActorCriticAgent::getStateValue(const EnhancedState& state) {
    if (m_isUsingTorch && m_torchModel && m_torchModel->isLoaded()) {
        try {
            return m_torchModel->getValue(state);
        } catch (const std::exception& e) {
            spdlog::error("ActorCriticAgent: Value estimation failed: {}", e.what());
        }
    }
    return 0.0f; // Fallback value
}

Direction ActorCriticAgent::makeValueBasedDecision(const EnhancedState& state) {
    const auto& basic = state.basic;
    std::array<float, 4> actionValues = {0.5f, 0.5f, 0.5f, 0.5f};
    
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
    
    // Add exploration noise
    std::normal_distribution<float> noise(0.0f, 0.1f);
    for (auto& value : actionValues) {
        value += noise(m_rng);
    }
    
    int bestAction = std::distance(actionValues.begin(), 
                                 std::max_element(actionValues.begin(), actionValues.end()));
    
    return static_cast<Direction>(bestAction);
}

bool ActorCriticAgent::loadModel(const std::string& path) {
    return tryLoadTorchModel(path);
}

std::string ActorCriticAgent::getAgentInfo() const {
    if (m_isUsingTorch) {
        return "Actor-Critic (LibTorch) | Model loaded successfully";
    }
    return "Actor-Critic (Value-based Fallback) | Model loading failed";
}

std::string ActorCriticAgent::getModelInfo() const {
    if (m_isUsingTorch && m_torchModel && m_torchModel->isLoaded()) {
        return "LibTorch Actor-Critic - Path: " + m_torchModel->getModelPath() + 
               " | Best: " + std::to_string(static_cast<int>(m_modelInfo.performance.bestScore));
    }
    if (m_modelInfo.performance.bestScore > 0) {
        return "AC Model (Fallback) - Best: " + std::to_string(static_cast<int>(m_modelInfo.performance.bestScore)) +
               " | Consistency: " + std::to_string(m_modelInfo.performance.consistency).substr(0, 4);
    }
    return "Actor-Critic fallback - LibTorch model not loaded";
}

void ActorCriticAgent::updateAgent(const EnhancedState&, Direction, float, const EnhancedState&) {}

// Enhanced TrainedModelManager Implementation
TrainedModelManager::TrainedModelManager(const std::string& modelsDir) 
    : m_modelsDirectory(modelsDir) {
    spdlog::info("TrainedModelManager: Initialized with directory: {}", modelsDir);
    
    if (!std::filesystem::exists(modelsDir)) {
        std::filesystem::create_directories(modelsDir);
        spdlog::info("TrainedModelManager: Created models directory: {}", modelsDir);
    }
    
    loadEvaluationReport();
    scanForModels();
}

void TrainedModelManager::loadEvaluationReport(const std::string& reportPath) {
    m_evaluationData = EvaluationReportData::loadFromFile(reportPath);
    spdlog::info("TrainedModelManager: Loaded evaluation report with {} models", 
                 m_evaluationData.modelPerformance.size());
}

void TrainedModelManager::scanForModels() {
    m_availableModels.clear();
    
    if (!std::filesystem::exists(m_modelsDirectory)) {
        spdlog::warn("TrainedModelManager: Models directory does not exist: {}", m_modelsDirectory);
        return;
    }
    
    // Scan for Q-Learning models with performance data
    auto qlearningDir = std::filesystem::path(m_modelsDirectory) / "qlearning";
    if (std::filesystem::exists(qlearningDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(qlearningDir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                
                if (filename.substr(0, 7) == "qtable_" && 
                    filename.length() > 12 && 
                    filename.substr(filename.length() - 5) == ".json" &&
                    filename.find("session") == std::string::npos &&
                    filename.find("report") == std::string::npos) {
                    
                    std::string profile = filename.substr(7, filename.length() - 12);
                    std::string modelKey = "qtable_" + profile;
                    
                    TrainedModelInfo info;
                    info.name = "Q-Learning " + profile;
                    info.profile = profile;
                    info.modelPath = entry.path().string();
                    info.modelType = "qlearning";
                    info.description = "Q-Learning model with " + profile + " training profile";
                    info.isLoaded = false;
                    
                    // Load performance data from evaluation report
                    if (m_evaluationData.modelPerformance.find(modelKey) != m_evaluationData.modelPerformance.end()) {
                        info.performance = m_evaluationData.modelPerformance[modelKey];
                        spdlog::info("TrainedModelManager: Loaded performance data for {}: best={:.1f}", 
                                     info.name, info.performance.bestScore);
                    }
                    
                    if (validateQlearningModel(info.modelPath)) {
                        m_availableModels.push_back(info);
                        spdlog::info("TrainedModelManager: Found Q-Learning model: {} ({})", 
                                     profile, info.modelPath);
                    }
                }
            }
        }
    }
    
    // Add neural network models with performance data
    std::vector<std::pair<std::string, std::string>> neuralModels = {
        {"dqn", "dqn_balanced"},
        {"ppo", "ppo_balanced"}, 
        {"actor_critic", "ac_balanced"}
    };
    
    for (const auto& [type, key] : neuralModels) {
        if (m_evaluationData.modelPerformance.find(key) != m_evaluationData.modelPerformance.end()) {
            TrainedModelInfo info;
            info.name = (type == "actor_critic" ? "Actor-Critic" : 
                        (type == "dqn" ? "DQN" : "PPO")) + std::string(" balanced");
            info.profile = "balanced";
            info.modelPath = "models/" + type + "/" + key + ".pth";
            info.modelType = type;
            info.description = info.name + " model with balanced training profile";
            info.performance = m_evaluationData.modelPerformance[key];
            info.isLoaded = false;
            
            m_availableModels.push_back(info);
            spdlog::info("TrainedModelManager: Found {} model with best score: {:.1f}", 
                         info.name, info.performance.bestScore);
        }
    }
    
    spdlog::info("TrainedModelManager: Found {} total models", m_availableModels.size());
}

std::vector<TrainedModelInfo> TrainedModelManager::getTopPerformingModels(int count) const {
    std::vector<TrainedModelInfo> sorted = m_availableModels;
    std::sort(sorted.begin(), sorted.end(), 
              [](const TrainedModelInfo& a, const TrainedModelInfo& b) {
                  return a.performance.bestScore > b.performance.bestScore;
              });
    
    if (count > 0 && count < static_cast<int>(sorted.size())) {
        sorted.resize(count);
    }
    
    return sorted;
}

std::string TrainedModelManager::getPerformanceSummary() const {
    if (m_availableModels.empty()) {
        return "No trained models available";
    }
    
    auto topModels = getTopPerformingModels(3);
    std::string summary = "Top performers: ";
    
    for (size_t i = 0; i < topModels.size(); ++i) {
        if (i > 0) summary += ", ";
        summary += topModels[i].name + " (" + std::to_string(static_cast<int>(topModels[i].performance.bestScore)) + ")";
    }
    
    return summary;
}

std::vector<std::pair<std::string, float>> TrainedModelManager::getLeaderboardData() const {
    std::vector<std::pair<std::string, float>> leaderboard;
    
    for (const auto& model : m_availableModels) {
        if (model.performance.bestScore > 0) {
            leaderboard.emplace_back(model.name, model.performance.bestScore);
        }
    }
    
    std::sort(leaderboard.begin(), leaderboard.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return leaderboard;
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
        case AgentType::PPO:
            return std::make_unique<PPOAgent>();
        case AgentType::ACTOR_CRITIC:
            return std::make_unique<ActorCriticAgent>();
        default:
            spdlog::warn("AgentFactory: Unknown agent type, defaulting to Q-Learning");
            return std::make_unique<QLearningAgentEnhanced>();
    }
}

std::unique_ptr<IAgent> AgentFactory::createAgentWithPerformanceData(const TrainedModelInfo& modelInfo) {
    spdlog::info("AgentFactory: Creating agent with performance data: {} (best: {:.1f})", 
                 modelInfo.name, modelInfo.performance.bestScore);
    
    if (modelInfo.modelType == "qlearning") {
        return std::make_unique<QLearningAgentEnhanced>(modelInfo);
    } else if (modelInfo.modelType == "dqn") {
        return std::make_unique<DQNAgent>(modelInfo);
    } else if (modelInfo.modelType == "ppo") {
        return std::make_unique<PPOAgent>(modelInfo);
    } else if (modelInfo.modelType == "actor_critic") {
        return std::make_unique<ActorCriticAgent>(modelInfo);
    }
    
    return nullptr;
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
        spdlog::info("AgentFactory: Creating LibTorch agent for: {}", modelInfo->name);
        
        if (modelInfo->modelType == "dqn") {
            auto agent = std::make_unique<DQNAgent>(*modelInfo);
            return agent;
        } else if (modelInfo->modelType == "ppo") {
            auto agent = std::make_unique<PPOAgent>(*modelInfo);
            return agent;
        } else if (modelInfo->modelType == "actor_critic") {
            auto agent = std::make_unique<ActorCriticAgent>(*modelInfo);
            return agent;
        }
    }
    
    return nullptr;
}

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