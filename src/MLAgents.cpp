#include "MLAgents.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
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
        info.isLoaded = false;
        
        spdlog::info("TrainedModelInfo: Loaded model info: {} (avg score: {:.2f})", 
                     info.name, info.averageScore);
        
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
    // Match Python 9-bit encoding exactly
    std::string binary;
    
    // Danger flags (3 bits)
    binary += state.dangerStraight ? "1" : "0";
    binary += state.dangerLeft ? "1" : "0"; 
    binary += state.dangerRight ? "1" : "0";
    
    // Direction (2 bits: 00=UP, 01=DOWN, 10=LEFT, 11=RIGHT)
    switch (state.currentDirection) {
        case Direction::UP:    binary += "00"; break;
        case Direction::DOWN:  binary += "01"; break;
        case Direction::LEFT:  binary += "10"; break;
        case Direction::RIGHT: binary += "11"; break;
    }
    
    // Food flags (4 bits)
    binary += state.foodLeft ? "1" : "0";
    binary += state.foodRight ? "1" : "0";
    binary += state.foodUp ? "1" : "0";
    binary += state.foodDown ? "1" : "0";
    
    spdlog::debug("QLearningAgent: State encoded as 9-bit: {}", binary);
    return binary;
}

AgentState QLearningAgentEnhanced::decodeState9Bit(const std::string& stateStr) const {
    AgentState state;
    
    if (stateStr.length() != 9) {
        spdlog::warn("QLearningAgent: Invalid state string length: {}", stateStr.length());
        return state;
    }
    
    // Decode danger flags
    state.dangerStraight = (stateStr[0] == '1');
    state.dangerLeft = (stateStr[1] == '1');
    state.dangerRight = (stateStr[2] == '1');
    
    // Decode direction
    std::string dirBits = stateStr.substr(3, 2);
    if (dirBits == "00") state.currentDirection = Direction::UP;
    else if (dirBits == "01") state.currentDirection = Direction::DOWN;
    else if (dirBits == "10") state.currentDirection = Direction::LEFT;
    else if (dirBits == "11") state.currentDirection = Direction::RIGHT;
    
    // Decode food flags
    state.foodLeft = (stateStr[5] == '1');
    state.foodRight = (stateStr[6] == '1');
    state.foodUp = (stateStr[7] == '1');
    state.foodDown = (stateStr[8] == '1');
    
    return state;
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

void QLearningAgentEnhanced::updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) {
    if (m_hasLastState) {
        updateQValue(m_lastState, m_lastAction, reward, state.basic);
        spdlog::debug("QLearningAgent: Updated Q-value for action {}, reward {:.2f}", 
                     static_cast<int>(m_lastAction), reward);
    }
    
    m_lastState = state.basic;
    m_lastAction = action;
    m_hasLastState = true;
}

void QLearningAgentEnhanced::updateQValue(const AgentState& state, Direction action, float reward, const AgentState& nextState) {
    std::string stateKey = encodeState9Bit(state);
    std::string nextStateKey = encodeState9Bit(nextState);
    int actionIdx = static_cast<int>(action);
    
    // Initialize Q-values if not exists
    if (m_qTable.find(stateKey) == m_qTable.end()) {
        m_qTable[stateKey] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    if (m_qTable.find(nextStateKey) == m_qTable.end()) {
        m_qTable[nextStateKey] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    
    float currentQ = m_qTable[stateKey][actionIdx];
    float maxNextQ = *std::max_element(m_qTable[nextStateKey].begin(), m_qTable[nextStateKey].end());
    
    // Q-learning update rule
    float newQ = currentQ + m_learningRate * (reward + m_discountFactor * maxNextQ - currentQ);
    m_qTable[stateKey][actionIdx] = newQ;
    
    spdlog::debug("QLearningAgent: Q({}, {}) = {:.3f} -> {:.3f} (reward: {:.2f})", 
                 stateKey, actionIdx, currentQ, newQ, reward);
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
                // Validate state string is 9-bit binary
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
        
        // Load hyperparameters
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
        spdlog::info("QLearningAgent: Hyperparameters - lr: {:.3f}, gamma: {:.3f}, epsilon: {:.3f}",
                     m_learningRate, m_discountFactor, m_epsilon);
        
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("QLearningAgent: Error loading model from {}: {}", modelPath, e.what());
        return false;
    }
}

bool QLearningAgentEnhanced::loadModel(const std::string& path) {
    return loadTrainedModel(path);
}

void QLearningAgentEnhanced::startEpisode() {
    m_hasLastState = false;
    spdlog::debug("QLearningAgent: Started new episode");
}

void QLearningAgentEnhanced::endEpisode() {
    m_hasLastState = false;
    spdlog::debug("QLearningAgent: Ended episode");
}

void QLearningAgentEnhanced::saveModel(const std::string& path) {
    try {
        nlohmann::json j;
        j["qTable"] = nlohmann::json::object();
        
        for (const auto& [state, actions] : m_qTable) {
            j["qTable"][state] = actions;
        }
        
        j["hyperparameters"] = {
            {"learningRate", m_learningRate},
            {"discountFactor", m_discountFactor},
            {"epsilon", m_epsilon}
        };
        
        j["metadata"] = {
            {"modelName", m_modelInfo.name},
            {"profile", m_modelInfo.profile},
            {"saveDate", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"totalStates", m_qTable.size()}
        };
        
        std::ofstream file(path);
        if (file.is_open()) {
            file << j.dump(4);
            spdlog::info("QLearningAgent: Model saved to {} with {} states", path, m_qTable.size());
        }
    } catch (const std::exception& e) {
        spdlog::error("QLearningAgent: Failed to save model to {}: {}", path, e.what());
    }
}

float QLearningAgentEnhanced::getEpsilon() const {
    return m_epsilon;
}

void QLearningAgentEnhanced::decayEpsilon() {
    float oldEpsilon = m_epsilon;
    m_epsilon *= 0.995f;
    m_epsilon = std::max(0.01f, m_epsilon);
    
    if (oldEpsilon != m_epsilon) {
        spdlog::debug("QLearningAgent: Epsilon decayed from {:.4f} to {:.4f}", oldEpsilon, m_epsilon);
    }
}

std::string QLearningAgentEnhanced::getAgentInfo() const {
    return "Q-Learning | States: " + std::to_string(m_qTable.size()) + 
           " | ε: " + std::to_string(m_epsilon).substr(0, 5);
}

std::string QLearningAgentEnhanced::getModelInfo() const {
    if (m_isPreTrained) {
        return m_modelInfo.name + " (Avg: " + std::to_string(m_modelInfo.averageScore).substr(0, 5) + 
               ", Episodes: " + std::to_string(m_modelInfo.episodesTrained) + ")";
    }
    return "Fresh Q-Learning Agent (Training from scratch)";
}

Direction QLearningAgentEnhanced::getMaxQAction(const AgentState& state) const {
    std::string stateKey = encodeState9Bit(state);
    
    if (m_qTable.find(stateKey) == m_qTable.end()) {
        // Random action if state not seen
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

// TrainedModelManager Implementation
TrainedModelManager::TrainedModelManager(const std::string& modelsDir) 
    : m_modelsDirectory(modelsDir) {
    spdlog::info("TrainedModelManager: Initialized with directory: {}", modelsDir);
    
    // Check if src/models exists and use that instead
    if (!std::filesystem::exists(modelsDir) && std::filesystem::exists("src/models/")) {
        m_modelsDirectory = "src/models/";
        spdlog::info("TrainedModelManager: Using src/models/ directory instead");
    }
    
    scanForModels();
    createModelInfoFiles();
}

void TrainedModelManager::scanForModels() {
    m_availableModels.clear();
    
    if (!std::filesystem::exists(m_modelsDirectory)) {
        spdlog::warn("TrainedModelManager: Models directory does not exist: {}", m_modelsDirectory);
        return;
    }
    
    // Scan for qtable_*.json files (exclude checkpoints and reports)
    for (const auto& entry : std::filesystem::directory_iterator(m_modelsDirectory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            
            // Look for pattern: qtable_<profile>.json
            if (filename.substr(0, 7) == "qtable_" && 
                filename.length() > 5 && filename.substr(filename.length() - 5) == ".json" &&
                filename.find("checkpoint") == std::string::npos &&
                filename.find("report") == std::string::npos) {
                
                std::string profile = filename.substr(7); // Remove "qtable_"
                profile = profile.substr(0, profile.length() - 5); // Remove ".json"
                
                std::string modelPath = entry.path().string();
                std::string infoPath = m_modelsDirectory + profile + "_info.json";
                
                TrainedModelInfo info;
                if (std::filesystem::exists(infoPath)) {
                    info = TrainedModelInfo::fromFile(infoPath);
                } else {
                    createDefaultModelInfo(profile, modelPath);
                    info = TrainedModelInfo::fromFile(infoPath);
                }
                
                if (validateModel(modelPath)) {
                    m_availableModels.push_back(info);
                    spdlog::info("TrainedModelManager: Found valid model: {} ({})", profile, modelPath);
                } else {
                    spdlog::warn("TrainedModelManager: Invalid model file: {}", modelPath);
                }
            }
        }
    }
    
    spdlog::info("TrainedModelManager: Found {} valid trained models", m_availableModels.size());
}

void TrainedModelManager::createModelInfoFiles() {
    // Create info files for models that don't have them
    for (const auto& model : m_availableModels) {
        std::string infoPath = m_modelsDirectory + model.profile + "_info.json";
        if (!std::filesystem::exists(infoPath)) {
            createDefaultModelInfo(model.profile, model.modelPath);
        }
    }
}

void TrainedModelManager::createDefaultModelInfo(const std::string& profile, const std::string& modelPath) {
    TrainedModelInfo info;
    
    // Set defaults based on profile
    if (profile == "aggressive") {
        info.name = "Aggressive Q-Learning";
        info.description = "Fast learning, high exploration - good for quick games";
        info.averageScore = 0.0f; // To be filled manually
        info.episodesTrained = 3000;
    } else if (profile == "balanced") {
        info.name = "Balanced Q-Learning";
        info.description = "Stable learning, moderate exploration - best overall performance";
        info.averageScore = 0.0f; // To be filled manually
        info.episodesTrained = 5000;
    } else if (profile == "conservative") {
        info.name = "Conservative Q-Learning";
        info.description = "Careful learning, low exploration - very stable behavior";
        info.averageScore = 0.0f; // To be filled manually
        info.episodesTrained = 7000;
    } else {
        info.name = profile + " Q-Learning";
        info.description = "Custom trained Q-Learning model";
        info.averageScore = 0.0f;
        info.episodesTrained = 0;
    }
    
    info.profile = profile;
    info.modelPath = modelPath;
    
    std::string infoPath = m_modelsDirectory + profile + "_info.json";
    info.saveToFile(infoPath);
    
    spdlog::info("TrainedModelManager: Created default info file: {}", infoPath);
}

std::vector<TrainedModelInfo> TrainedModelManager::getAvailableModels() const {
    return m_availableModels;
}

TrainedModelInfo* TrainedModelManager::findModel(const std::string& profile) {
    auto it = std::find_if(m_availableModels.begin(), m_availableModels.end(),
                          [&profile](const TrainedModelInfo& model) {
                              return model.profile == profile;
                          });
    
    return (it != m_availableModels.end()) ? &(*it) : nullptr;
}

bool TrainedModelManager::validateModel(const std::string& modelPath) const {
    try {
        std::ifstream file(modelPath);
        if (!file.is_open()) return false;
        
        nlohmann::json j;
        file >> j;
        
        // Check required structure
        if (!j.contains("qTable") || !j.contains("hyperparameters")) {
            spdlog::warn("TrainedModelManager: Model missing required structure: {}", modelPath);
            return false;
        }
        
        // Validate Q-table has proper format
        auto& qTable = j["qTable"];
        if (qTable.empty()) {
            spdlog::warn("TrainedModelManager: Empty Q-table in model: {}", modelPath);
            return false;
        }
        
        // Check a few entries for proper format
        int validEntries = 0;
        for (auto& [state, actions] : qTable.items()) {
            if (state.length() == 9 && actions.is_array() && actions.size() == 4) {
                validEntries++;
                if (validEntries >= 5) break; // Check first 5 entries
            }
        }
        
        if (validEntries == 0) {
            spdlog::warn("TrainedModelManager: No valid Q-table entries in model: {}", modelPath);
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("TrainedModelManager: Error validating model {}: {}", modelPath, e.what());
        return false;
    }
}

// Updated Agent Factory Implementation
std::unique_ptr<IAgent> AgentFactory::createAgent(const AgentConfig& config) {
    spdlog::info("AgentFactory: Creating agent of type: {}", static_cast<int>(config.type));
    
    switch (config.type) {
        case AgentType::Q_LEARNING:
            return std::make_unique<QLearningAgentEnhanced>(config.learningRate, config.discountFactor, config.epsilon);
        case AgentType::DEEP_Q_NETWORK:
            return std::make_unique<DQNAgent>();
        case AgentType::POLICY_GRADIENT:
            return std::make_unique<PolicyGradientAgent>();
        default:
            spdlog::warn("AgentFactory: Unknown agent type, defaulting to Q-Learning");
            return std::make_unique<QLearningAgentEnhanced>();
    }
}

std::unique_ptr<IAgent> AgentFactory::createTrainedAgent(const std::string& modelProfile) {
    TrainedModelManager manager;
    
    // Extract profile from display name if needed
    std::string profile = modelProfile;
    if (modelProfile.find("Aggressive") != std::string::npos) profile = "aggressive";
    else if (modelProfile.find("Balanced") != std::string::npos) profile = "balanced";
    else if (modelProfile.find("Conservative") != std::string::npos) profile = "conservative";
    
    auto* modelInfo = manager.findModel(profile);
    
    if (!modelInfo) {
        spdlog::error("AgentFactory: Trained model not found: {} (profile: {})", modelProfile, profile);
        return nullptr;
    }
    
    spdlog::info("AgentFactory: Creating trained agent: {}", modelInfo->name);
    return std::make_unique<QLearningAgentEnhanced>(*modelInfo);
}

std::vector<AgentConfig> AgentFactory::getAvailableTrainedAgents() {
    std::vector<AgentConfig> configs;
    TrainedModelManager manager;
    
    auto models = manager.getAvailableModels();
    for (const auto& model : models) {
        AgentConfig config;
        config.type = AgentType::Q_LEARNING;
        config.name = model.name;
        config.description = model.description + " (Pre-trained)";
        config.isImplemented = true;
        config.modelPath = model.modelPath;
        
        configs.push_back(config);
    }
    
    spdlog::info("AgentFactory: Found {} available trained agents", configs.size());
    return configs;
}

// DQN Agent Implementation (unchanged, keeping existing)
DQNAgent::NeuralNetwork::NeuralNetwork() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    weights.push_back(std::vector<float>(inputSize * hiddenSize));
    for (auto& w : weights[0]) w = dist(gen);
    
    weights.push_back(std::vector<float>(hiddenSize * outputSize));
    for (auto& w : weights[1]) w = dist(gen);
    
    biases.resize(hiddenSize + outputSize);
    for (auto& b : biases) b = dist(gen);
}

std::vector<float> DQNAgent::NeuralNetwork::forward(const std::vector<float>& input) {
    if (input.size() != inputSize) {
        return {0.25f, 0.25f, 0.25f, 0.25f};
    }
    
    std::vector<float> hidden(hiddenSize, 0.0f);
    std::vector<float> output(outputSize, 0.0f);
    
    for (int h = 0; h < hiddenSize; ++h) {
        for (int i = 0; i < inputSize; ++i) {
            hidden[h] += input[i] * weights[0][i * hiddenSize + h];
        }
        hidden[h] += biases[h];
        hidden[h] = std::max(0.0f, hidden[h]);
    }
    
    for (int o = 0; o < outputSize; ++o) {
        for (int h = 0; h < hiddenSize; ++h) {
            output[o] += hidden[h] * weights[1][h * outputSize + o];
        }
        output[o] += biases[hiddenSize + o];
    }
    
    return output;
}

DQNAgent::DQNAgent() : m_epsilon(0.1f), m_rng(std::random_device{}()) {
    spdlog::info("DQNAgent: Initialized (placeholder implementation)");
}

Direction DQNAgent::getAction(const EnhancedState& state, bool training) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    if (training && dist(m_rng) < m_epsilon) {
        std::uniform_int_distribution<int> actionDist(0, 3);
        return static_cast<Direction>(actionDist(m_rng));
    }
    
    auto stateVector = state.toVector();
    auto qValues = m_network.forward(stateVector);
    
    int maxIdx = std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
    return static_cast<Direction>(maxIdx);
}

void DQNAgent::saveModel(const std::string& path) {
    nlohmann::json j;
    j["weights"] = m_network.weights;
    j["biases"] = m_network.biases;
    std::ofstream file(path);
    if (file.is_open()) file << j.dump(4);
}

bool DQNAgent::loadModel(const std::string& path) {
    std::ifstream file(path);
    if (file.is_open()) {
        nlohmann::json j;
        file >> j;
        if (j.contains("weights")) m_network.weights = j["weights"].get<std::vector<std::vector<float>>>();
        if (j.contains("biases")) m_network.biases = j["biases"].get<std::vector<float>>();
        return true;
    }
    return false;
}

float DQNAgent::getEpsilon() const {
    return m_epsilon;
}

void DQNAgent::decayEpsilon() {
    m_epsilon *= 0.995f;
    m_epsilon = std::max(0.01f, m_epsilon);
}

std::string DQNAgent::getAgentInfo() const {
    return "DQN (Placeholder) | Neurons: " + std::to_string(m_network.hiddenSize) + " | ε: " + std::to_string(m_epsilon);
}

// Policy Gradient Agent Implementation (unchanged, keeping existing)
std::vector<float> PolicyGradientAgent::PolicyNetwork::forward(const std::vector<float>& input) {
    std::vector<float> output(outputSize);
    for (int o = 0; o < outputSize; ++o) {
        output[o] = 0.0f;
        for (size_t i = 0; i < input.size() && i < 20; ++i) {
            output[o] += input[i] * (0.1f + o * 0.05f);
        }
    }
    return output;
}

PolicyGradientAgent::PolicyGradientAgent() : m_rng(std::random_device{}()) {
    spdlog::info("PolicyGradientAgent: Initialized (placeholder implementation)");
}

Direction PolicyGradientAgent::getAction(const EnhancedState& state, bool training) {
    auto stateVector = state.toVector();
    auto probabilities = m_network.forward(stateVector);
    
    float sum = 0.0f;
    for (auto& p : probabilities) {
        p = std::exp(p);
        sum += p;
    }
    for (auto& p : probabilities) {
        p /= sum;
    }
    
    std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
    return static_cast<Direction>(dist(m_rng));
}

void PolicyGradientAgent::updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) {
    m_episodeRewards.push_back(reward);
}

// State Generator Implementation (unchanged, keeping existing)
EnhancedState StateGenerator::generateState(const Snake& snake, const Apple& apple, const Grid& grid) {
    EnhancedState state;
    
    state.basic = generateBasicState(snake, apple, grid);
    
    auto head = snake.getHeadPosition();
    auto food = apple.getPosition();
    
    state.distanceToFood = static_cast<float>(abs(head.x - food.x) + abs(head.y - food.y));
    
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
        density[i] = static_cast<float>(quadrantCounts[i]) / quadrantSize;
    }
}

float StateGenerator::calculatePathToFood(const Snake& snake, const Apple& apple, const Grid& grid) {
    auto head = snake.getHeadPosition();
    auto food = apple.getPosition();
    
    float baseDistance = static_cast<float>(abs(head.x - food.x) + abs(head.y - food.y));
    
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