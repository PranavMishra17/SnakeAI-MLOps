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

// Enhanced Q-Learning Agent Implementation (unchanged from previous)
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

// [Previous Q-Learning implementation methods remain the same]
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

// [Additional Q-Learning methods remain the same as previous implementation]

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

// [Other Q-Learning methods remain the same]

// Neural Network Agent Placeholders
DQNAgent::DQNAgent() : m_epsilon(0.1f), m_rng(std::random_device{}()) {
    spdlog::info("DQNAgent: Initialized (placeholder implementation)");
    spdlog::warn("DQNAgent: Neural network inference not implemented in C++");
    spdlog::info("DQNAgent: For full functionality, use Python evaluation or convert to ONNX");
}

Direction DQNAgent::getAction(const EnhancedState& state, bool training) {
    // Placeholder implementation - returns random action
    spdlog::debug("DQNAgent: Using placeholder random action (neural network not implemented)");
    std::uniform_int_distribution<int> actionDist(0, 3);
    return static_cast<Direction>(actionDist(m_rng));
}

bool DQNAgent::loadModel(const std::string& path) {
    spdlog::warn("DQNAgent: Neural network model loading not implemented in C++");
    spdlog::info("DQNAgent: Model file exists: {}", std::filesystem::exists(path) ? "Yes" : "No");
    spdlog::info("DQNAgent: To use DQN models, please use Python evaluation");
    return false;
}

void DQNAgent::saveModel(const std::string& path) {
    spdlog::warn("DQNAgent: Model saving not implemented (placeholder agent)");
}

std::string DQNAgent::getAgentInfo() const {
    return "DQN (C++ Placeholder) | Note: Use Python for full neural network support";
}

std::string DQNAgent::getModelInfo() const {
    return "DQN Placeholder - Neural network inference requires PyTorch/ONNX runtime";
}

// Policy Gradient Agent Placeholder
PolicyGradientAgent::PolicyGradientAgent() : m_rng(std::random_device{}()) {
    spdlog::info("PolicyGradientAgent: Initialized (placeholder implementation)");
    spdlog::warn("PolicyGradientAgent: Neural network inference not implemented in C++");
}

Direction PolicyGradientAgent::getAction(const EnhancedState& state, bool training) {
    spdlog::debug("PolicyGradientAgent: Using placeholder random action");
    std::uniform_int_distribution<int> actionDist(0, 3);
    return static_cast<Direction>(actionDist(m_rng));
}

bool PolicyGradientAgent::loadModel(const std::string& path) {
    spdlog::warn("PolicyGradientAgent: Model loading not implemented in C++");
    return false;
}

std::string PolicyGradientAgent::getAgentInfo() const {
    return "Policy Gradient (C++ Placeholder) | Use Python for neural network support";
}

// Actor-Critic Agent Placeholder
ActorCriticAgent::ActorCriticAgent() : m_rng(std::random_device{}()) {
    spdlog::info("ActorCriticAgent: Initialized (placeholder implementation)");
    spdlog::warn("ActorCriticAgent: Neural network inference not implemented in C++");
}

Direction ActorCriticAgent::getAction(const EnhancedState& state, bool training) {
    spdlog::debug("ActorCriticAgent: Using placeholder random action");
    std::uniform_int_distribution<int> actionDist(0, 3);
    return static_cast<Direction>(actionDist(m_rng));
}

bool ActorCriticAgent::loadModel(const std::string& path) {
    spdlog::warn("ActorCriticAgent: Model loading not implemented in C++");
    return false;
}

std::string ActorCriticAgent::getAgentInfo() const {
    return "Actor-Critic (C++ Placeholder) | Use Python for neural network support";
}

// Updated TrainedModelManager Implementation
TrainedModelManager::TrainedModelManager(const std::string& modelsDir) 
    : m_modelsDirectory(modelsDir) {
    spdlog::info("TrainedModelManager: Initialized with directory: {}", modelsDir);
    
    if (!std::filesystem::exists(modelsDir)) {
        std::filesystem::create_directories(modelsDir);
        spdlog::info("TrainedModelManager: Created models directory: {}", modelsDir);
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
    
    // Scan for Q-Learning models in qlearning subdirectory
    auto qlearningDir = std::filesystem::path(m_modelsDirectory) / "qlearning";
    if (std::filesystem::exists(qlearningDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(qlearningDir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                
                if (filename.substr(0, 7) == "qtable_" && 
                    filename.length() > 5 && filename.substr(filename.length() - 5) == ".json" &&
                    filename.find("checkpoint") == std::string::npos &&
                    filename.find("report") == std::string::npos) {
                    
                    std::string profile = filename.substr(7);
                    profile = profile.substr(0, profile.length() - 5);
                    
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
    
    // Scan for neural network models
    std::vector<std::string> neuralTechniques = {"dqn", "policy_gradient", "actor_critic"};
    for (const auto& technique : neuralTechniques) {
        auto techDir = std::filesystem::path(m_modelsDirectory) / technique;
        if (std::filesystem::exists(techDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(techDir)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    
                    // Look for .pth files (PyTorch models)
                    if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".pth" &&
                        filename.find("checkpoint") == std::string::npos &&
                        filename.find("best") == std::string::npos) {
                        
                        // Extract profile from filename (e.g., "dqn_balanced.pth" -> "balanced")
                        auto prefixLen = technique.substr(0, 2).length() + 1; // "dq_", "pg_", "ac_"
                        if (filename.length() > prefixLen + 4) {
                            std::string profile = filename.substr(prefixLen);
                            profile = profile.substr(0, profile.length() - 4); // Remove .pth
                            
                            TrainedModelInfo info;
                            info.name = technique + " " + profile;
                            info.profile = profile;
                            info.modelPath = entry.path().string();
                            info.modelType = technique;
                            info.description = technique + " neural network model (" + profile + " profile)";
                            info.isLoaded = false;
                            
                            // Note: We can't validate neural network models in C++ without PyTorch
                            m_availableModels.push_back(info);
                            spdlog::info("TrainedModelManager: Found {} model: {} ({})", 
                                         technique, profile, info.modelPath);
                            spdlog::warn("TrainedModelManager: Neural network model requires Python for execution");
                        }
                    }
                }
            }
        }
    }
    
    spdlog::info("TrainedModelManager: Found {} total models ({} Q-Learning, {} neural network)", 
                 m_availableModels.size(),
                 std::count_if(m_availableModels.begin(), m_availableModels.end(),
                              [](const auto& m) { return m.modelType == "qlearning"; }),
                 std::count_if(m_availableModels.begin(), m_availableModels.end(),
                              [](const auto& m) { return m.modelType != "qlearning"; }));
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

// Updated Agent Factory Implementation
std::unique_ptr<IAgent> AgentFactory::createAgent(const AgentConfig& config) {
    spdlog::info("AgentFactory: Creating agent of type: {} ({})", 
                 static_cast<int>(config.type), config.getAgentTypeString());
    
    switch (config.type) {
        case AgentType::Q_LEARNING:
            return std::make_unique<QLearningAgentEnhanced>(config.learningRate, config.discountFactor, config.epsilon);
        case AgentType::DEEP_Q_NETWORK:
            spdlog::warn("AgentFactory: DQN agent created as placeholder - use Python for full functionality");
            return std::make_unique<DQNAgent>();
        case AgentType::POLICY_GRADIENT:
            spdlog::warn("AgentFactory: Policy Gradient agent created as placeholder - use Python for full functionality");
            return std::make_unique<PolicyGradientAgent>();
        case AgentType::ACTOR_CRITIC:
            spdlog::warn("AgentFactory: Actor-Critic agent created as placeholder - use Python for full functionality");
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
        spdlog::warn("AgentFactory: Neural network model {} requires Python execution", modelInfo->name);
        spdlog::info("AgentFactory: Returning placeholder agent for C++ compatibility");
        
        // Return appropriate placeholder based on model type
        if (modelInfo->modelType == "dqn") {
            return std::make_unique<DQNAgent>();
        } else if (modelInfo->modelType == "policy_gradient") {
            return std::make_unique<PolicyGradientAgent>();
        } else if (modelInfo->modelType == "actor_critic") {
            return std::make_unique<ActorCriticAgent>();
        }
    }
    
    return nullptr;
}

std::vector<AgentConfig> AgentFactory::getAvailableTrainedAgents() {
    std::vector<AgentConfig> configs;
    TrainedModelManager manager;
    
    auto models = manager.getAvailableModels();
    for (const auto& model : models) {
        AgentConfig config;
        
        // Map model type string to enum
        if (model.modelType == "qlearning") {
            config.type = AgentType::Q_LEARNING;
        } else if (model.modelType == "dqn") {
            config.type = AgentType::DEEP_Q_NETWORK;
        } else if (model.modelType == "policy_gradient") {
            config.type = AgentType::POLICY_GRADIENT;
        } else if (model.modelType == "actor_critic") {
            config.type = AgentType::ACTOR_CRITIC;
        } else {
            continue; // Skip unknown types
        }
        
        config.name = model.name;
        config.description = model.description;
        config.isImplemented = (model.modelType == "qlearning"); // Only Q-Learning fully implemented in C++
        config.modelPath = model.modelPath;
        
        configs.push_back(config);
    }
    
    spdlog::info("AgentFactory: Found {} available trained agents ({} Q-Learning, {} neural network placeholders)", 
                 configs.size(),
                 std::count_if(configs.begin(), configs.end(), [](const auto& c) { return c.isImplemented; }),
                 std::count_if(configs.begin(), configs.end(), [](const auto& c) { return !c.isImplemented; }));
    
    return configs;
}

// State Generator Implementation (unchanged)
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

// TrainedModelManager missing methods
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

// QLearningAgentEnhanced missing methods
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

// Add to MLAgents.cpp:

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