#include "MLAgents.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <cmath>

// Enhanced Q-Learning Agent Implementation
QLearningAgentEnhanced::QLearningAgentEnhanced(float lr, float gamma, float eps)
    : m_learningRate(lr), m_discountFactor(gamma), m_epsilon(eps), m_rng(std::random_device{}()) {
    spdlog::info("QLearningAgentEnhanced: Initialized with lr={}, gamma={}, epsilon={}", lr, gamma, eps);
}

Direction QLearningAgentEnhanced::getAction(const EnhancedState& state, bool training) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    if (training && dist(m_rng) < m_epsilon) {
        std::uniform_int_distribution<int> actionDist(0, 3);
        return static_cast<Direction>(actionDist(m_rng));
    }
    
    return getMaxQAction(state.basic);
}

void QLearningAgentEnhanced::updateAgent(const EnhancedState& state, Direction action, float reward, const EnhancedState& nextState) {
    std::string stateKey = state.basic.toString();
    std::string nextStateKey = nextState.basic.toString();
    int actionIdx = static_cast<int>(action);
    
    if (m_qTable.find(stateKey) == m_qTable.end()) {
        m_qTable[stateKey] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    if (m_qTable.find(nextStateKey) == m_qTable.end()) {
        m_qTable[nextStateKey] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    
    float currentQ = m_qTable[stateKey][actionIdx];
    float maxNextQ = *std::max_element(m_qTable[nextStateKey].begin(), m_qTable[nextStateKey].end());
    
    m_qTable[stateKey][actionIdx] = currentQ + m_learningRate * (reward + m_discountFactor * maxNextQ - currentQ);
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
        
        std::ofstream file(path);
        if (file.is_open()) {
            file << j.dump(4);
            spdlog::info("QLearningAgentEnhanced: Model saved to {}", path);
        }
    } catch (const std::exception& e) {
        spdlog::error("QLearningAgentEnhanced: Failed to save model: {}", e.what());
    }
}

void QLearningAgentEnhanced::loadModel(const std::string& path) {
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            spdlog::warn("QLearningAgentEnhanced: Model file {} not found", path);
            return;
        }
        
        nlohmann::json j;
        file >> j;
        
        m_qTable.clear();
        if (j.contains("qTable")) {
            for (auto& [state, actions] : j["qTable"].items()) {
                std::array<float, 4> actionValues;
                for (size_t i = 0; i < 4; ++i) {
                    actionValues[i] = actions[i];
                }
                m_qTable[state] = actionValues;
            }
        }
        
        if (j.contains("hyperparameters")) {
            auto params = j["hyperparameters"];
            if (params.contains("epsilon")) {
                m_epsilon = params["epsilon"];
            }
        }
        
        spdlog::info("QLearningAgentEnhanced: Model loaded from {} with {} states", path, m_qTable.size());
    } catch (const std::exception& e) {
        spdlog::error("QLearningAgentEnhanced: Failed to load model: {}", e.what());
    }
}

float QLearningAgentEnhanced::getEpsilon() const {
    return m_epsilon;
}

void QLearningAgentEnhanced::decayEpsilon() {
    m_epsilon *= 0.995f;
    m_epsilon = std::max(0.01f, m_epsilon);
}

std::string QLearningAgentEnhanced::getAgentInfo() const {
    return "Q-Learning | States: " + std::to_string(m_qTable.size()) + " | ε: " + std::to_string(m_epsilon);
}

Direction QLearningAgentEnhanced::getMaxQAction(const AgentState& state) const {
    std::string stateKey = state.toString();
    
    if (m_qTable.find(stateKey) == m_qTable.end()) {
        std::uniform_int_distribution<int> dist(0, 3);
        return static_cast<Direction>(dist(m_rng));
    }
    
    const auto& actions = m_qTable.at(stateKey);
    int maxIdx = std::distance(actions.begin(), std::max_element(actions.begin(), actions.end()));
    return static_cast<Direction>(maxIdx);
}

// DQN Agent Implementation
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

void DQNAgent::loadModel(const std::string& path) {
    std::ifstream file(path);
    if (file.is_open()) {
        nlohmann::json j;
        file >> j;
        if (j.contains("weights")) m_network.weights = j["weights"].get<std::vector<std::vector<float>>>();
        if (j.contains("biases")) m_network.biases = j["biases"].get<std::vector<float>>();
    }
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

// Policy Gradient Agent Implementation
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

// Agent Factory Implementation
std::unique_ptr<IAgent> AgentFactory::createAgent(const AgentConfig& config) {
    switch (config.type) {
        case AgentType::Q_LEARNING:
            return std::make_unique<QLearningAgentEnhanced>(config.learningRate, config.discountFactor, config.epsilon);
        case AgentType::DEEP_Q_NETWORK:
            return std::make_unique<DQNAgent>();
        case AgentType::POLICY_GRADIENT:
            return std::make_unique<PolicyGradientAgent>();
        default:
            return std::make_unique<QLearningAgentEnhanced>();
    }
}

// State Generator Implementation
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