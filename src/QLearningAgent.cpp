 
#include "QLearningAgent.hpp"
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

QLearningAgent::QLearningAgent(float learningRate, float discountFactor, float epsilon)
    : m_learningRate(learningRate)
    , m_discountFactor(discountFactor)
    , m_epsilon(epsilon)
    , m_rng(std::random_device{}())
    , m_dist(0.0f, 1.0f) {
}

QLearningAgent::~QLearningAgent() {
    saveQTable();
}

Direction QLearningAgent::getAction(const AgentState& state, bool training) {
    // Epsilon-greedy strategy
    if (training && m_dist(m_rng) < m_epsilon) {
        // Random action
        std::uniform_int_distribution<int> actionDist(0, 3);
        return indexToDirection(actionDist(m_rng));
    }
    
    // Greedy action
    return getMaxQAction(state);
}

void QLearningAgent::updateQValue(const AgentState& state, Direction action, 
                                  float reward, const AgentState& nextState) {
    float currentQ = getQValue(state, action);
    float maxNextQ = getQValue(nextState, getMaxQAction(nextState));
    
    // Q-learning update rule
    float newQ = currentQ + m_learningRate * (reward + m_discountFactor * maxNextQ - currentQ);
    
    // Update Q-table
    std::string stateKey = state.toString();
    int actionIndex = directionToIndex(action);
    
    if (m_qTable.find(stateKey) == m_qTable.end()) {
        m_qTable[stateKey] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    m_qTable[stateKey][actionIndex] = newQ;
}

AgentState QLearningAgent::getState(const Snake& snake, const Apple& apple, const Grid& grid) const {
    AgentState state;
    auto head = snake.getHeadPosition();
    auto food = apple.getPosition();
    auto direction = snake.getDirection();
    
    // Check dangers
    sf::Vector2i straight = head, left = head, right = head;
    
    switch (direction) {
        case Direction::UP:
            straight.y--;
            left.x--;
            right.x++;
            break;
        case Direction::DOWN:
            straight.y++;
            left.x++;
            right.x--;
            break;
        case Direction::LEFT:
            straight.x--;
            left.y++;
            right.y--;
            break;
        case Direction::RIGHT:
            straight.x++;
            left.y--;
            right.y++;
            break;
    }
    
    state.dangerStraight = !grid.isValidPosition(straight) || snake.isPositionOnSnake(straight);
    state.dangerLeft = !grid.isValidPosition(left) || snake.isPositionOnSnake(left);
    state.dangerRight = !grid.isValidPosition(right) || snake.isPositionOnSnake(right);
    state.currentDirection = direction;
    
    // Food location relative to head
    state.foodLeft = food.x < head.x;
    state.foodRight = food.x > head.x;
    state.foodUp = food.y < head.y;
    state.foodDown = food.y > head.y;
    
    return state;
}

float QLearningAgent::calculateReward(const Snake& snake, const Apple& apple, 
                                     bool ateFood, bool died) const {
    if (died) {
        return Reward::DEATH;
    }
    
    if (ateFood) {
        return Reward::EAT_FOOD;
    }
    
    // Distance-based reward
    auto head = snake.getHeadPosition();
    auto food = apple.getPosition();
    
    int currentDistance = std::abs(head.x - food.x) + std::abs(head.y - food.y);
    int previousDistance = std::abs(m_previousFoodDistance.x - food.x) + 
                          std::abs(m_previousFoodDistance.y - food.y);
    
    if (currentDistance < previousDistance) {
        return Reward::MOVE_TOWARDS_FOOD;
    } else {
        return Reward::MOVE_AWAY_FROM_FOOD;
    }
}

void QLearningAgent::saveQTable(const std::string& filename) {
    nlohmann::json j;
    
    for (const auto& [state, actions] : m_qTable) {
        j[state] = actions;
    }
    
    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(4);
        spdlog::info("Q-table saved to {}", filename);
    }
}

void QLearningAgent::loadQTable(const std::string& filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        nlohmann::json j;
        file >> j;
        
        m_qTable.clear();
        for (auto& [state, actions] : j.items()) {
            std::array<float, 4> actionValues;
            for (size_t i = 0; i < 4; ++i) {
                actionValues[i] = actions[i];
            }
            m_qTable[state] = actionValues;
        }
        
        spdlog::info("Q-table loaded from {}", filename);
    }
}

void QLearningAgent::decayEpsilon(float decayRate) {
    m_epsilon *= decayRate;
    m_epsilon = std::max(0.01f, m_epsilon); // Minimum epsilon
}

Direction QLearningAgent::getMaxQAction(const AgentState& state) const {
    std::string stateKey = state.toString();
    
    if (m_qTable.find(stateKey) == m_qTable.end()) {
        // Random action if state not seen
        return static_cast<Direction>(rand() % 4);
    }
    
    const auto& actions = m_qTable.at(stateKey);
    int maxIndex = 0;
    float maxValue = actions[0];
    
    for (int i = 1; i < 4; ++i) {
        if (actions[i] > maxValue) {
            maxValue = actions[i];
            maxIndex = i;
        }
    }
    
    return indexToDirection(maxIndex);
}

float QLearningAgent::getQValue(const AgentState& state, Direction action) const {
    std::string stateKey = state.toString();
    
    if (m_qTable.find(stateKey) == m_qTable.end()) {
        return 0.0f;
    }
    
    return m_qTable.at(stateKey)[directionToIndex(action)];
}

int QLearningAgent::directionToIndex(Direction dir) const {
    switch (dir) {
        case Direction::UP:    return 0;
        case Direction::DOWN:  return 1;
        case Direction::LEFT:  return 2;
        case Direction::RIGHT: return 3;
        default: return 0;
    }
}

Direction QLearningAgent::indexToDirection(int index) const {
    switch (index) {
        case 0: return Direction::UP;
        case 1: return Direction::DOWN;
        case 2: return Direction::LEFT;
        case 3: return Direction::RIGHT;
        default: return Direction::UP;
    }
}