 #pragma once
#include <map>
#include <string>
#include <random>
#include <fstream>
#include <nlohmann/json.hpp>
#include "GameState.hpp"
#include "Snake.hpp"
#include "Apple.hpp"
#include "Grid.hpp"

class QLearningAgent {
public:
    QLearningAgent(float learningRate = 0.1f, float discountFactor = 0.9f, float epsilon = 0.1f);
    ~QLearningAgent();
    
    Direction getAction(const AgentState& state, bool training = true);
    void updateQValue(const AgentState& state, Direction action, float reward, const AgentState& nextState);
    
    AgentState getState(const Snake& snake, const Apple& apple, const Grid& grid) const;
    float calculateReward(const Snake& snake, const Apple& apple, bool ateFood, bool died) const;
    
    void saveQTable(const std::string& filename = "qtable.json");
    void loadQTable(const std::string& filename = "qtable.json");
    
    void setEpsilon(float epsilon) { m_epsilon = epsilon; }
    float getEpsilon() const { return m_epsilon; }
    
    void decayEpsilon(float decayRate = 0.995f);
    
private:
    std::map<std::string, std::array<float, 4>> m_qTable;
    float m_learningRate;
    float m_discountFactor;
    float m_epsilon; // Exploration rate
    
    std::mt19937 m_rng;
    std::uniform_real_distribution<float> m_dist;
    
    Direction getMaxQAction(const AgentState& state) const;
    float getQValue(const AgentState& state, Direction action) const;
    int directionToIndex(Direction dir) const;
    Direction indexToDirection(int index) const;
    
    sf::Vector2i m_previousFoodDistance;
};
