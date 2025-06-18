#pragma once
#include <SFML/System/Vector2.hpp>
#include <string>
#include <vector>
#include <chrono>

enum class GameState {
    MENU,
    PLAYING,
    PAUSED,
    GAME_OVER,
    AGENT_SELECTION,
    LEADERBOARD,
    SETTINGS,
    HOW_TO_PLAY
};

enum class GameMode {
    SINGLE_PLAYER,      // Human controls snake, system spawns apples
    AGENT_VS_PLAYER,    // ML agent controls snake, human places apples
    AGENT_VS_SYSTEM     // ML agent controls snake, system spawns apples
};

enum class AgentType {
    HUMAN,
    Q_LEARNING,
    DEEP_Q_NETWORK,     
    PPO,                // CHANGED: Policy Gradient -> PPO
    ACTOR_CRITIC,       
    GENETIC_ALGORITHM   // Placeholder
};

enum class Direction {
    UP,
    DOWN,
    LEFT,
    RIGHT
};

// State representation for Q-Learning
struct AgentState {
    // Danger in each direction
    bool dangerStraight;
    bool dangerLeft;
    bool dangerRight;
    
    // Current direction
    Direction currentDirection;
    
    // Food location relative to head
    bool foodLeft;
    bool foodRight;
    bool foodUp;
    bool foodDown;
    
    // Convert to string for Q-table key
    std::string toString() const {
        return std::to_string(dangerStraight) + 
               std::to_string(dangerLeft) + 
               std::to_string(dangerRight) +
               std::to_string(static_cast<int>(currentDirection)) +
               std::to_string(foodLeft) +
               std::to_string(foodRight) +
               std::to_string(foodUp) +
               std::to_string(foodDown);
    }
};

// Enhanced state representation for neural networks
struct EnhancedState {
    // Basic features
    AgentState basic;
    
    // Distance features
    float distanceToFood;
    float distanceToWall[4];  // Up, Down, Left, Right
    float bodyDensity[4];     // Body segment density by quadrant
    
    // Additional features for advanced agents
    int snakeLength;
    int emptySpaces;
    float pathToFood;  // A* distance if path exists, -1 if blocked
    
    std::vector<float> toVector() const {
        return {
            static_cast<float>(basic.dangerStraight),
            static_cast<float>(basic.dangerLeft),
            static_cast<float>(basic.dangerRight),
            static_cast<float>(basic.currentDirection),
            static_cast<float>(basic.foodLeft),
            static_cast<float>(basic.foodRight),
            static_cast<float>(basic.foodUp),
            static_cast<float>(basic.foodDown),
            distanceToFood,
            distanceToWall[0], distanceToWall[1], distanceToWall[2], distanceToWall[3],
            bodyDensity[0], bodyDensity[1], bodyDensity[2], bodyDensity[3],
            static_cast<float>(snakeLength),
            static_cast<float>(emptySpaces),
            pathToFood
        };
    }
};

// Reward system
struct Reward {
    static constexpr float EAT_FOOD = 10.0f;
    static constexpr float DEATH = -10.0f;
    static constexpr float MOVE_TOWARDS_FOOD = 1.0f;
    static constexpr float MOVE_AWAY_FROM_FOOD = -1.0f;
    static constexpr float MOVE_PENALTY = -0.1f;
    static constexpr float EFFICIENCY_BONUS = 2.0f;
    static constexpr float SURVIVAL_BONUS = 0.05f;
    static constexpr float EXPLORATION_BONUS = 0.2f;
    static constexpr float SAFETY_BONUS = 0.1f;
    static constexpr float WALL_PENALTY = -0.5f;
    static constexpr float SELF_COLLISION_WARNING = -2.0f;
};

// Leaderboard entry
struct LeaderboardEntry {
    std::string playerName;
    AgentType agentType;
    int score;
    int episode;
    std::chrono::system_clock::time_point timestamp;
    float efficiency; // Score per episode ratio
    
    std::string getAgentTypeString() const {
        switch (agentType) {
            case AgentType::HUMAN: return "Human";
            case AgentType::Q_LEARNING: return "Q-Learning";
            case AgentType::DEEP_Q_NETWORK: return "DQN";
            case AgentType::PPO: return "PPO";  // CHANGED: Policy Gradient -> PPO
            case AgentType::ACTOR_CRITIC: return "Actor-Critic";
            case AgentType::GENETIC_ALGORITHM: return "Genetic Algorithm";
            default: return "Unknown";
        }
    }
    
    std::string getDisplayName() const {
        return playerName + " (" + getAgentTypeString() + ")";
    }
};

// Agent configuration
struct AgentConfig {
    AgentType type;
    std::string name;
    std::string description;
    bool isImplemented;
    std::string modelPath;  // For saved models
    
    // Hyperparameters (different for each agent type)
    float learningRate = 0.1f;
    float epsilon = 0.1f;
    float discountFactor = 0.95f;
    int hiddenLayers = 3;
    int neuronsPerLayer = 128;
    
    std::string getAgentTypeString() const {
        switch (type) {
            case AgentType::HUMAN: return "Human";
            case AgentType::Q_LEARNING: return "Q-Learning";
            case AgentType::DEEP_Q_NETWORK: return "DQN";
            case AgentType::PPO: return "PPO";  // CHANGED: Policy Gradient -> PPO
            case AgentType::ACTOR_CRITIC: return "Actor-Critic";
            case AgentType::GENETIC_ALGORITHM: return "Genetic Algorithm";
            default: return "Unknown";
        }
    }
};

// Settings structure
struct GameSettings {
    float minSpeed = 0.5f;
    float maxSpeed = 3.0f;
    float defaultSpeed = 1.0f;
    int gridSize = 20;
    bool showQValues = false;
    bool showTrainingStats = true;
    std::string leaderboardPath = "leaderboard.json";
};