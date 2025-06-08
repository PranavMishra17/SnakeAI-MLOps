#pragma once
#include <SFML/System/Vector2.hpp>

enum class GameState {
    MENU,
    PLAYING,
    PAUSED,
    GAME_OVER
};

enum class GameMode {
    SINGLE_PLAYER,      // Human controls snake, system spawns apples
    AGENT_VS_PLAYER,    // ML agent controls snake, human places apples
    AGENT_VS_SYSTEM     // ML agent controls snake, system spawns apples
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

// Reward system
struct Reward {
    static constexpr float EAT_FOOD = 10.0f;
    static constexpr float DEATH = -10.0f;
    static constexpr float MOVE_TOWARDS_FOOD = 1.0f;
    static constexpr float MOVE_AWAY_FROM_FOOD = -1.0f;
}; 
