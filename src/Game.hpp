#pragma once
#include <SFML/Graphics.hpp>
#include <memory>
#include <spdlog/spdlog.h>
#include "GameState.hpp"
#include "Grid.hpp"
#include "Snake.hpp"
#include "Apple.hpp"
#include "Menu.hpp"
#include "QLearningAgent.hpp"
#include "DataCollector.hpp"
#include "InputManager.hpp"

class Game {
public:
    Game();
    ~Game();
    void run();

private:
    void processEvents();
    void update(float deltaTime);
    void render();
    void renderUI();
    void handleMenuSelection(GameMode mode);
    void startGame();
    void resetGame();
    void updateGame(float deltaTime);
    
    // Game components
    std::unique_ptr<sf::RenderWindow> m_window;
    std::unique_ptr<Grid> m_grid;
    std::unique_ptr<Snake> m_snake;
    std::unique_ptr<Apple> m_apple;
    std::unique_ptr<Menu> m_menu;
    std::unique_ptr<QLearningAgent> m_agent;
    std::unique_ptr<DataCollector> m_dataCollector;
    std::unique_ptr<InputManager> m_inputManager;
    
    // Game state
    GameState m_currentState;
    GameMode m_gameMode;
    
    // Timing
    sf::Clock m_clock;
    sf::Clock m_gameClock;
    float m_moveTimer;
    float m_moveSpeed; // blocks per second
    
    // Game statistics
    int m_score;
    int m_episode;
    bool m_gameOver;
    
    // Apple placement for agent vs player mode
    sf::Vector2i m_nextApplePos;
    bool m_hasNextApple;
    
    // Settings
    float m_minSpeed = 0.5f;
    float m_maxSpeed = 3.0f;
};