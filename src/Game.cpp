 
#include "Game.hpp"
#include <filesystem>

Game::Game() 
    : m_currentState(GameState::MENU)
    , m_gameMode(GameMode::SINGLE_PLAYER)
    , m_moveTimer(0.0f)
    , m_moveSpeed(1.0f)
    , m_score(0)
    , m_episode(0)
    , m_gameOver(false)
    , m_hasNextApple(false) {
    
    spdlog::info("Initializing SnakeAI-MLOps Game");
    
    // Create fullscreen window
    auto videoMode = sf::VideoMode::getDesktopMode();
    m_window = std::make_unique<sf::RenderWindow>(videoMode, "SnakeAI-MLOps", sf::Style::Fullscreen);
    m_window->setFramerateLimit(60);
    
    // Calculate grid size to fit screen (square grid)
    int screenMin = std::min(videoMode.width, videoMode.height);
    int gridCells = 20; // 20x20 grid
    
    // Initialize components
    m_grid = std::make_unique<Grid>(gridCells);
    m_grid->initialize(*m_window);
    
    m_snake = std::make_unique<Snake>(m_grid.get());
    m_apple = std::make_unique<Apple>(m_grid.get());
    m_menu = std::make_unique<Menu>();
    m_menu->initialize(*m_window);
    
    m_agent = std::make_unique<QLearningAgent>();
    m_dataCollector = std::make_unique<DataCollector>();
    m_inputManager = std::make_unique<InputManager>();
    
    // Load Q-table if exists
    if (std::filesystem::exists("models/qtable.json")) {
        m_agent->loadQTable("models/qtable.json");
        spdlog::info("Loaded existing Q-table");
    }
    
    // Set menu callback
    m_menu->setSelectionCallback([this](GameMode mode) {
        handleMenuSelection(mode);
    });
}

Game::~Game() {
    // Save Q-table
    std::filesystem::create_directories("models");
    m_agent->saveQTable("models/qtable.json");
    m_dataCollector->saveSummary();
    spdlog::info("Game shutting down, data saved");
}

void Game::run() {
    while (m_window->isOpen()) {
        processEvents();
        
        float deltaTime = m_clock.restart().asSeconds();
        update(deltaTime);
        
        render();
    }
}

void Game::processEvents() {
    while (auto event = m_window->pollEvent()) {
        if (event->is<sf::Event::Closed>()) {
            m_window->close();
        }
        
        if (auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
            if (keyPressed->code == sf::Keyboard::Key::Escape) {
                if (m_currentState == GameState::PLAYING) {
                    m_currentState = GameState::PAUSED;
                } else if (m_currentState == GameState::PAUSED) {
                    m_currentState = GameState::PLAYING;
                } else if (m_currentState == GameState::GAME_OVER) {
                    m_currentState = GameState::MENU;
                }
            }
        }
        
        switch (m_currentState) {
            case GameState::MENU:
                m_menu->handleEvent(*event);
                break;
                
            case GameState::PLAYING:
                if (m_gameMode == GameMode::SINGLE_PLAYER) {
                    m_inputManager->handleSnakeInput(*event, *m_snake);
                } else if (m_gameMode == GameMode::AGENT_VS_PLAYER) {
                    // Mouse input for apple placement
                    if (auto* mousePressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                        if (mousePressed->button == sf::Mouse::Button::Left) {
                            auto mousePos = sf::Mouse::getPosition(*m_window);
                            auto gridPos = m_grid->screenToGrid(mousePos);
                            
                            if (m_grid->isValidPosition(gridPos) && 
                                !m_snake->isPositionOnSnake(gridPos)) {
                                if (!m_apple->isActive()) {
                                    m_apple->setPosition(gridPos);
                                    m_apple->setActive(true);
                                } else {
                                    m_nextApplePos = gridPos;
                                    m_hasNextApple = true;
                                }
                            }
                        }
                    }
                }
                
                // Speed adjustment
                if (auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                    if (keyPressed->code == sf::Keyboard::Key::Add) {
                        m_moveSpeed = std::min(m_maxSpeed, m_moveSpeed + 0.5f);
                        spdlog::info("Speed increased to {} blocks/sec", m_moveSpeed);
                    } else if (keyPressed->code == sf::Keyboard::Key::Subtract) {
                        m_moveSpeed = std::max(m_minSpeed, m_moveSpeed - 0.5f);
                        spdlog::info("Speed decreased to {} blocks/sec", m_moveSpeed);
                    }
                }
                break;
        }
    }
}

void Game::update(float deltaTime) {
    switch (m_currentState) {
        case GameState::MENU:
            m_menu->update();
            break;
            
        case GameState::PLAYING:
            if (!m_gameOver) {
                updateGame(deltaTime);
            }
            break;
    }
}

void Game::updateGame(float deltaTime) {
    m_moveTimer += deltaTime;
    
    if (m_moveTimer >= 1.0f / m_moveSpeed) {
        m_moveTimer = 0.0f;
        
        // Get current state for agent
        AgentState currentState = m_agent->getState(*m_snake, *m_apple, *m_grid);
        
        // Agent decision
        if (m_gameMode == GameMode::AGENT_VS_PLAYER || m_gameMode == GameMode::AGENT_VS_SYSTEM) {
            Direction action = m_agent->getAction(currentState, true);
            m_snake->setDirection(action);
        }
        
        // Store previous position for reward calculation
        auto prevHeadPos = m_snake->getHeadPosition();
        
        // Move snake
        m_snake->move();
        
        // Check collisions
        if (m_snake->checkSelfCollision() || m_snake->checkWallCollision()) {
            m_gameOver = true;
            m_currentState = GameState::GAME_OVER;
            
            // Update Q-values for death
            if (m_gameMode != GameMode::SINGLE_PLAYER) {
                float reward = m_agent->calculateReward(*m_snake, *m_apple, false, true);
                AgentState nextState = m_agent->getState(*m_snake, *m_apple, *m_grid);
                m_agent->updateQValue(currentState, m_snake->getDirection(), reward, nextState);
                m_dataCollector->recordStep(currentState, m_snake->getDirection(), reward);
            }
            
            m_dataCollector->endEpisode(m_score, true, m_agent->getEpsilon());
            spdlog::info("Game Over! Episode: {}, Score: {}", m_episode, m_score);
            return;
        }
        
        // Check apple collision
        bool ateFood = false;
        if (m_snake->getHeadPosition() == m_apple->getPosition() && m_apple->isActive()) {
            m_snake->grow();
            m_score++;
            ateFood = true;
            m_apple->setActive(false);
            
            // Spawn new apple
            if (m_gameMode == GameMode::SINGLE_PLAYER || m_gameMode == GameMode::AGENT_VS_SYSTEM) {
                m_apple->respawn(*m_snake);
            } else if (m_gameMode == GameMode::AGENT_VS_PLAYER && m_hasNextApple) {
                if (!m_snake->isPositionOnSnake(m_nextApplePos)) {
                    m_apple->setPosition(m_nextApplePos);
                    m_apple->setActive(true);
                    m_hasNextApple = false;
                }
            }
            
            spdlog::info("Apple eaten! Score: {}", m_score);
        }
        
        // Update Q-values
        if (m_gameMode != GameMode::SINGLE_PLAYER) {
            float reward = m_agent->calculateReward(*m_snake, *m_apple, ateFood, false);
            AgentState nextState = m_agent->getState(*m_snake, *m_apple, *m_grid);
            m_agent->updateQValue(currentState, m_snake->getDirection(), reward, nextState);
            m_dataCollector->recordStep(currentState, m_snake->getDirection(), reward);
        }
    }
}

void Game::render() {
    m_window->clear(sf::Color(30, 30, 30));
    
    switch (m_currentState) {
        case GameState::MENU:
            m_menu->render(*m_window);
            break;
            
        case GameState::PLAYING:
        case GameState::PAUSED:
        case GameState::GAME_OVER:
            m_grid->render(*m_window);
            m_snake->render(*m_window);
            m_apple->render(*m_window);
            
            // Render next apple preview in AGENT_VS_PLAYER mode
            if (m_gameMode == GameMode::AGENT_VS_PLAYER && m_hasNextApple) {
                sf::CircleShape preview(m_grid->getCellSize() / 2.0f * 0.8f);
                preview.setFillColor(sf::Color(255, 0, 0, 128)); // Semi-transparent red
                auto screenPos = m_grid->gridToScreen(m_nextApplePos);
                preview.setPosition(screenPos);
                m_window->draw(preview);
            }
            
            // Render UI
            renderUI();
            break;
    }
    
    m_window->display();
}

void Game::renderUI() {
    // Create UI background
    sf::RectangleShape uiPanel(sf::Vector2f(300, 200));
    uiPanel.setPosition(10, 10);
    uiPanel.setFillColor(sf::Color(0, 0, 0, 180));
    m_window->draw(uiPanel);
    
    // Render stats
    sf::Font font;
    if (font.loadFromFile("assets/fonts/arial.ttf")) {
        sf::Text scoreText("Score: " + std::to_string(m_score), font, 20);
        scoreText.setPosition(20, 20);
        m_window->draw(scoreText);
        
        sf::Text episodeText("Episode: " + std::to_string(m_episode), font, 20);
        episodeText.setPosition(20, 50);
        m_window->draw(episodeText);
        
        sf::Text speedText("Speed: " + std::to_string(m_moveSpeed) + " blocks/sec", font, 20);
        speedText.setPosition(20, 80);
        m_window->draw(speedText);
        
        if (m_gameMode != GameMode::SINGLE_PLAYER) {
            sf::Text epsilonText("Epsilon: " + std::to_string(m_agent->getEpsilon()), font, 20);
            epsilonText.setPosition(20, 110);
            m_window->draw(epsilonText);
        }
        
        if (m_currentState == GameState::GAME_OVER) {
            sf::Text gameOverText("GAME OVER - Press ESC for menu", font, 30);
            gameOverText.setPosition(m_window->getSize().x / 2 - 250, m_window->getSize().y / 2);
            gameOverText.setFillColor(sf::Color::Red);
            m_window->draw(gameOverText);
        }
    }
}

void Game::handleMenuSelection(GameMode mode) {
    m_gameMode = mode;
    startGame();
}

void Game::startGame() {
    m_currentState = GameState::PLAYING;
    m_episode++;
    resetGame();
    m_dataCollector->startEpisode(m_episode);
    
    spdlog::info("Starting game - Mode: {}, Episode: {}", 
                 static_cast<int>(m_gameMode), m_episode);
}

void Game::resetGame() {
    m_score = 0;
    m_gameOver = false;
    m_moveTimer = 0.0f;
    m_hasNextApple = false;
    
    m_snake->reset();
    
    if (m_gameMode == GameMode::SINGLE_PLAYER || m_gameMode == GameMode::AGENT_VS_SYSTEM) {
        m_apple->respawn(*m_snake);
    } else {
        m_apple->setActive(false);
    }
    
    // Decay epsilon for agent
    if (m_gameMode != GameMode::SINGLE_PLAYER) {
        m_agent->decayEpsilon();
    }
}