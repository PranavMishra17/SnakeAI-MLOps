#include "Game.hpp"
#include "MLAgents.hpp"
#include "PauseMenu.hpp"
#include "AgentSelection.hpp"
#include "Leaderboard.hpp"
#include <filesystem>

Game::Game() 
    : m_currentState(GameState::MENU)
    , m_gameMode(GameMode::SINGLE_PLAYER)
    , m_currentAgentType(AgentType::HUMAN)
    , m_moveTimer(0.0f)
    , m_moveSpeed(1.0f)
    , m_score(0)
    , m_episode(0)
    , m_gameOver(false)
    , m_hasNextApple(false) {
    
    spdlog::info("Game: Starting initialization...");
    
    try {
        // Create window
        auto videoMode = sf::VideoMode::getDesktopMode();
        m_window = std::make_unique<sf::RenderWindow>(videoMode, "SnakeAI-MLOps", sf::Style::Default);
        m_window->setFramerateLimit(60);
        
        int gridCells = 20;
        
        // Initialize core components
        m_grid = std::make_unique<Grid>(gridCells);
        m_grid->initialize(*m_window);
        
        m_snake = std::make_unique<Snake>(m_grid.get());
        m_apple = std::make_unique<Apple>(m_grid.get());
        
        // Initialize UI components
        m_menu = std::make_unique<Menu>();
        m_menu->initialize(*m_window);
        
        m_pauseMenu = std::make_unique<PauseMenu>();
        m_pauseMenu->initialize(*m_window);
        
        m_agentSelection = std::make_unique<AgentSelection>();
        m_agentSelection->initialize(*m_window);
        
        m_leaderboard = std::make_unique<Leaderboard>();
        m_leaderboard->initialize(*m_window);
        
        m_dataCollector = std::make_unique<DataCollector>();
        m_inputManager = std::make_unique<InputManager>();
        
        // Initialize default agent (Q-Learning)
        AgentConfig defaultConfig;
        defaultConfig.type = AgentType::Q_LEARNING;
        defaultConfig.name = "Q-Learning Agent";
        defaultConfig.isImplemented = true;
        m_currentAgent = AgentFactory::createAgent(defaultConfig);
        m_currentAgentConfig = defaultConfig;
        
        // Load existing model
        if (std::filesystem::exists("models/qtable.json")) {
            m_currentAgent->loadModel("models/qtable.json");
        }
        
        setupCallbacks();
        
        spdlog::info("Game: Initialization complete!");
        
    } catch (const std::exception& e) {
        spdlog::error("Game: Exception during initialization: {}", e.what());
        throw;
    }
}

Game::~Game() {
    if (m_currentAgent) {
        std::filesystem::create_directories("models");
        m_currentAgent->saveModel("models/" + m_currentAgentConfig.modelPath);
    }
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

void Game::setupCallbacks() {
    // Menu callbacks
    m_menu->setSelectionCallback([this](GameMode mode) {
        handleMenuSelection(mode);
    });
    
    // Pause menu callbacks
    m_pauseMenu->setResumeCallback([this]() {
        m_currentState = GameState::PLAYING;
    });
    
    m_pauseMenu->setRestartCallback([this]() {
        resetGame();
        m_currentState = GameState::PLAYING;
    });
    
    m_pauseMenu->setMainMenuCallback([this]() {
        m_currentState = GameState::MENU;
    });
    
    m_pauseMenu->setSpeedCallback([this](float speed) {
        m_moveSpeed = speed;
    });
    
    // Agent selection callbacks
    m_agentSelection->setSelectionCallback([this](const AgentConfig& config) {
        selectAgent(config);
        m_currentState = GameState::MENU;
    });
    
    m_agentSelection->setBackCallback([this]() {
        m_currentState = GameState::MENU;
    });
    
    // Leaderboard callbacks
    m_leaderboard->setBackCallback([this]() {
        m_currentState = GameState::MENU;
    });
}

void Game::processEvents() {
    while (auto event = m_window->pollEvent()) {
        if (event->is<sf::Event::Closed>()) {
            m_window->close();
        }
        
        handleGlobalKeys(*event);
        
        switch (m_currentState) {
            case GameState::MENU:
                m_menu->handleEvent(*event);
                break;
            case GameState::AGENT_SELECTION:
                m_agentSelection->handleEvent(*event);
                break;
            case GameState::LEADERBOARD:
                m_leaderboard->handleEvent(*event);
                break;
            case GameState::PLAYING:
                handleGameplayEvents(*event);
                break;
            case GameState::PAUSED:
                m_pauseMenu->handleEvent(*event);
                break;
            case GameState::GAME_OVER:
                handleGameOverEvents(*event);
                break;
        }
    }
}

void Game::handleGlobalKeys(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        switch (keyPressed->code) {
            case sf::Keyboard::Key::F1:
                m_currentState = GameState::LEADERBOARD;
                break;
            case sf::Keyboard::Key::F2:
                if (m_gameMode != GameMode::SINGLE_PLAYER) {
                    m_currentState = GameState::AGENT_SELECTION;
                }
                break;
        }
    }
}

void Game::handleGameplayEvents(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if (keyPressed->code == sf::Keyboard::Key::Escape) {
            m_pauseMenu->setCurrentSpeed(m_moveSpeed);
            m_pauseMenu->setCurrentAgent(m_currentAgentConfig);
            m_pauseMenu->setGameStats(m_score, m_episode, m_currentAgent->getEpsilon());
            m_currentState = GameState::PAUSED;
            return;
        }
        
        // Speed adjustment
        if (keyPressed->code == sf::Keyboard::Key::Equal) {
            m_moveSpeed = std::min(m_maxSpeed, m_moveSpeed + 0.5f);
            spdlog::info("Speed increased to {} blocks/sec", m_moveSpeed);
        } else if (keyPressed->code == sf::Keyboard::Key::Hyphen) {
            m_moveSpeed = std::max(m_minSpeed, m_moveSpeed - 0.5f);
            spdlog::info("Speed decreased to {} blocks/sec", m_moveSpeed);
        }
    }
    
    // Handle input based on game mode
    if (m_gameMode == GameMode::SINGLE_PLAYER) {
        m_inputManager->handleSnakeInput(event, *m_snake);
    } else if (m_gameMode == GameMode::AGENT_VS_PLAYER) {
        handleApplePlacement(event);
    }
}

void Game::handleApplePlacement(const sf::Event& event) {
    if (auto* mousePressed = event.getIf<sf::Event::MouseButtonPressed>()) {
        if (mousePressed->button == sf::Mouse::Button::Left) {
            auto mousePos = sf::Mouse::getPosition(*m_window);
            auto gridPos = m_grid->screenToGrid(mousePos);
            
            if (m_grid->isValidPosition(gridPos) && !m_snake->isPositionOnSnake(gridPos)) {
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

void Game::handleGameOverEvents(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if (keyPressed->code == sf::Keyboard::Key::Escape) {
            // Prompt for leaderboard entry if high score
            if (shouldAddToLeaderboard()) {
                m_leaderboard->promptForName(m_score, m_currentAgentType, m_episode);
                m_currentState = GameState::LEADERBOARD;
            } else {
                m_currentState = GameState::MENU;
            }
        } else if (keyPressed->code == sf::Keyboard::Key::R) {
            resetGame();
            m_currentState = GameState::PLAYING;
        }
    }
}

void Game::update(float deltaTime) {
    switch (m_currentState) {
        case GameState::MENU:
            m_menu->update();
            break;
        case GameState::AGENT_SELECTION:
            m_agentSelection->update();
            break;
        case GameState::LEADERBOARD:
            m_leaderboard->update();
            break;
        case GameState::PLAYING:
            if (!m_gameOver) {
                updateGame(deltaTime);
            }
            break;
        case GameState::PAUSED:
            m_pauseMenu->update();
            break;
    }
}

void Game::updateGame(float deltaTime) {
    m_moveTimer += deltaTime;
    
    if (m_moveTimer >= 1.0f / m_moveSpeed) {
        m_moveTimer = 0.0f;
        
        // Generate enhanced state
        EnhancedState currentState = StateGenerator::generateState(*m_snake, *m_apple, *m_grid);
        
        // Agent decision (for AI modes)
        if (m_gameMode != GameMode::SINGLE_PLAYER) {
            Direction action = m_currentAgent->getAction(currentState, true);
            m_snake->setDirection(action);
        }
        
        // Move snake
        m_snake->move();
        
        // Check collisions
        if (m_snake->checkSelfCollision() || m_snake->checkWallCollision()) {
            handleSnakeDeath(currentState);
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
            spawnNewApple();
            
            spdlog::info("Apple eaten! Score: {}", m_score);
        }
        
        // Update agent (for AI modes)
        if (m_gameMode != GameMode::SINGLE_PLAYER) {
            float reward = calculateReward(ateFood, false);
            EnhancedState nextState = StateGenerator::generateState(*m_snake, *m_apple, *m_grid);
            
            m_currentAgent->updateAgent(currentState, m_snake->getDirection(), reward, nextState);
            m_dataCollector->recordStep(currentState.basic, m_snake->getDirection(), reward);
        }
    }
}

void Game::handleSnakeDeath(const EnhancedState& currentState) {
    m_gameOver = true;
    m_currentState = GameState::GAME_OVER;
    
    // Update agent with death penalty
    if (m_gameMode != GameMode::SINGLE_PLAYER) {
        float reward = calculateReward(false, true);
        EnhancedState nextState = StateGenerator::generateState(*m_snake, *m_apple, *m_grid);
        
        m_currentAgent->updateAgent(currentState, m_snake->getDirection(), reward, nextState);
        m_dataCollector->recordStep(currentState.basic, m_snake->getDirection(), reward);
        
        // Decay exploration
        m_currentAgent->decayEpsilon();
    }
    
    m_dataCollector->endEpisode(m_score, true, m_currentAgent->getEpsilon());
    spdlog::info("Game Over! Episode: {}, Score: {}, Agent: {}", 
                 m_episode, m_score, m_currentAgentConfig.name);
}

float Game::calculateReward(bool ateFood, bool died) const {
    if (died) return Reward::DEATH;
    if (ateFood) return Reward::EAT_FOOD;
    
    // Distance-based reward (simplified)
    auto head = m_snake->getHeadPosition();
    auto food = m_apple->getPosition();
    
    int currentDistance = abs(head.x - food.x) + abs(head.y - food.y);
    
    // Simple reward shaping
    if (currentDistance < m_previousDistance) {
        m_previousDistance = currentDistance;
        return Reward::MOVE_TOWARDS_FOOD;
    } else {
        m_previousDistance = currentDistance;
        return Reward::MOVE_AWAY_FROM_FOOD;
    }
}

void Game::spawnNewApple() {
    if (m_gameMode == GameMode::SINGLE_PLAYER || m_gameMode == GameMode::AGENT_VS_SYSTEM) {
        m_apple->respawn(*m_snake);
    } else if (m_gameMode == GameMode::AGENT_VS_PLAYER && m_hasNextApple) {
        if (!m_snake->isPositionOnSnake(m_nextApplePos)) {
            m_apple->setPosition(m_nextApplePos);
            m_apple->setActive(true);
            m_hasNextApple = false;
        }
    }
}

void Game::render() {
    m_window->clear(sf::Color(30, 30, 30));
    
    switch (m_currentState) {
        case GameState::MENU:
            m_menu->render(*m_window);
            break;
        case GameState::AGENT_SELECTION:
            m_agentSelection->render(*m_window);
            break;
        case GameState::LEADERBOARD:
            m_leaderboard->render(*m_window);
            break;
        case GameState::PLAYING:
        case GameState::PAUSED:
        case GameState::GAME_OVER:
            renderGame();
            if (m_currentState == GameState::PAUSED) {
                m_pauseMenu->render(*m_window);
            }
            break;
    }
    
    m_window->display();
}

void Game::renderGame() {
    m_grid->render(*m_window);
    m_snake->render(*m_window);
    m_apple->render(*m_window);
    
    // Render next apple preview
    if (m_gameMode == GameMode::AGENT_VS_PLAYER && m_hasNextApple) {
        sf::CircleShape preview(m_grid->getCellSize() / 2.0f * 0.8f);
        preview.setFillColor(sf::Color(255, 0, 0, 128));
        auto screenPos = m_grid->gridToScreen(m_nextApplePos);
        preview.setPosition(screenPos);
        m_window->draw(preview);
    }
    
    renderUI();
}

void Game::renderUI() {
    // UI background panel
    sf::RectangleShape uiPanel(sf::Vector2f(350.0f, 250.0f));
    uiPanel.setPosition(sf::Vector2f(10.0f, 10.0f));
    uiPanel.setFillColor(sf::Color(0, 0, 0, 180));
    uiPanel.setOutlineThickness(1.0f);
    uiPanel.setOutlineColor(sf::Color(100, 100, 100));
    m_window->draw(uiPanel);
    
    // Load font
    sf::Font font;
    if (!font.openFromFile("assets/fonts/arial.ttf")) return;
    
    // Game stats
    std::vector<std::string> lines = {
        "Score: " + std::to_string(m_score),
        "Episode: " + std::to_string(m_episode),
        "Speed: " + std::to_string(m_moveSpeed).substr(0, 4) + " blocks/sec",
        "Agent: " + m_currentAgentConfig.name
    };
    
    if (m_currentAgentType != AgentType::HUMAN) {
        lines.push_back("Epsilon: " + std::to_string(m_currentAgent->getEpsilon()).substr(0, 5));
        lines.push_back(m_currentAgent->getAgentInfo());
    }
    
    lines.push_back("");
    lines.push_back("Controls:");
    lines.push_back("ESC: Pause | +/-: Speed");
    lines.push_back("F1: Leaderboard | F2: Change Agent");
    
    for (size_t i = 0; i < lines.size(); ++i) {
        sf::Text text(font);
        text.setString(lines[i]);
        text.setCharacterSize(16);
        text.setPosition(sf::Vector2f(20.0f, 20.0f + i * 20.0f));
        text.setFillColor(i < 6 ? sf::Color::White : sf::Color(180, 180, 180));
        m_window->draw(text);
    }
    
    // Game over overlay
    if (m_currentState == GameState::GAME_OVER) {
        sf::RectangleShape overlay(sf::Vector2f(600.0f, 200.0f));
        overlay.setPosition(sf::Vector2f(m_window->getSize().x / 2.0f - 300.0f, 
                                        m_window->getSize().y / 2.0f - 100.0f));
        overlay.setFillColor(sf::Color(0, 0, 0, 200));
        overlay.setOutlineThickness(3.0f);
        overlay.setOutlineColor(sf::Color::Red);
        m_window->draw(overlay);
        
        std::vector<std::string> gameOverLines = {
            "GAME OVER",
            "",
            "Final Score: " + std::to_string(m_score),
            "Agent: " + m_currentAgentConfig.getAgentTypeString(),
            "",
            "ESC: Menu | R: Restart"
        };
        
        for (size_t i = 0; i < gameOverLines.size(); ++i) {
            sf::Text text(font);
            text.setString(gameOverLines[i]);
            text.setCharacterSize(i == 0 ? 36 : 20);
            text.setFillColor(i == 0 ? sf::Color::Red : sf::Color::White);
            
            auto bounds = text.getLocalBounds();
            m_window->draw(text);
        }
    }
}

void Game::handleMenuSelection(GameMode mode) {
    m_gameMode = mode;
    
    if (mode == GameMode::SINGLE_PLAYER) {
        m_currentAgentType = AgentType::HUMAN;
        startGame();
    } else {
        // Show agent selection for AI modes
        m_currentState = GameState::AGENT_SELECTION;
    }
}

void Game::selectAgent(const AgentConfig& config) {
    if (!config.isImplemented) {
        spdlog::warn("Agent {} is not implemented yet", config.name);
        return;
    }
    
    m_currentAgent = AgentFactory::createAgent(config);
    m_currentAgentConfig = config;
    m_currentAgentType = config.type;
    
    // Load existing model
    std::string modelPath = "models/" + config.modelPath;
    if (std::filesystem::exists(modelPath)) {
        m_currentAgent->loadModel(modelPath);
        spdlog::info("Loaded model for agent: {}", config.name);
    }
    
    startGame();
}

void Game::startGame() {
    m_currentState = GameState::PLAYING;
    m_episode++;
    resetGame();
    m_dataCollector->startEpisode(m_episode);
    
    spdlog::info("Starting game - Mode: {}, Episode: {}, Agent: {}", 
                 static_cast<int>(m_gameMode), m_episode, m_currentAgentConfig.name);
}

void Game::resetGame() {
    m_score = 0;
    m_gameOver = false;
    m_moveTimer = 0.0f;
    m_hasNextApple = false;
    m_previousDistance = 999;
    
    m_snake->reset();
    
    if (m_gameMode == GameMode::SINGLE_PLAYER || m_gameMode == GameMode::AGENT_VS_SYSTEM) {
        m_apple->respawn(*m_snake);
    } else {
        m_apple->setActive(false);
    }
}

bool Game::shouldAddToLeaderboard() const {
    // Add to leaderboard if score > 5 or is human player
    return m_score > 5 || m_currentAgentType == AgentType::HUMAN;
}