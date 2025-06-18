#include "Game.hpp"
#include "MLAgents.hpp"
#include "PauseMenu.hpp"
#include "AgentSelection.hpp"
#include "Leaderboard.hpp"
#include "Grid.hpp"
#include "Snake.hpp"
#include "Apple.hpp"
#include "UnifiedDataCollector.hpp"
#include "InputManager.hpp"
#include <filesystem>
#include <spdlog/spdlog.h>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <optional>
#include <sstream>

// Constructor
Game::Game()
    : m_currentState(GameState::MENU)
    , m_gameMode(GameMode::SINGLE_PLAYER)
    , m_currentAgentType(AgentType::HUMAN)
    , m_moveTimer(0.0f)
    , m_moveSpeed(1.0f)
    , m_score(0)
    , m_episode(0)
    , m_gameOver(false)
    , m_hasNextApple(false)
    , m_minSpeed(1.0f)
    , m_maxSpeed(10.0f)
    , m_previousDistance(999)
{
    spdlog::info("Game: Starting initialization...");

    try {
        std::filesystem::create_directories("models");
        std::filesystem::create_directories("data");
        std::filesystem::create_directories("logs");
        std::filesystem::create_directories("assets/fonts");

        auto videoMode = sf::VideoMode::getDesktopMode();
        m_window = std::make_unique<sf::RenderWindow>(videoMode, "SnakeAI-MLOps", sf::Style::Default);
        m_window->setFramerateLimit(60);

        int gridCells = 20;

        m_grid = std::make_unique<Grid>(gridCells);
        m_grid->initialize(*m_window);

        m_snake = std::make_unique<Snake>(m_grid.get());
        m_apple = std::make_unique<Apple>(m_grid.get());

        m_menu = std::make_unique<Menu>();
        m_menu->initialize(*m_window);

        m_pauseMenu = std::make_unique<PauseMenu>();
        m_pauseMenu->initialize(*m_window);

        m_agentSelection = std::make_unique<AgentSelection>();
        m_agentSelection->initialize(*m_window);

        m_leaderboard = std::make_unique<Leaderboard>();
        m_leaderboard->initialize(*m_window);

        m_dataCollector = std::make_unique<UnifiedDataCollector>();
        m_inputManager = std::make_unique<InputManager>();

        AgentConfig defaultConfig;
        defaultConfig.type = AgentType::Q_LEARNING;
        defaultConfig.name = "Q-Learning Agent";
        defaultConfig.isImplemented = true;
        m_currentAgent = AgentFactory::createAgent(defaultConfig);
        m_currentAgentConfig = defaultConfig;

        m_dataCollector->setAgentType(AgentType::Q_LEARNING);

        setupCallbacks();

        spdlog::info("Game: Initialization complete with trained model support!");

    } catch (const std::exception& e) {
        spdlog::error("Game: Exception during initialization: {}", e.what());
        throw;
    }
}

Game::~Game() {
    if (m_currentAgent) {
        std::filesystem::create_directories("models");

        if (m_currentAgentConfig.modelPath.empty()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            std::tm tm_buffer;
            #ifdef _MSC_VER
                localtime_s(&tm_buffer, &time_t);
            #else
                localtime_r(&time_t, &tm_buffer);
            #endif

            std::stringstream ss;
            ss << "qtable_session_" << std::put_time(&tm_buffer, "%Y%m%d_%H%M%S") << ".json";
            std::string savePath = "models/" + ss.str();

            m_currentAgent->saveModel(savePath);
            spdlog::info("Game: Saved session model to {}", savePath);
        } else {
            m_currentAgent->saveModel(m_currentAgentConfig.modelPath);
            spdlog::info("Game: Updated trained model: {}", m_currentAgentConfig.modelPath);
        }
    }

    m_dataCollector->saveTrainingData();
    spdlog::info("Game: Shutdown complete, all data saved");
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

    m_menu->setSettingsCallback([this]() {
        m_currentState = GameState::SETTINGS;
    });

    m_menu->setHowToPlayCallback([this]() {
        m_currentState = GameState::HOW_TO_PLAY;
    });

    // Add leaderboard callback
    m_menu->setLeaderboardCallback([this]() {
        m_currentState = GameState::LEADERBOARD;
    });

    // Add quit callback
    m_menu->setQuitCallback([this]() {
        m_window->close();
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
    while (const std::optional<sf::Event> eventOpt = m_window->pollEvent()) {
        if (!eventOpt.has_value()) {
            continue;
        }
        
        const sf::Event& event = eventOpt.value();

        if (event.is<sf::Event::Closed>()) {
            m_window->close();
        }

        handleGlobalKeys(event);

        switch (m_currentState) {
            case GameState::MENU:
                m_menu->handleEvent(event);
                break;
            case GameState::AGENT_SELECTION:
                m_agentSelection->handleEvent(event);
                break;
            case GameState::LEADERBOARD:
                m_leaderboard->handleEvent(event);
                break;
            case GameState::SETTINGS:
            case GameState::HOW_TO_PLAY:
                if (const auto* keyPressedEvent = event.getIf<sf::Event::KeyPressed>()) {
                    if (keyPressedEvent->scancode == sf::Keyboard::Scancode::Escape) {
                        m_currentState = GameState::MENU;
                    }
                }
                break;
            case GameState::PLAYING:
                handleGameplayEvents(event);
                break;
            case GameState::PAUSED:
                m_pauseMenu->handleEvent(event);
                break;
            case GameState::GAME_OVER:
                handleGameOverEvents(event);
                break;
            default:
                break;
        }
    }
}

void Game::handleGlobalKeys(const sf::Event& event) {
    if (const auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        switch (keyPressed->scancode) {
            case sf::Keyboard::Scancode::F1:
                m_currentState = GameState::LEADERBOARD;
                break;
            case sf::Keyboard::Scancode::F2:
                if (m_gameMode != GameMode::SINGLE_PLAYER) {
                    m_currentState = GameState::AGENT_SELECTION;
                }
                break;
            default:
                break;
        }
    }
}

void Game::handleGameplayEvents(const sf::Event& event) {
    if (const auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if (keyPressed->scancode == sf::Keyboard::Scancode::Escape) {
            m_pauseMenu->setCurrentSpeed(m_moveSpeed);
            m_pauseMenu->setCurrentAgent(m_currentAgentConfig);
            m_pauseMenu->setGameStats(m_score, m_episode, m_currentAgent->getEpsilon());
            m_currentState = GameState::PAUSED;
            return;
        }

        if (keyPressed->scancode == sf::Keyboard::Scancode::Equal) {
            m_moveSpeed = std::min(m_maxSpeed, m_moveSpeed + 0.5f);
            spdlog::info("Speed increased to {} blocks/sec", m_moveSpeed);
        } else if (keyPressed->scancode == sf::Keyboard::Scancode::Hyphen) {
            m_moveSpeed = std::max(m_minSpeed, m_moveSpeed - 0.5f);
            spdlog::info("Speed decreased to {} blocks/sec", m_moveSpeed);
        }
    }

    if (m_gameMode == GameMode::SINGLE_PLAYER) {
        m_inputManager->handleSnakeInput(event, *m_snake);
    } else if (m_gameMode == GameMode::AGENT_VS_PLAYER) {
        handleApplePlacement(event);
    }
}

void Game::handleApplePlacement(const sf::Event& event) {
    if (const auto* mousePressed = event.getIf<sf::Event::MouseButtonPressed>()) {
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
    if (const auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if (keyPressed->scancode == sf::Keyboard::Scancode::Escape) {
            if (shouldAddToLeaderboard()) {
                m_leaderboard->promptForName(m_score, m_currentAgentType, m_episode);
                m_currentState = GameState::LEADERBOARD;
            } else {
                m_currentState = GameState::MENU;
            }
        } else if (keyPressed->scancode == sf::Keyboard::Scancode::R) {
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
        case GameState::SETTINGS:
        case GameState::HOW_TO_PLAY:
        case GameState::GAME_OVER:
            break;
    }
}

void Game::updateGame(float deltaTime) {
    m_moveTimer += deltaTime;

    if (m_moveTimer >= 1.0f / m_moveSpeed) {
        m_moveTimer = 0.0f;

        EnhancedState currentState = StateGenerator::generateState(*m_snake, *m_apple, *m_grid);

        Direction action = Direction::UP;
        if (m_gameMode != GameMode::SINGLE_PLAYER) {
            action = m_currentAgent->getAction(currentState, true);
            m_snake->setDirection(action);
        } else {
            action = m_snake->getDirection();
        }

        m_snake->move();

        if (m_snake->checkSelfCollision() || m_snake->checkWallCollision()) {
            handleSnakeDeath(currentState, action);
            return;
        }

        bool ateFood = false;
        if (m_snake->getHeadPosition() == m_apple->getPosition() && m_apple->isActive()) {
            m_snake->grow();
            m_score++;
            ateFood = true;
            m_apple->setActive(false);
            spawnNewApple();
            spdlog::info("Apple eaten! Score: {}", m_score);
        }

        float reward = calculateReward(ateFood, false);
        EnhancedState nextState = StateGenerator::generateState(*m_snake, *m_apple, *m_grid);

        UnifiedTransition transition;
        transition.basicState = currentState.basic;
        transition.enhancedState = currentState;
        transition.actionIndex = static_cast<int>(action);
        transition.actionDirection = action;
        transition.reward = reward;
        transition.terminal = false;
        transition.agentType = m_currentAgentType;
        transition.epsilon = (m_gameMode == GameMode::SINGLE_PLAYER) ? 0.0f : m_currentAgent->getEpsilon();
        transition.learningRate = 0.1f;

        m_dataCollector->recordTransition(transition);

        if (m_gameMode != GameMode::SINGLE_PLAYER) {
            m_currentAgent->updateAgent(currentState, action, reward, nextState);
        }
    }
}

void Game::handleSnakeDeath(const EnhancedState& currentState, Direction action) {
    m_gameOver = true;
    m_currentState = GameState::GAME_OVER;

    if (m_gameMode != GameMode::SINGLE_PLAYER) {
        float reward = calculateReward(false, true);
        EnhancedState nextState = StateGenerator::generateState(*m_snake, *m_apple, *m_grid);

        UnifiedTransition transition;
        transition.basicState = currentState.basic;
        transition.enhancedState = currentState;
        transition.actionIndex = static_cast<int>(action);
        transition.actionDirection = action;
        transition.reward = reward;
        transition.terminal = true;
        transition.agentType = m_currentAgentType;
        transition.epsilon = m_currentAgent->getEpsilon();
        transition.learningRate = 0.1f;

        m_dataCollector->recordTransition(transition);
        m_currentAgent->updateAgent(currentState, action, reward, nextState);
        m_currentAgent->decayEpsilon();
    }

    m_dataCollector->endEpisode(m_score, true, m_currentAgent->getEpsilon());
    spdlog::info("Game Over! Episode: {}, Score: {}, Agent: {}",
                 m_episode, m_score, m_currentAgentConfig.name);
}

float Game::calculateReward(bool ateFood, bool died) const {
    if (died) return Reward::DEATH;
    if (ateFood) return Reward::EAT_FOOD;

    auto head = m_snake->getHeadPosition();
    auto food = m_apple->getPosition();

    int currentDistance = std::abs(head.x - food.x) + std::abs(head.y - food.y);

    if (currentDistance < m_previousDistance) {
        return Reward::MOVE_TOWARDS_FOOD;
    } else {
        return Reward::MOVE_AWAY_FROM_FOOD;
    }
}

void Game::spawnNewApple() {
    if (m_gameMode == GameMode::SINGLE_PLAYER ||
        m_gameMode == GameMode::AGENT_VS_SYSTEM) {
        m_apple->respawn(*m_snake);
    } else if (m_gameMode == GameMode::AGENT_VS_PLAYER) {
        if (m_hasNextApple) {
            if (!m_snake->isPositionOnSnake(m_nextApplePos)) {
                m_apple->setPosition(m_nextApplePos);
                m_apple->setActive(true);
                m_hasNextApple = false;
            }
        }
    }
}

void Game::render() {
    m_window->clear(sf::Color(245, 245, 220)); // Light beige background

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
        case GameState::SETTINGS:
            renderSettings();
            break;
        case GameState::HOW_TO_PLAY:
            renderHowToPlay();
            break;
        case GameState::PLAYING:
        case GameState::PAUSED:
        case GameState::GAME_OVER:
            renderGame();
            if (m_currentState == GameState::PAUSED) {
                m_pauseMenu->render(*m_window);
            }
            break;
        default:
            break;
    }

    m_window->display();
}

void Game::renderGame() {
    m_grid->render(*m_window);
    m_snake->render(*m_window);
    m_apple->render(*m_window);

    if (m_gameMode == GameMode::AGENT_VS_PLAYER && m_hasNextApple) {
        sf::CircleShape preview(m_grid->getCellSize() / 2.0f * 0.8f);
        preview.setFillColor(sf::Color(255, 182, 193, 128)); // Light pink with transparency
        auto screenPos = m_grid->gridToScreen(m_nextApplePos);
        preview.setPosition(screenPos);
        m_window->draw(preview);
    }

    renderUI();
}

void Game::renderUI() {
    // Light green UI panel
    sf::RectangleShape uiPanel(sf::Vector2f(280.0f, 300.0f));
    uiPanel.setPosition(sf::Vector2f(10.0f, 10.0f));
    uiPanel.setFillColor(sf::Color(240, 255, 240, 200)); // Light mint green with transparency
    uiPanel.setOutlineThickness(2.0f);
    uiPanel.setOutlineColor(sf::Color(34, 139, 34)); // Forest green outline
    m_window->draw(uiPanel);

    sf::Font font;
    bool fontLoaded = false;
    std::vector<std::string> fontPaths = {
        "assets/fonts/arial.ttf",
        "assets/fonts/ARIAL.TTF",
        "assets/fonts/ArialCE.ttf",
        "assets/fonts/Roboto.ttf"
    };

    for (const auto& path : fontPaths) {
        if (font.openFromFile(path)) {
            fontLoaded = true;
            break;
        }
    }

    if (!fontLoaded) {
        spdlog::error("Game: Failed to load font for UI. Display will be incomplete.");
        return;
    }

    std::vector<std::string> lines = {
        "Score: " + std::to_string(m_score),
        "Episode: " + std::to_string(m_episode),
        "Speed: " + std::to_string(m_moveSpeed).substr(0, std::to_string(m_moveSpeed).find(".") + 2) + " blocks/sec",
        "Agent: " + m_currentAgentConfig.name
    };

    if (m_currentAgentType != AgentType::HUMAN) {
        lines.push_back("Epsilon: " + std::to_string(m_currentAgent->getEpsilon()).substr(0, std::min((size_t)7, std::to_string(m_currentAgent->getEpsilon()).length())));
        lines.push_back(m_currentAgent->getAgentInfo());
        lines.push_back("Model: " + m_currentAgent->getModelInfo());
    }

    lines.push_back("");
    lines.push_back("Mode: " + getGameModeString());
    lines.push_back("Controls:");
    lines.push_back("ESC: Pause | +/-: Speed");

    if (m_gameMode == GameMode::AGENT_VS_PLAYER) {
        lines.push_back("Click: Place apple");
    }

    lines.push_back("F1: Leaderboard | F2: Change Agent");

    for (size_t i = 0; i < lines.size(); ++i) {
        sf::Text text(font);
        text.setString(lines[i]);
        text.setCharacterSize(16);
        text.setPosition(sf::Vector2f(20.0f, 20.0f + i * 20.0f));
        text.setFillColor(i < 7 ? sf::Color(47, 79, 47) : sf::Color(25, 25, 112)); // Dark green for stats, midnight blue for controls
        m_window->draw(text);
    }

    if (m_currentState == GameState::GAME_OVER) {
        sf::RectangleShape overlay(sf::Vector2f(600.0f, 250.0f));
        overlay.setPosition(sf::Vector2f(m_window->getSize().x / 2.0f - 300.0f,
                                         m_window->getSize().y / 2.0f - 125.0f));
        overlay.setFillColor(sf::Color(255, 255, 224, 230)); // Light yellow with transparency
        overlay.setOutlineThickness(4.0f);
        overlay.setOutlineColor(sf::Color(220, 20, 60)); // Crimson
        m_window->draw(overlay);

        std::vector<std::string> gameOverLines = {
            "GAME OVER",
            "",
            "Final Score: " + std::to_string(m_score),
            "Agent: " + m_currentAgentConfig.name,
            "Episode: " + std::to_string(m_episode),
            "",
            "ESC: Menu | R: Restart"
        };

        for (size_t i = 0; i < gameOverLines.size(); ++i) {
            sf::Text text(font);
            text.setString(gameOverLines[i]);
            text.setCharacterSize(i == 0 ? 36 : 20);
            text.setFillColor(i == 0 ? sf::Color(220, 20, 60) : sf::Color(47, 79, 47)); // Crimson for title, dark green for text

            auto bounds = text.getLocalBounds();
            text.setOrigin(bounds.size / 2.0f);
            text.setPosition(sf::Vector2f(m_window->getSize().x / 2.0f,
                                          m_window->getSize().y / 2.0f - 100.0f + i * 30.0f));
            m_window->draw(text);
        }
    }
}

void Game::handleMenuSelection(GameMode mode) {
    m_gameMode = mode;

    if (mode == GameMode::SINGLE_PLAYER) {
        m_currentAgentType = AgentType::HUMAN;
        AgentConfig humanConfig;
        humanConfig.type = AgentType::Q_LEARNING;
        humanConfig.name = "Human Player";
        humanConfig.isImplemented = true;
        humanConfig.modelPath = "";
        m_currentAgent = AgentFactory::createAgent(humanConfig);
        m_currentAgentConfig = humanConfig;
        startGame();
    } else {
        m_currentState = GameState::AGENT_SELECTION;
    }
}

void Game::selectAgent(const AgentConfig& config) {
    if (!config.isImplemented) {
        spdlog::warn("Game: Agent {} is not implemented yet", config.name);
        return;
    }

    spdlog::info("Game: Attempting to select agent: {} (model: {})", config.name, config.modelPath);

    if (!config.modelPath.empty() && std::filesystem::exists(config.modelPath)) {
        auto trainedAgent = AgentFactory::createTrainedAgent(config.name);
        if (trainedAgent) {
            m_currentAgent = std::move(trainedAgent);
            spdlog::info("Game: Successfully loaded trained agent: {}", config.name);
            spdlog::info("Game: Model info: {}", m_currentAgent->getModelInfo());
        } else {
            spdlog::error("Game: Failed to load trained agent '{}', falling back to fresh agent", config.name);
            m_currentAgent = AgentFactory::createAgent(config);
        }
    } else {
        m_currentAgent = AgentFactory::createAgent(config);
        spdlog::info("Game: Created fresh agent: {}", config.name);
    }

    m_currentAgentConfig = config;
    m_currentAgentType = config.type;

    m_dataCollector->setAgentType(config.type);

    startGame();
}

void Game::startGame() {
    m_currentState = GameState::PLAYING;
    m_episode++;
    resetGame();

    if (m_gameMode != GameMode::SINGLE_PLAYER) {
        m_dataCollector->startEpisode(m_episode);

        if (auto* qAgent = dynamic_cast<QLearningAgentEnhanced*>(m_currentAgent.get())) {
            qAgent->startEpisode();
        }
    }

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
    return m_score > 5 || m_currentAgentType == AgentType::HUMAN;
}

std::string Game::getGameModeString() const {
    switch (m_gameMode) {
        case GameMode::SINGLE_PLAYER: return "Human vs System";
        case GameMode::AGENT_VS_PLAYER: return "AI vs Player";
        case GameMode::AGENT_VS_SYSTEM: return "AI vs System";
        default: return "Unknown";
    }
}

void Game::renderSettings() {
    sf::RectangleShape background(sf::Vector2f(m_window->getSize().x, m_window->getSize().y));
    background.setFillColor(sf::Color(245, 245, 220)); // Light beige
    m_window->draw(background);

    sf::Font font;
    if (!font.openFromFile("assets/fonts/arial.ttf")) {
        spdlog::error("Game: Failed to load font for Settings screen.");
        return;
    }

    sf::Text title(font);
    title.setString("Settings");
    title.setCharacterSize(48);
    title.setFillColor(sf::Color(47, 79, 47)); // Dark green
    title.setPosition(sf::Vector2f(m_window->getSize().x / 2.0f - 100.0f, 100.0f));
    m_window->draw(title);

    std::vector<std::string> settings = {
        "Game Settings:",
        "",
        "Default Speed: " + std::to_string(m_moveSpeed).substr(0, std::to_string(m_moveSpeed).find(".") + 2) + " blocks/sec",
        "Grid Size: 20x20",
        "Q-Learning Parameters:",
        "  - Learning Rate: 0.1",
        "  - Discount Factor: 0.95",
        "  - Exploration Rate: " + std::to_string(m_currentAgent->getEpsilon()).substr(0, std::min((size_t)7, std::to_string(m_currentAgent->getEpsilon()).length())),
        "",
        "Data Collection:",
        "Training data saved to: data/",
        "Models saved to: models/",
        "Logs saved to: logs/",
        "",
        "ESC: Back to Menu"
    };

    for (size_t i = 0; i < settings.size(); ++i) {
        sf::Text text(font);
        text.setString(settings[i]);
        text.setCharacterSize(i == 0 || i == 9 ? 28 : 20);
        text.setFillColor(i == 0 || i == 9 ? sf::Color(70, 130, 180) : sf::Color(47, 79, 47)); // Steel blue for headers, dark green for content
        text.setPosition(sf::Vector2f(200.0f, 200.0f + i * 30.0f));
        m_window->draw(text);
    }
}

void Game::renderHowToPlay() {
    sf::RectangleShape background(sf::Vector2f(m_window->getSize().x, m_window->getSize().y));
    background.setFillColor(sf::Color(245, 245, 220)); // Light beige
    m_window->draw(background);

    sf::Font font;
    if (!font.openFromFile("assets/fonts/arial.ttf")) {
        spdlog::error("Game: Failed to load font for How To Play screen.");
        return;
    }

    sf::Text title(font);
    title.setString("How to Play");
    title.setCharacterSize(48);
    title.setFillColor(sf::Color(47, 79, 47)); // Dark green
    title.setPosition(sf::Vector2f(m_window->getSize().x / 2.0f - 150.0f, 80.0f));
    m_window->draw(title);

    std::vector<std::string> instructions = {
        "Game Modes:",
        "",
        "Single Player:",
        "  Use arrow keys or WASD to control snake",
        "  Eat red apples to grow and score points",
        "  Avoid hitting walls or your own body",
        "",
        "Agent vs Player:",
        "  AI controls the snake automatically",
        "  Click with mouse to place apples",
        "  Test AI behavior with strategic apple placement",
        "",
        "Agent vs System:",
        "  Pure AI training mode",
        "  AI controls snake, system spawns apples",
        "  Watch the AI learn and improve over time",
        "",
        "Controls:",
        "ESC: Pause game / Open menus",
        "+/-: Adjust game speed",
        "F1: View leaderboard",
        "F2: Change AI agent (in AI modes)",
        "",
        "ESC: Back to Menu"
    };

    for (size_t i = 0; i < instructions.size(); ++i) {
        sf::Text text(font);
        text.setString(instructions[i]);
        text.setCharacterSize(instructions[i].find("Game Modes:") != std::string::npos ||
                               instructions[i].find("Controls:") != std::string::npos ? 28 : 18);
        
        sf::Color textColor = sf::Color(47, 79, 47); // Default dark green
        if (instructions[i].find("Single Player:") != std::string::npos ||
            instructions[i].find("Agent vs Player:") != std::string::npos ||
            instructions[i].find("Agent vs System:") != std::string::npos) {
            textColor = sf::Color(218, 165, 32); // Goldenrod for mode headers
        } else if (instructions[i].find("Game Modes:") != std::string::npos ||
                  instructions[i].find("Controls:") != std::string::npos) {
            textColor = sf::Color(70, 130, 180); // Steel blue for section headers
        }
        
        text.setFillColor(textColor);
        text.setPosition(sf::Vector2f(150.0f, 160.0f + i * 24.0f));
        m_window->draw(text);
    }
}