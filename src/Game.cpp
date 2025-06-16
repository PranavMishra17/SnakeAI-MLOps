#include "Game.hpp"
#include "MLAgents.hpp" // Assuming this includes AgentFactory, IAgent, QLearningAgentEnhanced
#include "PauseMenu.hpp"
#include "AgentSelection.hpp"
#include "Leaderboard.hpp"
#include "Grid.hpp"        // Assuming Grid class is defined here
#include "Snake.hpp"       // Assuming Snake class is defined here
#include "Apple.hpp"       // Assuming Apple class is defined here
#include "UnifiedDataCollector.hpp" // Assuming UnifiedDataCollector is defined here
#include "InputManager.hpp" // Assuming InputManager is defined here
#include "StateGenerator.hpp" // Assuming StateGenerator and EnhancedState are defined here
#include "Reward.hpp"      // Assuming Reward constants are defined here
#include <filesystem>
#include <spdlog/spdlog.h> // For logging
#include <chrono>           // For time-based model saving
#include <iomanip>          // For std::put_time (potentially, if used for formatting)
#include <algorithm>        // For std::min, std::max
#include <optional>         // Required for std::optional from pollEvent


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
    , m_minSpeed(1.0f) // Assuming a default minimum speed
    , m_maxSpeed(10.0f) // Assuming a default maximum speed
    , m_previousDistance(999) // Initialize with a large value
{
    spdlog::info("Game: Starting initialization...");

    try {
        // Create necessary directories
        std::filesystem::create_directories("models");
        std::filesystem::create_directories("data");
        std::filesystem::create_directories("logs");
        std::filesystem::create_directories("assets/fonts");

        // Create window
        // sf::VideoMode::getDesktopMode() returns a sf::VideoMode object for SFML3
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

        // Initialize utility components
        m_dataCollector = std::make_unique<UnifiedDataCollector>();
        m_inputManager = std::make_unique<InputManager>();

        // Initialize default agent (fresh Q-Learning)
        AgentConfig defaultConfig;
        defaultConfig.type = AgentType::Q_LEARNING;
        defaultConfig.name = "Q-Learning Agent";
        defaultConfig.isImplemented = true;
        m_currentAgent = AgentFactory::createAgent(defaultConfig);
        m_currentAgentConfig = defaultConfig;

        // Set agent type for data collector
        m_dataCollector->setAgentType(AgentType::Q_LEARNING);

        setupCallbacks();

        spdlog::info("Game: Initialization complete with trained model support!");

    } catch (const std::exception& e) {
        spdlog::error("Game: Exception during initialization: {}", e.what());
        throw;
    }
}

// Destructor
Game::~Game() {
    if (m_currentAgent) {
        std::filesystem::create_directories("models");

        // For fresh agents, save with timestamp
        if (m_currentAgentConfig.modelPath.empty()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            // Use std::stringstream for thread-safe time formatting if needed,
            // or ensure std::localtime is used carefully. For simplicity here, assume it's okay.
            std::tm tm_buffer;
            #ifdef _MSC_VER
                localtime_s(&tm_buffer, &time_t); // Windows specific
            #else
                localtime_r(&time_t, &tm_buffer); // POSIX specific
            #endif

            std::stringstream ss;
            ss << "qtable_session_" << std::put_time(&tm_buffer, "%Y%m%d_%H%M%S") << ".json";
            std::string savePath = "models/" + ss.str();

            m_currentAgent->saveModel(savePath);
            spdlog::info("Game: Saved session model to {}", savePath);
        } else {
            // For trained models, save to their original path
            m_currentAgent->saveModel(m_currentAgentConfig.modelPath);
            spdlog::info("Game: Updated trained model: {}", m_currentAgentConfig.modelPath);
        }
    }

    m_dataCollector->saveTrainingData();
    spdlog::info("Game: Shutdown complete, all data saved");
}

// Main game loop execution
void Game::run() {
    while (m_window->isOpen()) {
        processEvents(); // Process all pending events

        float deltaTime = m_clock.restart().asSeconds(); // Get time since last frame
        update(deltaTime); // Update game logic

        render(); // Render game state
    }
}

// Setup all UI and game state callbacks
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
        m_currentState = GameState::MENU; // After selection, go back to menu to start game
    });

    m_agentSelection->setBackCallback([this]() {
        m_currentState = GameState::MENU;
    });

    // Leaderboard callbacks
    m_leaderboard->setBackCallback([this]() {
        m_currentState = GameState::MENU;
    });
}

// Processes all SFML events for the current frame
void Game::processEvents() {
    // SFML 3: pollEvent returns std::optional<sf::Event>
    // Loop through all pending events from the window
    while (const std::optional<sf::Event> eventOpt = m_window->pollEvent()) {
        // If the optional contains an event, get a reference to it
        if (!eventOpt.has_value()) {
            continue; // Should not happen in a while loop, but good for robustness
        }
        
        const sf::Event& event = eventOpt.value();

        // Handle global events (e.g., window close)
        if (event.is<sf::Event::Closed>()) { // SFML 3: Use is<T>() for simple type check
            m_window->close();
        }

        // Handle global key presses (like F1/F2) regardless of current state
        handleGlobalKeys(event);

        // Delegate event handling based on current game state
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
                // Specific handling for Escape in Settings/HowToPlay states
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
                // Handle unexpected states or do nothing
                break;
        }
    }
}

// Handles key presses that are global to the entire application
void Game::handleGlobalKeys(const sf::Event& event) {
    // SFML 3: Use getIf<T>() to check if the event is a KeyPressed event
    if (const auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        // Use sf::Keyboard::Scancode for layout-independent key recognition
        switch (keyPressed->scancode) {
            case sf::Keyboard::Scancode::F1:
                m_currentState = GameState::LEADERBOARD;
                break;
            case sf::Keyboard::Scancode::F2:
                // Allow changing agent only if not in single player mode
                if (m_gameMode != GameMode::SINGLE_PLAYER) {
                    m_currentState = GameState::AGENT_SELECTION;
                }
                break;
            default:
                break;
        }
    }
}

// Handles events specific to the PLAYING state
void Game::handleGameplayEvents(const sf::Event& event) {
    if (const auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        // Pause game on Escape key press
        if (keyPressed->scancode == sf::Keyboard::Scancode::Escape) {
            m_pauseMenu->setCurrentSpeed(m_moveSpeed);
            m_pauseMenu->setCurrentAgent(m_currentAgentConfig);
            // Assuming getEpsilon is a valid method on IAgent, otherwise dynamic_cast here
            m_pauseMenu->setGameStats(m_score, m_episode, m_currentAgent->getEpsilon());
            m_currentState = GameState::PAUSED;
            return;
        }

        // Speed adjustment using Equal (+) and Hyphen (-) keys
        if (keyPressed->scancode == sf::Keyboard::Scancode::Equal) {
            m_moveSpeed = std::min(m_maxSpeed, m_moveSpeed + 0.5f);
            spdlog::info("Speed increased to {} blocks/sec", m_moveSpeed);
        } else if (keyPressed->scancode == sf::Keyboard::Scancode::Hyphen) {
            m_moveSpeed = std::max(m_minSpeed, m_moveSpeed - 0.5f);
            spdlog::info("Speed decreased to {} blocks/sec", m_moveSpeed);
        }
    }

    // Handle input based on the current game mode
    if (m_gameMode == GameMode::SINGLE_PLAYER) {
        // Human player controls the snake directly
        m_inputManager->handleSnakeInput(event, *m_snake);
    } else if (m_gameMode == GameMode::AGENT_VS_PLAYER) {
        // Player places apples in this mode
        handleApplePlacement(event);
    }
    // Agent vs System mode does not require direct user input for gameplay
}

// Handles apple placement by player clicks in AGENT_VS_PLAYER mode
void Game::handleApplePlacement(const sf::Event& event) {
    if (const auto* mousePressed = event.getIf<sf::Event::MouseButtonPressed>()) {
        if (mousePressed->button == sf::Mouse::Button::Left) {
            auto mousePos = sf::Mouse::getPosition(*m_window);
            auto gridPos = m_grid->screenToGrid(mousePos);

            // Ensure the clicked position is valid and not on the snake's body
            if (m_grid->isValidPosition(gridPos) && !m_snake->isPositionOnSnake(gridPos)) {
                if (!m_apple->isActive()) {
                    // If no apple is currently active, place it directly
                    m_apple->setPosition(gridPos);
                    m_apple->setActive(true);
                } else {
                    // If an apple is active, queue the next apple placement
                    m_nextApplePos = gridPos;
                    m_hasNextApple = true;
                }
            }
        }
    }
}

// Handles events when the game is in the GAME_OVER state
void Game::handleGameOverEvents(const sf::Event& event) {
    if (const auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if (keyPressed->scancode == sf::Keyboard::Scancode::Escape) {
            // If the score is high enough, prompt for leaderboard entry
            if (shouldAddToLeaderboard()) {
                m_leaderboard->promptForName(m_score, m_currentAgentType, m_episode);
                m_currentState = GameState::LEADERBOARD;
            } else {
                // Otherwise, return to the main menu
                m_currentState = GameState::MENU;
            }
        } else if (keyPressed->scancode == sf::Keyboard::Scancode::R) {
            // Restart the game directly from game over screen
            resetGame();
            m_currentState = GameState::PLAYING;
        }
    }
}

// Updates game logic based on the current game state
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
            if (!m_gameOver) { // Only update game when not in game over state
                updateGame(deltaTime);
            }
            break;
        case GameState::PAUSED:
            m_pauseMenu->update();
            break;
        case GameState::SETTINGS:
        case GameState::HOW_TO_PLAY:
        case GameState::GAME_OVER:
            // These states typically don't have ongoing updates, just rendering
            break;
    }
}

// Updates the core game logic (snake movement, collisions, scoring)
void Game::updateGame(float deltaTime) {
    m_moveTimer += deltaTime;

    // Move the snake based on the calculated move speed
    if (m_moveTimer >= 1.0f / m_moveSpeed) {
        m_moveTimer = 0.0f;

        // Generate the current state for the agent
        EnhancedState currentState = StateGenerator::generateState(*m_snake, *m_apple, *m_grid);

        // Determine the snake's next action (AI decision or human input)
        Direction action = Direction::UP; // Default direction
        if (m_gameMode != GameMode::SINGLE_PLAYER) {
            // AI agent decides the action
            action = m_currentAgent->getAction(currentState, true);
            m_snake->setDirection(action); // Update snake's direction based on AI
        } else {
            // Human player controls, direction is already set by InputManager
            action = m_snake->getDirection();
        }

        // Move the snake
        m_snake->move();

        // Check for collisions (self or wall)
        if (m_snake->checkSelfCollision() || m_snake->checkWallCollision()) {
            handleSnakeDeath(currentState, action); // Handle game over scenario
            return; // Stop further updates for this frame after death
        }

        // Check if the snake ate the apple
        bool ateFood = false;
        if (m_snake->getHeadPosition() == m_apple->getPosition() && m_apple->isActive()) {
            m_snake->grow();
            m_score++;
            ateFood = true;
            m_apple->setActive(false); // Deactivate current apple
            spawnNewApple(); // Spawn a new apple
            spdlog::info("Apple eaten! Score: {}", m_score);
        }

        // Record data for all game modes (for potential training or analysis)
        float reward = calculateReward(ateFood, false); // Calculate reward for this step
        EnhancedState nextState = StateGenerator::generateState(*m_snake, *m_apple, *m_grid); // Generate next state

        // Create and record a unified transition
        UnifiedTransition transition;
        transition.basicState = currentState.basic;
        transition.enhancedState = currentState;
        transition.actionIndex = static_cast<int>(action);
        transition.actionDirection = action;
        transition.reward = reward;
        transition.terminal = false; // Not terminal yet
        transition.agentType = m_currentAgentType;
        transition.epsilon = (m_gameMode == GameMode::SINGLE_PLAYER) ? 0.0f : m_currentAgent->getEpsilon();
        transition.learningRate = 0.1f; // Example learning rate, could be configurable

        m_dataCollector->recordTransition(transition);

        // Update the agent's model only for AI modes
        if (m_gameMode != GameMode::SINGLE_PLAYER) {
            m_currentAgent->updateAgent(currentState, action, reward, nextState);
        }
    }
}

// Handles actions when the snake dies
void Game::handleSnakeDeath(const EnhancedState& currentState, Direction action) {
    m_gameOver = true;
    m_currentState = GameState::GAME_OVER;

    // Update agent with death penalty (for AI modes)
    if (m_gameMode != GameMode::SINGLE_PLAYER) {
        float reward = calculateReward(false, true); // Calculate death reward
        EnhancedState nextState = StateGenerator::generateState(*m_snake, *m_apple, *m_grid); // Generate final state

        // Create and record the final, terminal transition
        UnifiedTransition transition;
        transition.basicState = currentState.basic;
        transition.enhancedState = currentState;
        transition.actionIndex = static_cast<int>(action);
        transition.actionDirection = action;
        transition.reward = reward;
        transition.terminal = true; // Mark as terminal
        transition.agentType = m_currentAgentType;
        transition.epsilon = m_currentAgent->getEpsilon();
        transition.learningRate = 0.1f; // Example learning rate

        m_dataCollector->recordTransition(transition);
        m_currentAgent->updateAgent(currentState, action, reward, nextState);
        m_currentAgent->decayEpsilon(); // Decay epsilon for exploration in AI
    }

    m_dataCollector->endEpisode(m_score, true, m_currentAgent->getEpsilon()); // Log episode end
    spdlog::info("Game Over! Episode: {}, Score: {}, Agent: {}",
                 m_episode, m_score, m_currentAgentConfig.name);
}

// Calculates the reward for the agent based on game events
float Game::calculateReward(bool ateFood, bool died) const {
    if (died) return Reward::DEATH;      // Negative reward for death
    if (ateFood) return Reward::EAT_FOOD; // Positive reward for eating food

    // Distance-based reward shaping: reward for moving closer to food
    auto head = m_snake->getHeadPosition();
    auto food = m_apple->getPosition();

    // Manhattan distance
    int currentDistance = std::abs(head.x - food.x) + std::abs(head.y - food.y);

    if (currentDistance < m_previousDistance) {
        // If the snake moved closer to the food
        // m_previousDistance = currentDistance; // Don't update const function
        return Reward::MOVE_TOWARDS_FOOD;
    } else {
        // If the snake moved further or stayed same distance
        // m_previousDistance = currentDistance; // Don't update const function
        return Reward::MOVE_AWAY_FROM_FOOD;
    }
}

// Spawns a new apple based on game mode
void Game::spawnNewApple() {
    // For Single Player (human) and Agent vs System, apples spawn automatically
    if (m_gameMode == GameMode::SINGLE_PLAYER ||
        m_gameMode == GameMode::AGENT_VS_SYSTEM) {
        m_apple->respawn(*m_snake); // Randomly respawn apple
    } else if (m_gameMode == GameMode::AGENT_VS_PLAYER) {
        // For Agent vs Player, only spawn if player has clicked a next apple position
        if (m_hasNextApple) {
            // Ensure the next apple position is not on the snake
            if (!m_snake->isPositionOnSnake(m_nextApplePos)) {
                m_apple->setPosition(m_nextApplePos);
                m_apple->setActive(true);
                m_hasNextApple = false; // Reset flag after placement
            }
            // If m_nextApplePos is on snake, apple remains inactive until a valid click
        }
        // If m_hasNextApple is false, apple remains inactive until player clicks
    }
}

// Renders the current game state to the window
void Game::render() {
    m_window->clear(sf::Color(30, 30, 30)); // Clear window with a dark grey background

    // Render different screens based on current game state
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
            renderGame(); // Render the core game elements
            if (m_currentState == GameState::PAUSED) {
                m_pauseMenu->render(*m_window); // Render pause menu on top if paused
            }
            break;
        default:
            // Handle unexpected states (e.g., render a debug message)
            break;
    }

    m_window->display(); // Display the rendered frame
}

// Renders the main game elements: grid, snake, apple, and UI
void Game::renderGame() {
    m_grid->render(*m_window);
    m_snake->render(*m_window);
    m_apple->render(*m_window);

    // Render a preview of the next apple for Agent vs Player mode
    if (m_gameMode == GameMode::AGENT_VS_PLAYER && m_hasNextApple) {
        sf::CircleShape preview(m_grid->getCellSize() / 2.0f * 0.8f);
        preview.setFillColor(sf::Color(255, 0, 0, 128)); // Semi-transparent red
        auto screenPos = m_grid->gridToScreen(m_nextApplePos);
        preview.setPosition(screenPos);
        m_window->draw(preview);
    }

    renderUI(); // Render the overlay UI (score, info)
}

// Renders the in-game UI elements
void Game::renderUI() {
    // UI background panel
    sf::RectangleShape uiPanel(sf::Vector2f(280.0f, 300.0f)); // Adjusted width for better fit
    uiPanel.setPosition(sf::Vector2f(10.0f, 10.0f));
    uiPanel.setFillColor(sf::Color(0, 0, 0, 180)); // Semi-transparent black
    uiPanel.setOutlineThickness(1.0f);
    uiPanel.setOutlineColor(sf::Color(100, 100, 100)); // Grey outline
    m_window->draw(uiPanel);

    // Font loading - SFML 3: openFromFile returns bool
    sf::Font font;
    bool fontLoaded = false;
    std::vector<std::string> fontPaths = {
        "assets/fonts/arial.ttf", // Prefer standard font paths first
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
        return; // Cannot render UI text without font
    }

    // Prepare game stats and info lines
    std::vector<std::string> lines = {
        "Score: " + std::to_string(m_score),
        "Episode: " + std::to_string(m_episode),
        "Speed: " + std::to_string(m_moveSpeed).substr(0, std::to_string(m_moveSpeed).find(".") + 2) + " blocks/sec", // Format to 1 decimal place
        "Agent: " + m_currentAgentConfig.name
    };

    if (m_currentAgentType != AgentType::HUMAN) {
        lines.push_back("Epsilon: " + std::to_string(m_currentAgent->getEpsilon()).substr(0, std::min((size_t)7, std::to_string(m_currentAgent->getEpsilon()).length()))); // Format epsilon
        lines.push_back(m_currentAgent->getAgentInfo());
        lines.push_back("Model: " + m_currentAgent->getModelInfo());
    }

    // Game mode specific instructions and controls
    lines.push_back(""); // Empty line for spacing
    lines.push_back("Mode: " + getGameModeString());
    lines.push_back("Controls:");
    lines.push_back("ESC: Pause | +/-: Speed");

    if (m_gameMode == GameMode::AGENT_VS_PLAYER) {
        lines.push_back("Click: Place apple");
    }

    lines.push_back("F1: Leaderboard | F2: Change Agent");

    // Draw each line of text
    for (size_t i = 0; i < lines.size(); ++i) {
        sf::Text text(font);
        text.setString(lines[i]);
        text.setCharacterSize(16); // Standard text size
        text.setPosition(sf::Vector2f(20.0f, 20.0f + i * 20.0f)); // Position lines vertically
        text.setFillColor(i < 7 ? sf::Color::White : sf::Color(180, 180, 180)); // White for stats, grey for controls
        m_window->draw(text);
    }

    // Game over overlay
    if (m_currentState == GameState::GAME_OVER) {
        sf::RectangleShape overlay(sf::Vector2f(600.0f, 250.0f));
        overlay.setPosition(sf::Vector2f(m_window->getSize().x / 2.0f - 300.0f,
                                         m_window->getSize().y / 2.0f - 125.0f));
        overlay.setFillColor(sf::Color(0, 0, 0, 200)); // Semi-transparent black
        overlay.setOutlineThickness(3.0f);
        overlay.setOutlineColor(sf::Color::Red);
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
            text.setCharacterSize(i == 0 ? 36 : 20); // Larger for "GAME OVER"
            text.setFillColor(i == 0 ? sf::Color::Red : sf::Color::White);

            // Center text horizontally within the overlay
            auto bounds = text.getLocalBounds();
            text.setOrigin(bounds.left + bounds.width / 2.0f, bounds.top + bounds.height / 2.0f); // Set origin to center
            text.setPosition(sf::Vector2f(m_window->getSize().x / 2.0f,
                                          m_window->getSize().y / 2.0f - 100.0f + i * 30.0f));
            m_window->draw(text);
        }
    }
}

// Handles selection from the main menu, determining game mode
void Game::handleMenuSelection(GameMode mode) {
    m_gameMode = mode;

    if (mode == GameMode::SINGLE_PLAYER) {
        // For human player, set agent type to HUMAN and reset to a fresh Q-Learning agent
        m_currentAgentType = AgentType::HUMAN;
        AgentConfig humanConfig;
        humanConfig.type = AgentType::Q_LEARNING; // Still use Q_LEARNING for base, but it's "human controlled"
        humanConfig.name = "Human Player";
        humanConfig.isImplemented = true;
        humanConfig.modelPath = ""; // No model path for human
        m_currentAgent = AgentFactory::createAgent(humanConfig);
        m_currentAgentConfig = humanConfig; // Store the config
        startGame();
    } else {
        // For AI modes, transition to agent selection screen
        m_currentState = GameState::AGENT_SELECTION;
    }
}

// Selects and initializes an AI agent based on the provided configuration
void Game::selectAgent(const AgentConfig& config) {
    if (!config.isImplemented) {
        spdlog::warn("Game: Agent {} is not implemented yet", config.name);
        return; // Do not select unimplemented agents
    }

    spdlog::info("Game: Attempting to select agent: {} (model: {})", config.name, config.modelPath);

    // Try to load a trained agent if a model path is provided and exists
    if (!config.modelPath.empty() && std::filesystem::exists(config.modelPath)) {
        auto trainedAgent = AgentFactory::createTrainedAgent(config.name); // Assumes AgentFactory can load by name
        if (trainedAgent) {
            m_currentAgent = std::move(trainedAgent);
            spdlog::info("Game: Successfully loaded trained agent: {}", config.name);
            spdlog::info("Game: Model info: {}", m_currentAgent->getModelInfo());
        } else {
            spdlog::error("Game: Failed to load trained agent '{}', falling back to fresh agent", config.name);
            m_currentAgent = AgentFactory::createAgent(config); // Create a fresh instance
        }
    } else {
        // If no model path or path doesn't exist, create a fresh agent
        m_currentAgent = AgentFactory::createAgent(config);
        spdlog::info("Game: Created fresh agent: {}", config.name);
    }

    m_currentAgentConfig = config; // Store the active agent's configuration
    m_currentAgentType = config.type; // Update current agent type

    m_dataCollector->setAgentType(config.type); // Update data collector's agent type

    startGame(); // Start the game with the selected agent
}

// Starts a new game episode
void Game::startGame() {
    m_currentState = GameState::PLAYING;
    m_episode++; // Increment episode count for AI training
    resetGame(); // Reset game elements

    // Start episode tracking for AI agents in data collector
    if (m_gameMode != GameMode::SINGLE_PLAYER) {
        m_dataCollector->startEpisode(m_episode);

        // Special handling for Q-Learning agent (e.g., to reset episode-specific stats)
        if (auto* qAgent = dynamic_cast<QLearningAgentEnhanced*>(m_currentAgent.get())) {
            qAgent->startEpisode(); // Call Q-Learning specific start episode
        }
    }

    spdlog::info("Starting game - Mode: {}, Episode: {}, Agent: {}",
                 static_cast<int>(m_gameMode), m_episode, m_currentAgentConfig.name);
}

// Resets game state to start a new round
void Game::resetGame() {
    m_score = 0;
    m_gameOver = false;
    m_moveTimer = 0.0f;
    m_hasNextApple = false;
    m_previousDistance = 999; // Reset distance for reward calculation

    m_snake->reset(); // Reset snake position and length

    // Spawn apple based on game mode
    if (m_gameMode == GameMode::SINGLE_PLAYER || m_gameMode == GameMode::AGENT_VS_SYSTEM) {
        m_apple->respawn(*m_snake); // Auto spawn apple
    } else {
        m_apple->setActive(false); // In Agent vs Player, wait for player to place apple
    }
}

// Determines if the current score should be added to the leaderboard
bool Game::shouldAddToLeaderboard() const {
    // Example logic: add if score is above a threshold or if it was a human game
    return m_score > 5 || m_currentAgentType == AgentType::HUMAN;
}

// Returns a string representation of the current game mode
std::string Game::getGameModeString() const {
    switch (m_gameMode) {
        case GameMode::SINGLE_PLAYER: return "Human vs System";
        case GameMode::AGENT_VS_PLAYER: return "AI vs Player";
        case GameMode::AGENT_VS_SYSTEM: return "AI vs System";
        default: return "Unknown";
    }
}

// Renders the settings screen
void Game::renderSettings() {
    sf::RectangleShape background(sf::Vector2f(m_window->getSize().x, m_window->getSize().y));
    background.setFillColor(sf::Color(25, 25, 35)); // Dark blue-grey background
    m_window->draw(background);

    sf::Font font;
    if (!font.openFromFile("assets/fonts/arial.ttf")) {
        spdlog::error("Game: Failed to load font for Settings screen.");
        return; // Cannot render without font
    }

    sf::Text title(font);
    title.setString("⚙️ Settings");
    title.setCharacterSize(48);
    title.setFillColor(sf::Color::White);
    title.setPosition(sf::Vector2f(m_window->getSize().x / 2.0f - title.getLocalBounds().width / 2.0f, 100.0f)); // Center title
    m_window->draw(title);

    std::vector<std::string> settings = {
        "Game Settings:",
        "",
        "• Default Speed: " + std::to_string(m_moveSpeed).substr(0, std::to_string(m_moveSpeed).find(".") + 2) + " blocks/sec",
        "• Grid Size: 20x20",
        "• Q-Learning Parameters:",
        "  - Learning Rate: 0.1", // Assuming fixed values for display
        "  - Discount Factor: 0.95",
        "  - Exploration Rate: " + std::to_string(m_currentAgent->getEpsilon()).substr(0, std::min((size_t)7, std::to_string(m_currentAgent->getEpsilon()).length())),
        "",
        "Data Collection:",
        "• Training data saved to: data/",
        "• Models saved to: models/",
        "• Logs saved to: logs/",
        "",
        "ESC: Back to Menu"
    };

    for (size_t i = 0; i < settings.size(); ++i) {
        sf::Text text(font);
        text.setString(settings[i]);
        text.setCharacterSize(i == 0 || i == 9 ? 28 : 20); // Larger for section headers
        text.setFillColor(i == 0 || i == 9 ? sf::Color::Cyan : sf::Color::White); // Cyan for headers
        text.setPosition(sf::Vector2f(200.0f, 200.0f + i * 30.0f));
        m_window->draw(text);
    }
}

// Renders the "How to Play" screen
void Game::renderHowToPlay() {
    sf::RectangleShape background(sf::Vector2f(m_window->getSize().x, m_window->getSize().y));
    background.setFillColor(sf::Color(25, 35, 25)); // Dark green-grey background
    m_window->draw(background);

    sf::Font font;
    if (!font.openFromFile("assets/fonts/arial.ttf")) {
        spdlog::error("Game: Failed to load font for How To Play screen.");
        return; // Cannot render without font
    }

    sf::Text title(font);
    title.setString("❓ How to Play");
    title.setCharacterSize(48);
    title.setFillColor(sf::Color::White);
    title.setPosition(sf::Vector2f(m_window->getSize().x / 2.0f - title.getLocalBounds().width / 2.0f, 80.0f)); // Center title
    m_window->draw(title);

    std::vector<std::string> instructions = {
        "Game Modes:",
        "",
        "🎮 Single Player:",
        "  • Use arrow keys (↑↓←→) or WASD to control snake",
        "  • Eat red apples to grow and score points",
        "  • Avoid hitting walls or your own body",
        "",
        "🤖 Agent vs Player:",
        "  • AI controls the snake automatically",
        "  • Click with mouse to place apples",
        "  • Test AI behavior with strategic apple placement",
        "",
        "🔬 Agent vs System:",
        "  • Pure AI training mode",
        "  • AI controls snake, system spawns apples",
        "  • Watch the AI learn and improve over time",
        "",
        "Controls:",
        "• ESC: Pause game / Open menus",
        "• +/-: Adjust game speed",
        "• F1: View leaderboard",
        "• F2: Change AI agent (in AI modes)",
        "",
        "ESC: Back to Menu"
    };

    for (size_t i = 0; i < instructions.size(); ++i) {
        sf::Text text(font);
        text.setString(instructions[i]);
        text.setCharacterSize(instructions[i].find("Game Modes:") != std::string::npos ||
                               instructions[i].find("Controls:") != std::string::npos ? 28 : 18); // Larger for section headers
        text.setFillColor(instructions[i].find("🎮") != std::string::npos ||
                          instructions[i].find("🤖") != std::string::npos ||
                          instructions[i].find("🔬") != std::string::npos ? sf::Color::Yellow :
                          instructions[i].find("Game Modes:") != std::string::npos ||
                          instructions[i].find("Controls:") != std::string::npos ? sf::Color::Cyan : sf::Color::White); // Color highlighting for sections
        text.setPosition(sf::Vector2f(150.0f, 160.0f + i * 24.0f));
        m_window->draw(text);
    }
}
