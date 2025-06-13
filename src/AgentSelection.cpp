#include "AgentSelection.hpp"
#include <spdlog/spdlog.h>

AgentSelection::AgentSelection() : m_selectedIndex(0) {
    m_modelManager = std::make_unique<TrainedModelManager>();
}

void AgentSelection::initialize(sf::RenderWindow& window) {
    // Try multiple font paths for SFML 3
    bool fontLoaded = false;
    std::vector<std::string> fontPaths = {
        "assets/fonts/ARIAL.TTF",
        "assets/fonts/arial.ttf", 
        "assets/fonts/ArialCE.ttf",
        "assets/fonts/Roboto.ttf"
    };
    
    for (const auto& path : fontPaths) {
        if (m_font.openFromFile(path)) {
            fontLoaded = true;
            spdlog::info("AgentSelection: Font loaded from: {}", path);
            break;
        }
    }
    
    if (!fontLoaded) {
        spdlog::error("AgentSelection: Failed to load any font");
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Background
    m_background.setSize(sf::Vector2f(static_cast<float>(windowSize.x), 
                                     static_cast<float>(windowSize.y)));
    m_background.setFillColor(sf::Color(25, 25, 25));
    
    // Title
    m_title = std::make_unique<sf::Text>(m_font);
    m_title->setString("Select AI Agent");
    m_title->setCharacterSize(48);
    m_title->setFillColor(sf::Color::White);
    m_title->setPosition(sf::Vector2f(windowSize.x / 2.0f - 200.0f, 30.0f));
    
    // Section title
    m_sectionTitle = std::make_unique<sf::Text>(m_font);
    m_sectionTitle->setString("🤖 Available AI Agents");
    m_sectionTitle->setCharacterSize(28);
    m_sectionTitle->setFillColor(sf::Color::Cyan);
    m_sectionTitle->setPosition(sf::Vector2f(windowSize.x / 2.0f - 150.0f, 100.0f));
    
    // Instructions
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setString("UP/DOWN: Navigate | ENTER: Select | ESC: Back");
    m_instructions->setCharacterSize(20);
    m_instructions->setFillColor(sf::Color(150, 150, 150));
    m_instructions->setPosition(sf::Vector2f(windowSize.x / 2.0f - 250.0f, 
                                           windowSize.y - 60.0f));
    
    initializeAgents();
    loadTrainedModels();
    
    // Create displays for each agent
    float startY = 160.0f;
    for (size_t i = 0; i < m_agents.size(); ++i) {
        createAgentDisplay(m_agents[i], startY + i * 120.0f);
    }
    
    updateSelection();
    
    spdlog::info("AgentSelection: Initialized with {} agents ({} trained models)", 
                 m_agents.size(), m_modelManager->getAvailableModels().size());
}

void AgentSelection::initializeAgents() {
    m_agents.clear();
    
    // Basic Q-Learning Agent (from scratch)
    AgentConfig basicQLearning;
    basicQLearning.type = AgentType::Q_LEARNING;
    basicQLearning.name = "Q-Learning (Fresh)";
    basicQLearning.description = "Tabular RL agent training from scratch";
    basicQLearning.isImplemented = true;
    basicQLearning.modelPath = "";
    m_agents.emplace_back(basicQLearning);
}

void AgentSelection::loadTrainedModels() {
    auto trainedModels = m_modelManager->getAvailableModels();
    
    spdlog::info("AgentSelection: Loading {} trained models", trainedModels.size());
    
    // Add trained Q-Learning models right after the fresh Q-Learning
    for (const auto& modelInfo : trainedModels) {
        AgentConfig config;
        config.type = AgentType::Q_LEARNING;
        config.name = modelInfo.name;
        config.description = modelInfo.description + " (Pre-trained)";
        config.isImplemented = true;
        config.modelPath = modelInfo.modelPath;
        
        AgentMenuItem item(config);
        item.isTrainedModel = true;
        m_agents.push_back(std::move(item));
        
        spdlog::info("AgentSelection: Added trained model: {} ({})", 
                     modelInfo.name, modelInfo.profile);
    }
    
    // Add other agent types after Q-Learning variants
    // Deep Q-Network (Placeholder)
    AgentConfig dqn;
    dqn.type = AgentType::DEEP_Q_NETWORK;
    dqn.name = "Deep Q-Network (DQN)";
    dqn.description = "Neural network-based Q-learning with experience replay";
    dqn.isImplemented = false;
    dqn.modelPath = "dqn_model.bin";
    m_agents.emplace_back(dqn);
    
    // Policy Gradient (Placeholder)
    AgentConfig pg;
    pg.type = AgentType::POLICY_GRADIENT;
    pg.name = "Policy Gradient";
    pg.description = "Direct policy optimization using REINFORCE algorithm";
    pg.isImplemented = false;
    pg.modelPath = "policy_model.bin";
    m_agents.emplace_back(pg);
    
    // Actor-Critic (Placeholder)
    AgentConfig ac;
    ac.type = AgentType::ACTOR_CRITIC;
    ac.name = "Actor-Critic";
    ac.description = "Combines value function estimation with policy gradient";
    ac.isImplemented = false;
    ac.modelPath = "actor_critic.bin";
    m_agents.emplace_back(ac);
    
    // Genetic Algorithm (Placeholder)
    AgentConfig ga;
    ga.type = AgentType::GENETIC_ALGORITHM;
    ga.name = "Genetic Algorithm";
    ga.description = "Evolution-based approach with neural network population";
    ga.isImplemented = false;
    ga.modelPath = "genetic_best.bin";
    m_agents.emplace_back(ga);
}

void AgentSelection::createAgentDisplay(AgentMenuItem& item, float y) {
    sf::Vector2u windowSize = sf::Vector2u(1200, 800);
    
    // Background panel (larger for trained models)
    float panelHeight = item.isTrainedModel ? 100.0f : 80.0f;
    item.background.setSize(sf::Vector2f(900.0f, panelHeight));
    item.background.setPosition(sf::Vector2f(windowSize.x / 2.0f - 450.0f, y));
    item.background.setFillColor(sf::Color(40, 40, 40));
    item.background.setOutlineThickness(2.0f);
    item.background.setOutlineColor(sf::Color(80, 80, 80));
    
    // Agent name
    item.nameText = std::make_unique<sf::Text>(m_font);
    item.nameText->setString(item.config.name);
    item.nameText->setCharacterSize(24);
    item.nameText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 430.0f, y + 10.0f));
    
    // Description
    item.descText = std::make_unique<sf::Text>(m_font);
    item.descText->setString(item.config.description);
    item.descText->setCharacterSize(16);
    item.descText->setFillColor(sf::Color(180, 180, 180));
    item.descText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 430.0f, y + 40.0f));
    
    // Status
    item.statusText = std::make_unique<sf::Text>(m_font);
    std::string statusStr = item.config.isImplemented ? "READY" : "COMING SOON";
    if (item.isTrainedModel) {
        statusStr = "PRE-TRAINED";
    }
    item.statusText->setString(statusStr);
    item.statusText->setCharacterSize(18);
    
    sf::Color statusColor = sf::Color::Green;
    if (!item.config.isImplemented) statusColor = sf::Color::Yellow;
    else if (item.isTrainedModel) statusColor = sf::Color::Cyan;
    
    item.statusText->setFillColor(statusColor);
    item.statusText->setPosition(sf::Vector2f(windowSize.x / 2.0f + 300.0f, y + 15.0f));
    
    // Model info for trained models
    if (item.isTrainedModel) {
        item.modelInfoText = std::make_unique<sf::Text>(m_font);
        
        // Find model info from manager
        auto* modelInfo = m_modelManager->findModel(item.config.name);
        std::string infoStr = "Model Performance: ";
        if (modelInfo && modelInfo->averageScore > 0) {
            infoStr += "Avg Score: " + std::to_string(modelInfo->averageScore).substr(0, 5);
            infoStr += " | Episodes: " + std::to_string(modelInfo->episodesTrained);
        } else {
            infoStr += "Performance data not available";
        }
        
        item.modelInfoText->setString(infoStr);
        item.modelInfoText->setCharacterSize(14);
        item.modelInfoText->setFillColor(sf::Color(200, 255, 200));
        item.modelInfoText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 430.0f, y + 65.0f));
    }
}

void AgentSelection::handleEvent(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        switch (keyPressed->code) {
            case sf::Keyboard::Key::Up:
                m_selectedIndex = (m_selectedIndex - 1 + m_agents.size()) % m_agents.size();
                updateSelection();
                spdlog::debug("AgentSelection: Selected agent index: {}", m_selectedIndex);
                break;
            case sf::Keyboard::Key::Down:
                m_selectedIndex = (m_selectedIndex + 1) % m_agents.size();
                updateSelection();
                spdlog::debug("AgentSelection: Selected agent index: {}", m_selectedIndex);
                break;
            case sf::Keyboard::Key::Enter:
                if (m_agents[m_selectedIndex].config.isImplemented && m_selectionCallback) {
                    spdlog::info("AgentSelection: Selected agent: {}", 
                                m_agents[m_selectedIndex].config.name);
                    m_selectionCallback(m_agents[m_selectedIndex].config);
                } else {
                    spdlog::warn("AgentSelection: Attempted to select unimplemented agent: {}", 
                                m_agents[m_selectedIndex].config.name);
                }
                break;
            case sf::Keyboard::Key::Escape:
                if (m_backCallback) {
                    spdlog::info("AgentSelection: Going back to main menu");
                    m_backCallback();
                }
                break;
        }
    }
}

void AgentSelection::update() {
    // Add animations or other updates here
}

void AgentSelection::render(sf::RenderWindow& window) {
    window.draw(m_background);
    if (m_title) window.draw(*m_title);
    if (m_sectionTitle) window.draw(*m_sectionTitle);
    if (m_instructions) window.draw(*m_instructions);
    
    for (const auto& agent : m_agents) {
        window.draw(agent.background);
        if (agent.nameText) window.draw(*agent.nameText);
        if (agent.descText) window.draw(*agent.descText);
        if (agent.statusText) window.draw(*agent.statusText);
        if (agent.modelInfoText) window.draw(*agent.modelInfoText);
    }
}

void AgentSelection::updateSelection() {
    for (size_t i = 0; i < m_agents.size(); ++i) {
        if (i == m_selectedIndex) {
            // Highlight selected agent
            sf::Color highlightColor = m_agents[i].isTrainedModel ? 
                sf::Color(60, 120, 100) : sf::Color(60, 100, 60);
            
            m_agents[i].background.setFillColor(highlightColor);
            m_agents[i].background.setOutlineColor(sf::Color::Green);
            
            if (m_agents[i].nameText) {
                m_agents[i].nameText->setFillColor(sf::Color::White);
                m_agents[i].nameText->setStyle(sf::Text::Bold);
            }
        } else {
            // Normal appearance
            m_agents[i].background.setFillColor(sf::Color(40, 40, 40));
            m_agents[i].background.setOutlineColor(sf::Color(80, 80, 80));
            
            if (m_agents[i].nameText) {
                sf::Color textColor = m_agents[i].config.isImplemented ? 
                    sf::Color::White : sf::Color(120, 120, 120);
                m_agents[i].nameText->setFillColor(textColor);
                m_agents[i].nameText->setStyle(sf::Text::Regular);
            }
        }
    }
}

void AgentSelection::setSelectionCallback(std::function<void(const AgentConfig&)> callback) {
    m_selectionCallback = callback;
}

void AgentSelection::setBackCallback(std::function<void()> callback) {
    m_backCallback = callback;
}