#include "AgentSelection.hpp"
#include <spdlog/spdlog.h>

AgentSelection::AgentSelection() : m_selectedIndex(0) {}

void AgentSelection::initialize(sf::RenderWindow& window) {
    if (!m_font.openFromFile("assets/fonts/arial.ttf")) {
        spdlog::error("Failed to load font in AgentSelection");
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
    m_title->setPosition(sf::Vector2f(windowSize.x / 2.0f - 200.0f, 50.0f));
    
    // Instructions
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setString("UP/DOWN: Navigate | ENTER: Select | ESC: Back");
    m_instructions->setCharacterSize(20);
    m_instructions->setFillColor(sf::Color(150, 150, 150));
    m_instructions->setPosition(sf::Vector2f(windowSize.x / 2.0f - 250.0f, 
                                           windowSize.y - 60.0f));
    
    initializeAgents();
    
    // Create displays for each agent
    float startY = 150.0f;
    for (size_t i = 0; i < m_agents.size(); ++i) {
        createAgentDisplay(m_agents[i], startY + i * 100.0f);
    }
    
    updateSelection();
}

void AgentSelection::initializeAgents() {
    // Q-Learning Agent (Implemented)
    AgentConfig qLearning;
    qLearning.type = AgentType::Q_LEARNING;
    qLearning.name = "Q-Learning Agent";
    qLearning.description = "Tabular reinforcement learning with epsilon-greedy exploration";
    qLearning.isImplemented = true;
    qLearning.modelPath = "qtable.json";
    m_agents.emplace_back(qLearning);
    
    // Deep Q-Network (Placeholder)
    AgentConfig dqn;
    dqn.type = AgentType::DEEP_Q_NETWORK;
    dqn.name = "Deep Q-Network (DQN)";
    dqn.description = "Neural network-based Q-learning with experience replay";
    dqn.isImplemented = false;
    dqn.modelPath = "dqn_model.bin";
    dqn.hiddenLayers = 3;
    dqn.neuronsPerLayer = 128;
    m_agents.emplace_back(dqn);
    
    // Policy Gradient (Placeholder)
    AgentConfig pg;
    pg.type = AgentType::POLICY_GRADIENT;
    pg.name = "Policy Gradient";
    pg.description = "Direct policy optimization using REINFORCE algorithm";
    pg.isImplemented = false;
    pg.modelPath = "policy_model.bin";
    pg.learningRate = 0.001f;
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
    sf::Vector2u windowSize = sf::Vector2u(1200, 800); // Default size
    
    // Background panel
    item.background.setSize(sf::Vector2f(800.0f, 80.0f));
    item.background.setPosition(sf::Vector2f(windowSize.x / 2.0f - 400.0f, y));
    item.background.setFillColor(sf::Color(40, 40, 40));
    item.background.setOutlineThickness(2.0f);
    item.background.setOutlineColor(sf::Color(80, 80, 80));
    
    // Agent name
    item.nameText = std::make_unique<sf::Text>(m_font);
    item.nameText->setString(item.config.name);
    item.nameText->setCharacterSize(24);
    item.nameText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 380.0f, y + 10.0f));
    
    // Description
    item.descText = std::make_unique<sf::Text>(m_font);
    item.descText->setString(item.config.description);
    item.descText->setCharacterSize(16);
    item.descText->setFillColor(sf::Color(180, 180, 180));
    item.descText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 380.0f, y + 40.0f));
    
    // Status
    item.statusText = std::make_unique<sf::Text>(m_font);
    item.statusText->setString(item.config.isImplemented ? "READY" : "COMING SOON");
    item.statusText->setCharacterSize(18);
    item.statusText->setFillColor(item.config.isImplemented ? sf::Color::Green : sf::Color::Yellow);
    item.statusText->setPosition(sf::Vector2f(windowSize.x / 2.0f + 250.0f, y + 25.0f));
}

void AgentSelection::handleEvent(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        switch (keyPressed->code) {
            case sf::Keyboard::Key::Up:
                m_selectedIndex = (m_selectedIndex - 1 + m_agents.size()) % m_agents.size();
                updateSelection();
                break;
            case sf::Keyboard::Key::Down:
                m_selectedIndex = (m_selectedIndex + 1) % m_agents.size();
                updateSelection();
                break;
            case sf::Keyboard::Key::Enter:
                if (m_agents[m_selectedIndex].config.isImplemented && m_selectionCallback) {
                    m_selectionCallback(m_agents[m_selectedIndex].config);
                }
                break;
            case sf::Keyboard::Key::Escape:
                if (m_backCallback) {
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
    if (m_instructions) window.draw(*m_instructions);
    
    for (const auto& agent : m_agents) {
        window.draw(agent.background);
        if (agent.nameText) window.draw(*agent.nameText);
        if (agent.descText) window.draw(*agent.descText);
        if (agent.statusText) window.draw(*agent.statusText);
    }
}

void AgentSelection::updateSelection() {
    for (size_t i = 0; i < m_agents.size(); ++i) {
        if (i == m_selectedIndex) {
            m_agents[i].background.setFillColor(sf::Color(60, 100, 60));
            m_agents[i].background.setOutlineColor(sf::Color::Green);
            if (m_agents[i].nameText) {
                m_agents[i].nameText->setFillColor(sf::Color::White);
                m_agents[i].nameText->setStyle(sf::Text::Bold);
            }
        } else {
            m_agents[i].background.setFillColor(sf::Color(40, 40, 40));
            m_agents[i].background.setOutlineColor(sf::Color(80, 80, 80));
            if (m_agents[i].nameText) {
                m_agents[i].nameText->setFillColor(m_agents[i].config.isImplemented ? 
                                                  sf::Color::White : sf::Color(120, 120, 120));
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