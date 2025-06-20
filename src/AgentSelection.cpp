#include "AgentSelection.hpp"
#include <spdlog/spdlog.h>
#include <optional>
#include <sstream>
#include <iomanip>

AgentSelection::AgentSelection() : m_selectedIndex(0) {
    m_modelManager = std::make_unique<TrainedModelManager>();
}

void AgentSelection::initialize(sf::RenderWindow& window) {
    std::vector<std::string> fontPaths = {
        "assets/fonts/ARIAL.TTF", "assets/fonts/arial.ttf", 
        "assets/fonts/ArialCE.ttf", "assets/fonts/Roboto.ttf"
    };
    
    for (const auto& path : fontPaths) {
        if (m_font.openFromFile(path)) {
            spdlog::info("AgentSelection: Font loaded from: {}", path);
            break;
        }
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Enhanced background with gradient effect
    m_background.setSize(sf::Vector2f(windowSize.x, windowSize.y));
    m_background.setFillColor(sf::Color(245, 245, 220));
    
    // Title with enhanced styling
    m_title = std::make_unique<sf::Text>(m_font);
    m_title->setString("AI Agent Selection");
    m_title->setCharacterSize(48);
    m_title->setFillColor(sf::Color(47, 79, 47));
    m_title->setPosition(sf::Vector2f(windowSize.x / 2.0f - 220.0f, 20.0f));
    
    // Section title
    m_sectionTitle = std::make_unique<sf::Text>(m_font);
    m_sectionTitle->setString("Choose Your AI Opponent");
    m_sectionTitle->setCharacterSize(28);
    m_sectionTitle->setFillColor(sf::Color(70, 130, 180));
    m_sectionTitle->setPosition(sf::Vector2f(windowSize.x / 2.0f - 180.0f, 80.0f));
    
    // Performance summary panel
    m_summaryPanel.setSize(sf::Vector2f(600.0f, 60.0f));
    m_summaryPanel.setPosition(sf::Vector2f(windowSize.x / 2.0f - 300.0f, 120.0f));
    m_summaryPanel.setFillColor(sf::Color(240, 248, 255, 200));
    m_summaryPanel.setOutlineThickness(2.0f);
    m_summaryPanel.setOutlineColor(sf::Color(70, 130, 180));
    
    m_summaryText = std::make_unique<sf::Text>(m_font);
    m_summaryText->setCharacterSize(16);
    m_summaryText->setFillColor(sf::Color(47, 79, 47));
    m_summaryText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 290.0f, 135.0f));
    
    // Instructions with enhanced info
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setString("UP/DOWN: Navigate | ENTER: Select | ESC: Back | Performance data from training evaluation");
    m_instructions->setCharacterSize(18);
    m_instructions->setFillColor(sf::Color(47, 79, 47));
    m_instructions->setPosition(sf::Vector2f(50.0f, windowSize.y - 60.0f));
    
    loadEvaluationData();
    initializeAgents();
    loadTrainedModels();
    updateSummaryPanel();
    
    // Create enhanced displays
    float startY = 200.0f;
    for (size_t i = 0; i < m_agents.size(); ++i) {
        createAgentDisplay(m_agents[i], startY + i * 140.0f);
        createPerformanceDisplay(m_agents[i], startY + i * 140.0f);
    }
    
    updateSelection();
    
    spdlog::info("AgentSelection: Initialized with {} agents", m_agents.size());
}

void AgentSelection::loadEvaluationData() {
    m_evaluationData = m_modelManager->getEvaluationData();
    spdlog::info("AgentSelection: Loaded evaluation data for {} models", 
                 m_evaluationData.modelPerformance.size());
}

void AgentSelection::initializeAgents() {
    m_agents.clear();
    
    // Basic Q-Learning Agent (from scratch)
    AgentConfig basicQLearning;
    basicQLearning.type = AgentType::Q_LEARNING;
    basicQLearning.name = "Q-Learning (Fresh)";
    basicQLearning.description = "Train new Q-Learning agent from scratch";
    basicQLearning.isImplemented = true;
    basicQLearning.modelPath = "";
    m_agents.emplace_back(basicQLearning);
}

void AgentSelection::loadTrainedModels() {
    auto trainedModels = m_modelManager->getAvailableModels();
    
    for (const auto& modelInfo : trainedModels) {
        AgentConfig config;
        config.type = modelInfo.modelType == "qlearning" ? AgentType::Q_LEARNING :
                     modelInfo.modelType == "dqn" ? AgentType::DEEP_Q_NETWORK :
                     modelInfo.modelType == "ppo" ? AgentType::PPO :
                     modelInfo.modelType == "actor_critic" ? AgentType::ACTOR_CRITIC :
                     AgentType::Q_LEARNING;
        
        config.isImplemented = true;
        config.modelPath = modelInfo.modelPath;
        config.name = modelInfo.name;
        config.description = modelInfo.description;
        
        EnhancedAgentMenuItem item(config);
        item.modelInfo = modelInfo;
        item.isTrainedModel = true;
        item.hasPerformanceData = (modelInfo.performance.bestScore > 0);
        
        m_agents.push_back(std::move(item));
        
        spdlog::info("AgentSelection: Added trained model: {} (Best: {:.1f})", 
                     config.name, modelInfo.performance.bestScore);
    }
}

void AgentSelection::createAgentDisplay(EnhancedAgentMenuItem& item, float y) {
    sf::Vector2u windowSize = sf::Vector2u(1200, 800);
    
    // Enhanced background panel
    float panelHeight = item.hasPerformanceData ? 120.0f : 90.0f;
    item.background.setSize(sf::Vector2f(900.0f, panelHeight));
    item.background.setPosition(sf::Vector2f(windowSize.x / 2.0f - 450.0f, y));
    item.background.setFillColor(sf::Color(255, 255, 240));
    item.background.setOutlineThickness(3.0f);
    item.background.setOutlineColor(sf::Color(144, 238, 144));
    
    // Agent name with enhanced styling
    item.nameText = std::make_unique<sf::Text>(m_font);
    item.nameText->setString(item.config.name);
    item.nameText->setCharacterSize(24);
    item.nameText->setFillColor(sf::Color(47, 79, 47));
    item.nameText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 430.0f, y + 10.0f));
    
    // Description
    item.descText = std::make_unique<sf::Text>(m_font);
    item.descText->setString(item.config.description);
    item.descText->setCharacterSize(16);
    item.descText->setFillColor(sf::Color(25, 25, 112));
    item.descText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 430.0f, y + 40.0f));
    
    // Enhanced status with training info
    item.statusText = std::make_unique<sf::Text>(m_font);
    std::string statusStr;
    sf::Color statusColor;
    
    if (item.isTrainedModel && item.hasPerformanceData) {
        statusStr = "TRAINED (" + item.modelInfo.performance.trainingProfile + ")";
        statusColor = sf::Color(34, 139, 34);
    } else if (item.config.isImplemented) {
        statusStr = "READY TO TRAIN";
        statusColor = sf::Color(70, 130, 180);
    } else {
        statusStr = "COMING SOON";
        statusColor = sf::Color(255, 140, 0);
    }
    
    item.statusText->setString(statusStr);
    item.statusText->setCharacterSize(18);
    item.statusText->setFillColor(statusColor);
    item.statusText->setPosition(sf::Vector2f(windowSize.x / 2.0f + 200.0f, y + 15.0f));
}

void AgentSelection::createPerformanceDisplay(EnhancedAgentMenuItem& item, float y) {
    if (!item.hasPerformanceData) return;
    
    sf::Vector2u windowSize = sf::Vector2u(1200, 800);
    
    // Performance panel
    item.performancePanel.setSize(sf::Vector2f(880.0f, 35.0f));
    item.performancePanel.setPosition(sf::Vector2f(windowSize.x / 2.0f - 440.0f, y + 65.0f));
    item.performancePanel.setFillColor(sf::Color(240, 255, 240, 150));
    item.performancePanel.setOutlineThickness(1.0f);
    item.performancePanel.setOutlineColor(sf::Color(34, 139, 34));
    
    // Best score badge
    if (item.modelInfo.performance.bestScore >= 20.0f) {
        item.bestScoreBadge.setSize(sf::Vector2f(80.0f, 25.0f));
        item.bestScoreBadge.setPosition(sf::Vector2f(windowSize.x / 2.0f + 320.0f, y + 50.0f));
        item.bestScoreBadge.setFillColor(sf::Color(255, 215, 0, 200)); // Gold
        item.bestScoreBadge.setOutlineThickness(1.0f);
        item.bestScoreBadge.setOutlineColor(sf::Color(218, 165, 32));
        
        item.bestScoreText = std::make_unique<sf::Text>(m_font);
        item.bestScoreText->setString("ELITE");
        item.bestScoreText->setCharacterSize(12);
        item.bestScoreText->setFillColor(sf::Color(139, 69, 19));
        item.bestScoreText->setPosition(sf::Vector2f(windowSize.x / 2.0f + 340.0f, y + 55.0f));
    }
    
    // Performance metrics text
    item.performanceText = std::make_unique<sf::Text>(m_font);
    std::string perfStr = formatPerformanceMetrics(item.modelInfo.performance);
    item.performanceText->setString(perfStr);
    item.performanceText->setCharacterSize(14);
    item.performanceText->setFillColor(getPerformanceColor(item.modelInfo.performance.bestScore));
    item.performanceText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 430.0f, y + 70.0f));
    
    // Detailed stats for top performers
    if (item.modelInfo.performance.bestScore > 15.0f) {
        item.detailsText = std::make_unique<sf::Text>(m_font);
        item.detailsText->setString(item.modelInfo.getDetailedStats());
        item.detailsText->setCharacterSize(12);
        item.detailsText->setFillColor(sf::Color(47, 79, 47));
        item.detailsText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 430.0f, y + 95.0f));
    }
}

std::string AgentSelection::formatPerformanceMetrics(const ModelPerformanceData& data) const {
    std::ostringstream oss;
    oss << "Best Score: " << static_cast<int>(data.bestScore) 
        << " | Avg: " << std::fixed << std::setprecision(1) << data.averageScore
        << " | Episodes: " << data.totalEpisodes;
    
    if (data.successRate > 0) {
        oss << " | Success Rate: " << std::fixed << std::setprecision(1) << data.successRate << "%";
    }
    
    return oss.str();
}

sf::Color AgentSelection::getPerformanceColor(float score) const {
    if (score >= 25.0f) return sf::Color(255, 215, 0);      // Gold
    if (score >= 20.0f) return sf::Color(34, 139, 34);      // Forest Green
    if (score >= 15.0f) return sf::Color(70, 130, 180);     // Steel Blue
    if (score >= 10.0f) return sf::Color(218, 165, 32);     // Goldenrod
    return sf::Color(47, 79, 47);                           // Dark Green
}

void AgentSelection::updateSummaryPanel() {
    std::string summary = m_modelManager->getPerformanceSummary();
    
    if (summary.empty()) {
        summary = "Select an AI agent to compete against. Trained models show actual performance data.";
    } else {
        summary = "🏆 " + summary;
    }
    
    m_summaryText->setString(summary);
}

void AgentSelection::renderPerformanceBars(sf::RenderWindow& window, const EnhancedAgentMenuItem& item, float x, float y) const {
    if (!item.hasPerformanceData) return;
    
    const float barWidth = 100.0f;
    const float barHeight = 8.0f;
    const float maxScore = 30.0f; // Normalize to max expected score
    
    // Score bar
    sf::RectangleShape scoreBar(sf::Vector2f(barWidth * (item.modelInfo.performance.bestScore / maxScore), barHeight));
    scoreBar.setPosition(sf::Vector2f(x, y));
    scoreBar.setFillColor(getPerformanceColor(item.modelInfo.performance.bestScore));
    window.draw(scoreBar);
    
    // Background bar
    sf::RectangleShape bgBar(sf::Vector2f(barWidth, barHeight));
    bgBar.setPosition(sf::Vector2f(x, y));
    bgBar.setFillColor(sf::Color(200, 200, 200, 100));
    bgBar.setOutlineThickness(1.0f);
    bgBar.setOutlineColor(sf::Color(150, 150, 150));
    window.draw(bgBar);
}

void AgentSelection::handleEvent(const sf::Event& event) {
    if (const auto* keyPressedEvent = event.getIf<sf::Event::KeyPressed>()) {
        switch (keyPressedEvent->scancode) {
            case sf::Keyboard::Scancode::Up:
                m_selectedIndex = (m_selectedIndex - 1 + m_agents.size()) % m_agents.size();
                updateSelection();
                break;
            case sf::Keyboard::Scancode::Down:
                m_selectedIndex = (m_selectedIndex + 1) % m_agents.size();
                updateSelection();
                break;
            case sf::Keyboard::Scancode::Enter:
                if (m_agents[m_selectedIndex].config.isImplemented && m_selectionCallback) {
                    spdlog::info("AgentSelection: Selected agent: {}", m_agents[m_selectedIndex].config.name);
                    m_selectionCallback(m_agents[m_selectedIndex].config);
                }
                break;
            case sf::Keyboard::Scancode::Escape:
                if (m_backCallback) {
                    m_backCallback();
                }
                break;
            default:
                break;
        }
    }
}

void AgentSelection::update() {
    // Animation updates if needed
}

void AgentSelection::render(sf::RenderWindow& window) {
    window.draw(m_background);
    if (m_title) window.draw(*m_title);
    if (m_sectionTitle) window.draw(*m_sectionTitle);
    
    // Summary panel
    window.draw(m_summaryPanel);
    if (m_summaryText) window.draw(*m_summaryText);
    
    if (m_instructions) window.draw(*m_instructions);
    
    // Render agents with performance data
    for (const auto& agent : m_agents) {
        window.draw(agent.background);
        if (agent.hasPerformanceData) {
            window.draw(agent.performancePanel);
            if (agent.bestScoreText) {
                window.draw(agent.bestScoreBadge);
                window.draw(*agent.bestScoreText);
            }
        }
        
        if (agent.nameText) window.draw(*agent.nameText);
        if (agent.descText) window.draw(*agent.descText);
        if (agent.statusText) window.draw(*agent.statusText);
        if (agent.performanceText) window.draw(*agent.performanceText);
        if (agent.detailsText) window.draw(*agent.detailsText);
        
        // Render performance bars
        if (agent.hasPerformanceData) {
            sf::Vector2f pos = agent.background.getPosition();
            renderPerformanceBars(window, agent, pos.x + 750.0f, pos.y + 80.0f);
        }
    }
}

void AgentSelection::updateSelection() {
    for (size_t i = 0; i < m_agents.size(); ++i) {
        if (i == m_selectedIndex) {
            // Enhanced highlight for selected agent
            sf::Color highlightColor = m_agents[i].hasPerformanceData ? 
                sf::Color(255, 255, 0) : sf::Color(255, 255, 224);
            
            m_agents[i].background.setFillColor(highlightColor);
            m_agents[i].background.setOutlineColor(sf::Color(255, 140, 0));
            m_agents[i].background.setOutlineThickness(4.0f);
            
            if (m_agents[i].nameText) {
                m_agents[i].nameText->setFillColor(sf::Color(139, 69, 19));
                m_agents[i].nameText->setStyle(sf::Text::Bold);
            }
        } else {
            // Normal appearance
            m_agents[i].background.setFillColor(sf::Color(255, 255, 240));
            m_agents[i].background.setOutlineColor(sf::Color(144, 238, 144));
            m_agents[i].background.setOutlineThickness(3.0f);
            
            if (m_agents[i].nameText) {
                sf::Color textColor = m_agents[i].config.isImplemented ? 
                    sf::Color(47, 79, 47) : sf::Color(169, 169, 169);
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