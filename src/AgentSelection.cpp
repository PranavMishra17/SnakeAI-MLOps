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
    
    // Yellow theme background
    m_background.setSize(sf::Vector2f(windowSize.x, windowSize.y));
    m_background.setFillColor(sf::Color(255, 253, 208));
    
    // Title panel
    sf::RectangleShape titlePanel;
    titlePanel.setSize(sf::Vector2f(windowSize.x - 100.0f, 100.0f));
    titlePanel.setPosition(sf::Vector2f(50.0f, 20.0f));
    titlePanel.setFillColor(sf::Color(255, 248, 220, 150));
    titlePanel.setOutlineThickness(2.0f);
    titlePanel.setOutlineColor(sf::Color(218, 165, 32, 100));
    
    // Enhanced title with yellow theme
    m_title = std::make_unique<sf::Text>(m_font);
    m_title->setString("AI AGENT SELECTION");
    m_title->setCharacterSize(44);
    m_title->setFillColor(sf::Color(139, 69, 19));
    m_title->setStyle(sf::Text::Bold);
    
    auto titleBounds = m_title->getLocalBounds();
    m_title->setPosition(sf::Vector2f((windowSize.x - titleBounds.size.x) / 2.0f, 35.0f));
    
    // Section title with yellow theme
    m_sectionTitle = std::make_unique<sf::Text>(m_font);
    m_sectionTitle->setString("Choose Your AI Opponent");
    m_sectionTitle->setCharacterSize(24);
    m_sectionTitle->setFillColor(sf::Color(160, 82, 45));
    m_sectionTitle->setPosition(sf::Vector2f(80.0f, 90.0f));
    
    // Performance summary panel - yellow theme
    m_summaryPanel.setSize(sf::Vector2f(500.0f, 50.0f));
    m_summaryPanel.setPosition(sf::Vector2f(windowSize.x / 2.0f - 250.0f, 130.0f));
    m_summaryPanel.setFillColor(sf::Color(255, 250, 205, 200));
    m_summaryPanel.setOutlineThickness(2.0f);
    m_summaryPanel.setOutlineColor(sf::Color(218, 165, 32));
    
    m_summaryText = std::make_unique<sf::Text>(m_font);
    m_summaryText->setCharacterSize(14);
    m_summaryText->setFillColor(sf::Color(101, 67, 33));
    m_summaryText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 240.0f, 145.0f));
    
    // Instructions with yellow theme
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setString("UP/DOWN: Navigate | ENTER: Select | ESC: Back");
    m_instructions->setCharacterSize(18);
    m_instructions->setFillColor(sf::Color(101, 67, 33));
    m_instructions->setPosition(sf::Vector2f(70.0f, windowSize.y - 60.0f));
    
    loadEvaluationData();
    initializeAgents();
    loadTrainedModels();
    updateSummaryPanel();
    
    // Create enhanced displays
    float startY = 200.0f;
    for (size_t i = 0; i < m_agents.size(); ++i) {
        createAgentDisplay(m_agents[i], startY + i * 125.0f);
        createPerformanceDisplay(m_agents[i], startY + i * 125.0f);
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
    
    // Yellow theme background panel
    float panelHeight = item.hasPerformanceData ? 110.0f : 80.0f;
    item.background.setSize(sf::Vector2f(750.0f, panelHeight));
    item.background.setPosition(sf::Vector2f(windowSize.x / 2.0f - 375.0f, y));
    item.background.setFillColor(sf::Color(255, 248, 220, 240));
    item.background.setOutlineThickness(2.0f);
    item.background.setOutlineColor(sf::Color(218, 165, 32));
    
    // Agent name with yellow theme styling
    item.nameText = std::make_unique<sf::Text>(m_font);
    item.nameText->setString(item.config.name);
    item.nameText->setCharacterSize(22);
    item.nameText->setFillColor(sf::Color(139, 69, 19));
    item.nameText->setStyle(sf::Text::Bold);
    item.nameText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 355.0f, y + 10.0f));
    
    // Description with brown color
    item.descText = std::make_unique<sf::Text>(m_font);
    item.descText->setString(item.config.description);
    item.descText->setCharacterSize(14);
    item.descText->setFillColor(sf::Color(101, 67, 33));
    item.descText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 355.0f, y + 35.0f));
    
    // Enhanced status with training info
    item.statusText = std::make_unique<sf::Text>(m_font);
    std::string statusStr;
    sf::Color statusColor;
    
    if (item.isTrainedModel && item.hasPerformanceData) {
        statusStr = "TRAINED (" + item.modelInfo.performance.trainingProfile + ")";
        statusColor = sf::Color(160, 82, 45);
    } else if (item.config.isImplemented) {
        statusStr = "READY TO TRAIN";
        statusColor = sf::Color(218, 165, 32);
    } else {
        statusStr = "COMING SOON";
        statusColor = sf::Color(205, 133, 63);
    }
    
    item.statusText->setString(statusStr);
    item.statusText->setCharacterSize(16);
    item.statusText->setFillColor(statusColor);
    item.statusText->setStyle(sf::Text::Bold);
    item.statusText->setPosition(sf::Vector2f(windowSize.x / 2.0f + 150.0f, y + 15.0f));
}

void AgentSelection::createPerformanceDisplay(EnhancedAgentMenuItem& item, float y) {
    if (!item.hasPerformanceData) return;
    
    sf::Vector2u windowSize = sf::Vector2u(1200, 800);
    
    // Performance panel with yellow theme
    item.performancePanel.setSize(sf::Vector2f(730.0f, 30.0f));
    item.performancePanel.setPosition(sf::Vector2f(windowSize.x / 2.0f - 365.0f, y + 55.0f));
    item.performancePanel.setFillColor(sf::Color(255, 250, 205, 150));
    item.performancePanel.setOutlineThickness(1.0f);
    item.performancePanel.setOutlineColor(sf::Color(218, 165, 32));
    
    // Best score badge
    if (item.modelInfo.performance.bestScore >= 20.0f) {
        item.bestScoreBadge.setSize(sf::Vector2f(70.0f, 22.0f));
        item.bestScoreBadge.setPosition(sf::Vector2f(windowSize.x / 2.0f + 280.0f, y + 45.0f));
        item.bestScoreBadge.setFillColor(sf::Color(255, 215, 0, 200));
        item.bestScoreBadge.setOutlineThickness(1.0f);
        item.bestScoreBadge.setOutlineColor(sf::Color(218, 165, 32));
        
        item.bestScoreText = std::make_unique<sf::Text>(m_font);
        item.bestScoreText->setString("ELITE");
        item.bestScoreText->setCharacterSize(10);
        item.bestScoreText->setFillColor(sf::Color(139, 69, 19));
        item.bestScoreText->setStyle(sf::Text::Bold);
        item.bestScoreText->setPosition(sf::Vector2f(windowSize.x / 2.0f + 295.0f, y + 50.0f));
    }
    
    // Performance metrics text
    item.performanceText = std::make_unique<sf::Text>(m_font);
    std::string perfStr = formatPerformanceMetrics(item.modelInfo.performance);
    item.performanceText->setString(perfStr);
    item.performanceText->setCharacterSize(12);
    item.performanceText->setFillColor(sf::Color(101, 67, 33));
    item.performanceText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 355.0f, y + 60.0f));
    
    // Detailed stats for top performers
    if (item.modelInfo.performance.bestScore > 15.0f) {
        item.detailsText = std::make_unique<sf::Text>(m_font);
        item.detailsText->setString(item.modelInfo.getDetailedStats());
        item.detailsText->setCharacterSize(11);
        item.detailsText->setFillColor(sf::Color(139, 69, 19));
        item.detailsText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 355.0f, y + 80.0f));
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
            // Golden highlight for selected agent
            sf::Color highlightColor = sf::Color(255, 215, 0, 250);
            
            m_agents[i].background.setFillColor(highlightColor);
            m_agents[i].background.setOutlineColor(sf::Color(255, 140, 0));
            m_agents[i].background.setOutlineThickness(3.0f);
            
            if (m_agents[i].nameText) {
                m_agents[i].nameText->setFillColor(sf::Color(139, 69, 19));
                m_agents[i].nameText->setStyle(sf::Text::Bold);
            }
        } else {
            // Normal yellow theme appearance
            m_agents[i].background.setFillColor(sf::Color(255, 248, 220, 240));
            m_agents[i].background.setOutlineColor(sf::Color(218, 165, 32));
            m_agents[i].background.setOutlineThickness(2.0f);
            
            if (m_agents[i].nameText) {
                m_agents[i].nameText->setFillColor(sf::Color(139, 69, 19));
                m_agents[i].nameText->setStyle(sf::Text::Bold);
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