#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <functional>
#include "GameState.hpp"
#include "MLAgents.hpp"

class AgentSelection {
public:
    AgentSelection();
    void initialize(sf::RenderWindow& window);
    void handleEvent(const sf::Event& event);
    void update();
    void render(sf::RenderWindow& window);
    
    void setSelectionCallback(std::function<void(const AgentConfig&)> callback);
    void setBackCallback(std::function<void()> callback);
    
    const AgentConfig& getSelectedAgent() const { return m_agents[m_selectedIndex].config; }
    
private:
    struct EnhancedAgentMenuItem {
        AgentConfig config;
        TrainedModelInfo modelInfo;
        
        // UI Elements
        std::unique_ptr<sf::Text> nameText;
        std::unique_ptr<sf::Text> descText;
        std::unique_ptr<sf::Text> statusText;
        std::unique_ptr<sf::Text> performanceText;
        sf::RectangleShape background;
        sf::RectangleShape performancePanel;
        
        bool isTrainedModel = false;
        bool hasPerformanceData = false;
        
        EnhancedAgentMenuItem(const AgentConfig& cfg) : config(cfg) {}
    };
    
    std::vector<EnhancedAgentMenuItem> m_agents;
    int m_selectedIndex;
    sf::Font m_font;
    
    // UI Elements
    std::unique_ptr<sf::Text> m_title;
    std::unique_ptr<sf::Text> m_instructions;
    std::unique_ptr<sf::Text> m_sectionTitle;
    std::unique_ptr<sf::Text> m_summaryText;
    sf::RectangleShape m_background;
    sf::RectangleShape m_summaryPanel;
    
    // Model management
    std::unique_ptr<TrainedModelManager> m_modelManager;
    EvaluationReportData m_evaluationData;
    
    // Callbacks
    std::function<void(const AgentConfig&)> m_selectionCallback;
    std::function<void()> m_backCallback;
    
    void initializeAgents();
    void loadTrainedModels();
    void loadEvaluationData();
    void updateSelection();
    void createAgentDisplay(EnhancedAgentMenuItem& item, float y);
    void createPerformanceDisplay(EnhancedAgentMenuItem& item, float y);
    void updateSummaryPanel();
    
    // Performance helpers
    std::string formatPerformanceMetrics(const ModelPerformanceData& data) const;
};