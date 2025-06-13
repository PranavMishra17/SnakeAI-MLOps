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
    struct AgentMenuItem {
        AgentConfig config;
        std::unique_ptr<sf::Text> nameText;
        std::unique_ptr<sf::Text> descText;
        std::unique_ptr<sf::Text> statusText;
        std::unique_ptr<sf::Text> modelInfoText;  // New: Show model performance info
        sf::RectangleShape background;
        bool isTrainedModel = false;
        
        AgentMenuItem(const AgentConfig& cfg) : config(cfg) {}
    };
    
    std::vector<AgentMenuItem> m_agents;
    int m_selectedIndex;
    sf::Font m_font;
    
    // UI Elements
    std::unique_ptr<sf::Text> m_title;
    std::unique_ptr<sf::Text> m_instructions;
    std::unique_ptr<sf::Text> m_sectionTitle;
    sf::RectangleShape m_background;
    
    // Model management
    std::unique_ptr<TrainedModelManager> m_modelManager;
    
    // Callbacks
    std::function<void(const AgentConfig&)> m_selectionCallback;
    std::function<void()> m_backCallback;
    
    void initializeAgents();
    void loadTrainedModels();
    void updateSelection();
    void createAgentDisplay(AgentMenuItem& item, float y);
};