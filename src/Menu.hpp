#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <functional>
#include "GameState.hpp"

struct MenuItem {
    std::string text;
    GameMode mode;
    std::string explanation;
    std::unique_ptr<sf::Text> displayText;
    std::unique_ptr<sf::Text> explanationText;
    sf::RectangleShape explanationBox;
    sf::RectangleShape buttonBackground;
    bool selected;
    bool isPlayMode;
    bool isSpecialAction; // For quit, leaderboard, etc.
    
    MenuItem(const std::string& t, GameMode m, const std::string& exp = "", bool isPlay = true, bool special = false) 
        : text(t), mode(m), explanation(exp), selected(false), isPlayMode(isPlay), isSpecialAction(special) {}
};

enum class MenuSection {
    MAIN,
    PLAY_MODES
};

class Menu {
public:
    Menu();
    void initialize(sf::RenderWindow& window);
    void handleEvent(const sf::Event& event);
    void update();
    void render(sf::RenderWindow& window);
    
    void setSelectionCallback(std::function<void(GameMode)> callback);
    void setSettingsCallback(std::function<void()> callback);
    void setHowToPlayCallback(std::function<void()> callback);
    void setLeaderboardCallback(std::function<void()> callback);
    void setQuitCallback(std::function<void()> callback);
    
private:
    std::vector<MenuItem> m_mainItems;
    std::vector<MenuItem> m_playModeItems;
    MenuSection m_currentSection;
    int m_selectedIndex;
    sf::Font m_font;
    std::unique_ptr<sf::Text> m_title;
    std::unique_ptr<sf::Text> m_instructions;
    std::unique_ptr<sf::Text> m_sectionTitle;
    sf::RectangleShape m_background;
    
    std::function<void(GameMode)> m_selectionCallback;
    std::function<void()> m_settingsCallback;
    std::function<void()> m_howToPlayCallback;
    std::function<void()> m_leaderboardCallback;
    std::function<void()> m_quitCallback;
    
    void updateSelection();
    void setupMainMenu();
    void setupPlayModeMenu();
    void renderMainMenu(sf::RenderWindow& window);
    void renderPlayModeMenu(sf::RenderWindow& window);
};