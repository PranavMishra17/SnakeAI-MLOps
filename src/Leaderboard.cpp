#include "Leaderboard.hpp"
#include <spdlog/spdlog.h>

const std::string Leaderboard::LEADERBOARD_PATH = "leaderboard.json";

Leaderboard::Leaderboard() : m_state(LeaderboardState::VIEWING), m_pendingScore(0), 
                           m_pendingAgentType(AgentType::HUMAN), m_pendingEpisode(0) {}

void Leaderboard::initialize(sf::RenderWindow& window) {
    if (!m_font.openFromFile("assets/fonts/arial.ttf")) {
        spdlog::error("Failed to load font in Leaderboard");
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Background
    m_background.setSize(sf::Vector2f(static_cast<float>(windowSize.x), 
                                     static_cast<float>(windowSize.y)));
    m_background.setFillColor(sf::Color(20, 20, 30));
    
    // Title
    m_title = std::make_unique<sf::Text>(m_font);
    m_title->setString("🏆 LEADERBOARD 🏆");
    m_title->setCharacterSize(48);
    m_title->setFillColor(sf::Color::Yellow);
    m_title->setPosition(sf::Vector2f(windowSize.x / 2.0f - 200.0f, 30.0f));
    
    // Instructions
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setString("ESC: Back to Menu");
    m_instructions->setCharacterSize(20);
    m_instructions->setFillColor(sf::Color(150, 150, 150));
    m_instructions->setPosition(sf::Vector2f(50.0f, windowSize.y - 50.0f));
    
    // Name input prompt
    m_namePrompt = std::make_unique<sf::Text>(m_font);
    m_namePrompt->setString("Enter your name (ENTER to confirm):");
    m_namePrompt->setCharacterSize(24);
    m_namePrompt->setFillColor(sf::Color::White);
    m_namePrompt->setPosition(sf::Vector2f(windowSize.x / 2.0f - 250.0f, windowSize.y / 2.0f - 100.0f));
    
    // Input text display
    m_inputText = std::make_unique<sf::Text>(m_font);
    m_inputText->setCharacterSize(32);
    m_inputText->setFillColor(sf::Color::Cyan);
    m_inputText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 150.0f, windowSize.y / 2.0f - 50.0f));
    
    // Input box
    m_inputBox.setSize(sf::Vector2f(300.0f, 50.0f));
    m_inputBox.setPosition(sf::Vector2f(windowSize.x / 2.0f - 150.0f, windowSize.y / 2.0f - 60.0f));
    m_inputBox.setFillColor(sf::Color(40, 40, 40));
    m_inputBox.setOutlineThickness(2.0f);
    m_inputBox.setOutlineColor(sf::Color::Cyan);
    
    loadLeaderboard();
    updateDisplay();
}

void Leaderboard::handleEvent(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if (keyPressed->code == sf::Keyboard::Key::Escape) {
            if (m_state == LeaderboardState::ENTERING_NAME) {
                finalizeName();
            } else if (m_backCallback) {
                m_backCallback();
            }
        } else if (m_state == LeaderboardState::ENTERING_NAME) {
            if (keyPressed->code == sf::Keyboard::Key::Enter) {
                finalizeName();
            } else if (keyPressed->code == sf::Keyboard::Key::Backspace && !m_currentInput.empty()) {
                m_currentInput.pop_back();
                m_inputText->setString(m_currentInput + "_");
            }
        }
    }
    
    if (auto* textEntered = event.getIf<sf::Event::TextEntered>()) {
        if (m_state == LeaderboardState::ENTERING_NAME) {
            handleNameInput(textEntered->unicode);
        }
    }
}

void Leaderboard::handleNameInput(std::uint32_t unicode) {
    if (unicode >= 32 && unicode < 127 && m_currentInput.length() < MAX_NAME_LENGTH) {
        m_currentInput += static_cast<char>(unicode);
        m_inputText->setString(m_currentInput + "_");
    }
}

void Leaderboard::update() {
    if (m_state == LeaderboardState::ENTERING_NAME) {
        static sf::Clock clock;
        if (clock.getElapsedTime().asSeconds() > 0.5f) {
            static bool showCursor = true;
            showCursor = !showCursor;
            m_inputText->setString(m_currentInput + (showCursor ? "_" : " "));
            clock.restart();
        }
    }
}

void Leaderboard::render(sf::RenderWindow& window) {
    window.draw(m_background);
    if (m_title) window.draw(*m_title);
    if (m_instructions) window.draw(*m_instructions);
    
    if (m_state == LeaderboardState::VIEWING) {
        for (const auto& text : m_entryTexts) {
            if (text) window.draw(*text);
        }
    } else if (m_state == LeaderboardState::ENTERING_NAME) {
        if (m_namePrompt) window.draw(*m_namePrompt);
        window.draw(m_inputBox);
        if (m_inputText) window.draw(*m_inputText);
    }
}

void Leaderboard::addScore(const std::string& playerName, AgentType agentType, int score, int episode) {
    LeaderboardEntry entry;
    entry.playerName = playerName.empty() ? getDefaultName(agentType) : playerName;
    entry.agentType = agentType;
    entry.score = score;
    entry.episode = episode;
    entry.timestamp = std::chrono::system_clock::now();
    entry.efficiency = episode > 0 ? static_cast<float>(score) / episode : 0.0f;
    
    m_entries.push_back(entry);
    sortEntries();
    
    if (m_entries.size() > MAX_ENTRIES) {
        m_entries.resize(MAX_ENTRIES);
    }
    
    saveLeaderboard();
    updateDisplay();
}

void Leaderboard::promptForName(int score, AgentType agentType, int episode) {
    if (agentType == AgentType::HUMAN) {
        m_state = LeaderboardState::ENTERING_NAME;
        m_pendingScore = score;
        m_pendingAgentType = agentType;
        m_pendingEpisode = episode;
        m_currentInput.clear();
        m_inputText->setString("_");
    } else {
        addScore("", agentType, score, episode);
    }
}

void Leaderboard::finalizeName() {
    std::string finalName = m_currentInput.empty() ? getDefaultName(m_pendingAgentType) : m_currentInput;
    addScore(finalName, m_pendingAgentType, m_pendingScore, m_pendingEpisode);
    m_state = LeaderboardState::VIEWING;
    m_currentInput.clear();
}

std::string Leaderboard::getDefaultName(AgentType agentType) {
    switch (agentType) {
        case AgentType::HUMAN: return "Anonymous";
        case AgentType::Q_LEARNING: return "Q-Agent";
        case AgentType::DEEP_Q_NETWORK: return "DQN-Agent";
        case AgentType::POLICY_GRADIENT: return "PG-Agent";
        case AgentType::ACTOR_CRITIC: return "AC-Agent";
        case AgentType::GENETIC_ALGORITHM: return "GA-Agent";
        default: return "Unknown";
    }
}

void Leaderboard::sortEntries() {
    std::sort(m_entries.begin(), m_entries.end(), 
              [](const LeaderboardEntry& a, const LeaderboardEntry& b) {
                  if (a.score != b.score) return a.score > b.score;
                  return a.efficiency > b.efficiency;
              });
}

void Leaderboard::updateDisplay() {
    m_entryTexts.clear();
    
    sf::Vector2u windowSize = sf::Vector2u(1200, 800);
    float startY = 120.0f;
    
    auto header = std::make_unique<sf::Text>(m_font);
    header->setString("Rank  Player Name              Agent Type      Score  Episode  Efficiency");
    header->setCharacterSize(18);
    header->setFillColor(sf::Color(200, 200, 200));
    header->setPosition(sf::Vector2f(50.0f, startY));
    m_entryTexts.push_back(std::move(header));
    
    for (size_t i = 0; i < m_entries.size() && i < MAX_ENTRIES; ++i) {
        const auto& entry = m_entries[i];
        
        std::string rankStr = "#" + std::to_string(i + 1);
        std::string nameStr = entry.playerName;
        if (nameStr.length() > 15) nameStr = nameStr.substr(0, 12) + "...";
        
        std::string line = rankStr;
        line.resize(6, ' ');
        line += nameStr;
        line.resize(30, ' ');
        line += entry.getAgentTypeString();
        line.resize(46, ' ');
        line += std::to_string(entry.score);
        line.resize(53, ' ');
        line += std::to_string(entry.episode);
        line.resize(62, ' ');
        line += std::to_string(entry.efficiency).substr(0, 5);
        
        auto entryText = std::make_unique<sf::Text>(m_font);
        entryText->setString(line);
        entryText->setCharacterSize(16);
        
        sf::Color color = sf::Color::White;
        if (i == 0) color = sf::Color::Yellow;
        else if (i == 1) color = sf::Color(192, 192, 192);
        else if (i == 2) color = sf::Color(205, 127, 50);
        
        entryText->setFillColor(color);
        entryText->setPosition(sf::Vector2f(50.0f, startY + 40.0f + i * 30.0f));
        
        m_entryTexts.push_back(std::move(entryText));
    }
    
    if (m_entries.empty()) {
        auto emptyText = std::make_unique<sf::Text>(m_font);
        emptyText->setString("No scores yet! Be the first to play!");
        emptyText->setCharacterSize(24);
        emptyText->setFillColor(sf::Color(150, 150, 150));
        emptyText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 200.0f, windowSize.y / 2.0f));
        m_entryTexts.push_back(std::move(emptyText));
    }
}

void Leaderboard::loadLeaderboard() {
    std::ifstream file(LEADERBOARD_PATH);
    if (!file.is_open()) return;
    
    try {
        nlohmann::json j;
        file >> j;
        
        m_entries.clear();
        for (const auto& item : j["entries"]) {
            LeaderboardEntry entry;
            entry.playerName = item["playerName"];
            entry.agentType = static_cast<AgentType>(item["agentType"]);
            entry.score = item["score"];
            entry.episode = item["episode"];
            entry.efficiency = item.value("efficiency", 0.0f);
            entry.timestamp = std::chrono::system_clock::now();
            
            m_entries.push_back(entry);
        }
        
        sortEntries();
    } catch (const std::exception& e) {
        spdlog::error("Failed to load leaderboard: {}", e.what());
        m_entries.clear();
    }
}

void Leaderboard::saveLeaderboard() {
    try {
        nlohmann::json j;
        j["entries"] = nlohmann::json::array();
        
        for (const auto& entry : m_entries) {
            nlohmann::json entryJson;
            entryJson["playerName"] = entry.playerName;
            entryJson["agentType"] = static_cast<int>(entry.agentType);
            entryJson["score"] = entry.score;
            entryJson["episode"] = entry.episode;
            entryJson["efficiency"] = entry.efficiency;
            
            auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
            entryJson["timestamp"] = std::to_string(time_t);
            
            j["entries"].push_back(entryJson);
        }
        
        std::ofstream file(LEADERBOARD_PATH);
        if (file.is_open()) {
            file << j.dump(4);
            spdlog::info("Leaderboard saved with {} entries", m_entries.size());
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to save leaderboard: {}", e.what());
    }
}