#include "Leaderboard.hpp"
#include <spdlog/spdlog.h>

const std::string Leaderboard::LEADERBOARD_PATH = "leaderboard.json";

// ImageViewer Implementation
// In ImageViewer constructor, replace:
ImageViewer::ImageViewer(const sf::Texture& imageTexture) 
    : m_imageSprite(imageTexture), m_visible(false) 
{
    // Constructor body - font will be loaded in initialize()
}

void ImageViewer::initialize(sf::RenderWindow& window) {
    std::vector<std::string> fontPaths = {
        "assets/fonts/ARIAL.TTF", "assets/fonts/arial.ttf", 
        "assets/fonts/ArialCE.ttf", "assets/fonts/Roboto.ttf"
    };
    
    for (const auto& path : fontPaths) {
        if (m_font.openFromFile(path)) break;
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Dark semi-transparent background
    m_background.setSize(sf::Vector2f(windowSize.x, windowSize.y));
    m_background.setFillColor(sf::Color(0, 0, 0, 200));
    
    // Image frame
    m_imageFrame.setSize(sf::Vector2f(800, 600));
    m_imageFrame.setPosition(sf::Vector2f((windowSize.x - 800) / 2, (windowSize.y - 600) / 2));
    m_imageFrame.setFillColor(sf::Color::White);
    m_imageFrame.setOutlineThickness(3.0f);
    m_imageFrame.setOutlineColor(sf::Color(70, 130, 180));
    
    // Create text objects with loaded font
    m_titleText = std::make_unique<sf::Text>(m_font);
    m_titleText->setCharacterSize(24);
    m_titleText->setFillColor(sf::Color::White);
    
    m_instructionText = std::make_unique<sf::Text>(m_font);
    m_instructionText->setString("ESC: Close | Click outside to close");
    m_instructionText->setCharacterSize(16);
    m_instructionText->setFillColor(sf::Color(200, 200, 200));
}


void ImageViewer::loadImage(const std::string& imagePath, const std::string& title) {
    if (!std::filesystem::exists(imagePath)) {
        spdlog::warn("ImageViewer: Image not found: {}", imagePath);
        return;
    }
    
    if (!m_imageTexture.loadFromFile(imagePath)) {
        spdlog::error("ImageViewer: Failed to load image: {}", imagePath);
        return;
    }

    m_imageSprite = sf::Sprite(m_imageTexture);
    m_currentTitle = title;
    m_titleText->setString(title);
    
    // Use full window size minus padding
    sf::Vector2f windowSize(1920, 1080); // Get from window if available
    float padding = 100.0f;
    float frameWidth = windowSize.x - padding;
    float frameHeight = windowSize.y - padding;
    
    // Update frame to full screen
    m_imageFrame.setSize(sf::Vector2f(frameWidth, frameHeight));
    m_imageFrame.setPosition(sf::Vector2f(padding/2, padding/2));
    
    // Scale image to fit full frame
    sf::Vector2u imageSize = m_imageTexture.getSize();
    float scaleX = (frameWidth - 40) / imageSize.x;
    float scaleY = (frameHeight - 80) / imageSize.y; // Leave space for title
    float scale = std::min(scaleX, scaleY);
    
    m_imageSprite.setScale(sf::Vector2f(scale, scale));
    
    // Center image in full frame
    sf::Vector2f scaledSize(imageSize.x * scale, imageSize.y * scale);
    float offsetX = (frameWidth - scaledSize.x) / 2;
    float offsetY = (frameHeight - scaledSize.y) / 2 + 40; // Space for title
    
    m_imageSprite.setPosition(sf::Vector2f(padding/2 + offsetX, padding/2 + offsetY));
    
    // Center title
    m_titleText->setPosition(sf::Vector2f(windowSize.x/2 - 100, padding/2 + 10));
    
    m_visible = true;
    spdlog::info("ImageViewer: Loaded full-screen image: {}", imagePath);
}

void ImageViewer::handleEvent(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if (keyPressed->code == sf::Keyboard::Key::Escape) {
            close();
        }
    }
    
    if (auto* mousePressed = event.getIf<sf::Event::MouseButtonPressed>()) {
        sf::Vector2f mousePos(mousePressed->position.x, mousePressed->position.y);
        if (!m_imageFrame.getGlobalBounds().contains(mousePos)) {
            close();
        }
    }
}

void ImageViewer::render(sf::RenderWindow& window) {
    if (!m_visible) return;
    
    window.draw(m_background);
    window.draw(m_imageFrame);
    if (m_titleText) window.draw(*m_titleText);
    window.draw(m_imageSprite);
    if (m_instructionText) window.draw(*m_instructionText);
}

// Leaderboard Implementation
Leaderboard::Leaderboard() 
    : m_state(LeaderboardState::VIEWING), 
      m_currentSection(StatsSection::MAIN_LEADERBOARD),
      m_pendingScore(0), 
      m_pendingAgentType(AgentType::HUMAN), 
      m_pendingEpisode(0),
      m_selectedStatsButton(0) {
    sf::Texture imageTexture; // Create empty texture  
    m_imageViewer = std::make_unique<ImageViewer>(imageTexture);    
    // sf::Text members will be initialized in initialize() after font loads
}

void Leaderboard::initialize(sf::RenderWindow& window) {
    std::vector<std::string> fontPaths = {
        "assets/fonts/ARIAL.TTF", "assets/fonts/arial.ttf", 
        "assets/fonts/ArialCE.ttf", "assets/fonts/Roboto.ttf"
    };
    
    for (const auto& path : fontPaths) {
        if (m_font.openFromFile(path)) {
            spdlog::info("Leaderboard: Font loaded from: {}", path);
            break;
        }
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Light background
    m_background.setSize(sf::Vector2f(windowSize.x, windowSize.y));
    m_background.setFillColor(sf::Color(245, 245, 220));
    
    // Title
    m_title = std::make_unique<sf::Text>(m_font);
    m_title->setString("LEADERBOARD & STATISTICS");
    m_title->setCharacterSize(42);
    m_title->setFillColor(sf::Color(255, 140, 0));
    m_title->setPosition(sf::Vector2f(windowSize.x / 2.0f - 250.0f, 20.0f));
    
    // Section title
    m_sectionTitle = std::make_unique<sf::Text>(m_font);
    m_sectionTitle->setCharacterSize(24);
    m_sectionTitle->setFillColor(sf::Color(70, 130, 180));
    m_sectionTitle->setPosition(sf::Vector2f(50.0f, 80.0f));
    
    // Instructions
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setCharacterSize(18);
    m_instructions->setFillColor(sf::Color(47, 79, 47));
    m_instructions->setPosition(sf::Vector2f(50.0f, windowSize.y - 80.0f));
    
    // Name input elements
    m_namePrompt = std::make_unique<sf::Text>(m_font);
    m_namePrompt->setString("Enter your name (ENTER to confirm):");
    m_namePrompt->setCharacterSize(24);
    m_namePrompt->setFillColor(sf::Color(47, 79, 47));
    m_namePrompt->setPosition(sf::Vector2f(windowSize.x / 2.0f - 250.0f, windowSize.y / 2.0f - 100.0f));
    
    m_inputText = std::make_unique<sf::Text>(m_font);
    m_inputText->setCharacterSize(32);
    m_inputText->setFillColor(sf::Color(70, 130, 180));
    m_inputText->setPosition(sf::Vector2f(windowSize.x / 2.0f - 150.0f, windowSize.y / 2.0f - 50.0f));
    
    m_inputBox.setSize(sf::Vector2f(300.0f, 50.0f));
    m_inputBox.setPosition(sf::Vector2f(windowSize.x / 2.0f - 150.0f, windowSize.y / 2.0f - 60.0f));
    m_inputBox.setFillColor(sf::Color(255, 255, 240));
    m_inputBox.setOutlineThickness(3.0f);
    m_inputBox.setOutlineColor(sf::Color(70, 130, 180));
    
    // Initialize components
    m_imageViewer->initialize(window);
    loadModelPerformanceData();
    loadLeaderboard();
    initializeAnalysisImages();
    updateDisplay();
}

void Leaderboard::loadModelPerformanceData() {
    TrainedModelManager manager;
    auto leaderboardData = manager.getLeaderboardData();
    
    // Add model entries to separate list
    for (const auto& [modelName, bestScore] : leaderboardData) {
        LeaderboardEntry entry;
        entry.playerName = modelName;
        entry.agentType = AgentType::Q_LEARNING; // Will be set properly based on model type
        entry.score = static_cast<int>(bestScore);
        entry.episode = 1; // Models show best single performance
        entry.timestamp = std::chrono::system_clock::now();
        entry.efficiency = bestScore; // Best score efficiency
        
        // Set correct agent type based on model name
        if (modelName.find("DQN") != std::string::npos) {
            entry.agentType = AgentType::DEEP_Q_NETWORK;
        } else if (modelName.find("PPO") != std::string::npos) {
            entry.agentType = AgentType::PPO;
        } else if (modelName.find("Actor-Critic") != std::string::npos) {
            entry.agentType = AgentType::ACTOR_CRITIC;
        }
        
        m_modelEntries.push_back(entry);
    }
    
    spdlog::info("Leaderboard: Loaded {} model performance entries", m_modelEntries.size());
}


void Leaderboard::handleEvent(const sf::Event& event) {
    if (m_state == LeaderboardState::IMAGE_VIEWING) {
        m_imageViewer->handleEvent(event);
        if (!m_imageViewer->isVisible()) {
            m_state = LeaderboardState::MODEL_STATS;
        }
        return;
    }
    
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if (keyPressed->code == sf::Keyboard::Key::Escape) {
            if (m_state == LeaderboardState::ENTERING_NAME) {
                finalizeName();
            } else if (m_state == LeaderboardState::MODEL_STATS) {
                m_state = LeaderboardState::VIEWING;
                m_currentSection = StatsSection::MAIN_LEADERBOARD;
                updateDisplay();
            } else if (m_backCallback) {
                m_backCallback();
            }
        } else if (keyPressed->code == sf::Keyboard::Key::Tab) {
            // Switch between sections
            if (m_state == LeaderboardState::VIEWING) {
                switch (m_currentSection) {
                    case StatsSection::MAIN_LEADERBOARD:
                        m_currentSection = StatsSection::MODEL_PERFORMANCE;
                        break;
                    case StatsSection::MODEL_PERFORMANCE:
                        m_currentSection = StatsSection::ANALYSIS_CHARTS;
                        m_state = LeaderboardState::MODEL_STATS;
                        break;
                    case StatsSection::ANALYSIS_CHARTS:
                        m_currentSection = StatsSection::MAIN_LEADERBOARD;
                        m_state = LeaderboardState::VIEWING;
                        break;
                }
                updateDisplay();
            }
        } else if (m_state == LeaderboardState::ENTERING_NAME) {
            if (keyPressed->code == sf::Keyboard::Key::Enter) {
                finalizeName();
            } else if (keyPressed->code == sf::Keyboard::Key::Backspace && !m_currentInput.empty()) {
                m_currentInput.pop_back();
                m_inputText->setString(m_currentInput + "_");
            }
        } else if (m_state == LeaderboardState::MODEL_STATS) {
            handleStatsNavigation(event);
        }
    }
    
    if (auto* textEntered = event.getIf<sf::Event::TextEntered>()) {
        if (m_state == LeaderboardState::ENTERING_NAME) {
            handleNameInput(textEntered->unicode);
        }
    }
    
    if (auto* mousePressed = event.getIf<sf::Event::MouseButtonPressed>()) {
        if (m_state == LeaderboardState::MODEL_STATS && mousePressed->button == sf::Mouse::Button::Left) {
            sf::Vector2f mousePos(mousePressed->position.x, mousePressed->position.y);
            
            for (size_t i = 0; i < m_analysisImages.size(); ++i) {
                if (m_analysisImages[i].button.getGlobalBounds().contains(mousePos)) {
                    m_imageViewer->loadImage(m_analysisImages[i].path, m_analysisImages[i].name);
                    m_state = LeaderboardState::IMAGE_VIEWING;
                    break;
                }
            }
        }
    }
}


void Leaderboard::handleStatsNavigation(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        switch (keyPressed->code) {
            case sf::Keyboard::Key::Up:
                m_selectedStatsButton = (m_selectedStatsButton - 1 + m_analysisImages.size()) % m_analysisImages.size();
                updateSectionDisplay();
                break;
            case sf::Keyboard::Key::Down:
                m_selectedStatsButton = (m_selectedStatsButton + 1) % m_analysisImages.size();
                updateSectionDisplay();
                break;
            case sf::Keyboard::Key::Enter:
                if (m_selectedStatsButton < m_analysisImages.size()) {
                    m_imageViewer->loadImage(m_analysisImages[m_selectedStatsButton].path, 
                                           m_analysisImages[m_selectedStatsButton].name);
                    m_state = LeaderboardState::IMAGE_VIEWING;
                }
                break;
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
    
    switch (m_state) {
        case LeaderboardState::VIEWING:
            if (m_currentSection == StatsSection::MAIN_LEADERBOARD) {
                renderMainLeaderboard(window);
            } else if (m_currentSection == StatsSection::MODEL_PERFORMANCE) {
                renderModelPerformance(window);
            }
            break;
        case LeaderboardState::MODEL_STATS:
            renderAnalysisCharts(window);
            break;
        case LeaderboardState::ENTERING_NAME:
            if (m_namePrompt) window.draw(*m_namePrompt);
            window.draw(m_inputBox);
            if (m_inputText) window.draw(*m_inputText);
            break;
        case LeaderboardState::IMAGE_VIEWING:
            renderMainLeaderboard(window); // Show leaderboard in background
            m_imageViewer->render(window);
            break;
    }
    
    if (m_instructions) window.draw(*m_instructions);
}

void Leaderboard::renderMainLeaderboard(sf::RenderWindow& window) {
    m_sectionTitle->setString("Player Leaderboard");
    window.draw(*m_sectionTitle);
    
    for (const auto& text : m_entryTexts) {
        if (text) window.draw(*text);
    }
    
    m_instructions->setString("TAB: Switch Sections | ESC: Back | F1: Quick Access");
}

// In renderModelPerformance, fix sf::Text constructors:
void Leaderboard::renderModelPerformance(sf::RenderWindow& window) {
    m_sectionTitle->setString("Model Performance Leaderboard");
    window.draw(*m_sectionTitle);
    
    // Display model entries
    float startY = 120.0f;
    
    auto header = std::make_unique<sf::Text>(m_font); // FIXED CONSTRUCTOR
    header->setString("Rank  Model Name                   Best Score  Type");
    header->setCharacterSize(18);
    header->setFillColor(sf::Color(47, 79, 47));
    header->setPosition(sf::Vector2f(50.0f, startY));
    window.draw(*header);
    
    for (size_t i = 0; i < m_modelEntries.size() && i < 10; ++i) {
        const auto& entry = m_modelEntries[i];
        
        std::string line = "#" + std::to_string(i + 1);
        line.resize(6, ' ');
        line += entry.playerName;
        line.resize(35, ' ');
        line += std::to_string(entry.score);
        line.resize(47, ' ');
        line += entry.getAgentTypeString();
        
        auto entryText = std::make_unique<sf::Text>(m_font); // FIXED CONSTRUCTOR
        entryText->setString(line);
        entryText->setCharacterSize(16);
        
        sf::Color color = sf::Color(47, 79, 47);
        if (i == 0) color = sf::Color(255, 215, 0);
        else if (i == 1) color = sf::Color(192, 192, 192);
        else if (i == 2) color = sf::Color(205, 127, 50);
        
        entryText->setFillColor(color);
        entryText->setPosition(sf::Vector2f(50.0f, startY + 30.0f + i * 25.0f));
        
        window.draw(*entryText);
    }
    
    m_instructions->setString("TAB: Switch Sections | ESC: Back");
}

// In renderAnalysisCharts, fix sf::Text constructor:
void Leaderboard::renderAnalysisCharts(sf::RenderWindow& window) {
    m_sectionTitle->setString("Model Analysis Charts");
    window.draw(*m_sectionTitle);
    
    // Render analysis image buttons
    for (size_t i = 0; i < m_analysisImages.size(); ++i) {
        const auto& image = m_analysisImages[i];
        
        // Highlight selected button
        sf::RectangleShape button = image.button;
        if (i == m_selectedStatsButton) {
            button.setFillColor(sf::Color(255, 255, 0));
            button.setOutlineThickness(3.0f);
            button.setOutlineColor(sf::Color(255, 140, 0));
        }
        
        window.draw(button);
        if (image.buttonText) window.draw(*image.buttonText);
    }
    
    // Description panel
    if (m_selectedStatsButton < m_analysisImages.size()) {
        sf::RectangleShape descPanel(sf::Vector2f(400.0f, 200.0f));
        descPanel.setPosition(sf::Vector2f(400.0f, 200.0f));
        descPanel.setFillColor(sf::Color(240, 248, 255, 200));
        descPanel.setOutlineThickness(2.0f);
        descPanel.setOutlineColor(sf::Color(70, 130, 180));
        window.draw(descPanel);
        
        sf::Text descText(m_font); // FIXED CONSTRUCTOR
        descText.setString(m_analysisImages[m_selectedStatsButton].description);
        descText.setCharacterSize(16);
        descText.setFillColor(sf::Color(47, 79, 47));
        descText.setPosition(sf::Vector2f(420.0f, 220.0f));
        window.draw(descText);
    }
    
    m_instructions->setString("UP/DOWN: Navigate | ENTER: View Chart | ESC: Back");
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
        case AgentType::PPO: return "PPO-Agent";
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

void Leaderboard::updateDisplay() {
    m_entryTexts.clear();
    
    float startY = 120.0f;
    
    // Combine user and model entries for main leaderboard
    std::vector<LeaderboardEntry> combinedEntries = m_entries;
    combinedEntries.insert(combinedEntries.end(), m_modelEntries.begin(), m_modelEntries.end());
    
    std::sort(combinedEntries.begin(), combinedEntries.end(), 
              [](const LeaderboardEntry& a, const LeaderboardEntry& b) {
                  if (a.score != b.score) return a.score > b.score;
                  return a.efficiency > b.efficiency;
              });
    
    if (m_currentSection == StatsSection::MAIN_LEADERBOARD) {
        auto header = std::make_unique<sf::Text>(m_font);
        header->setString("Rank  Player/Model Name            Score  Type        Efficiency");
        header->setCharacterSize(18);
        header->setFillColor(sf::Color(47, 79, 47));
        header->setPosition(sf::Vector2f(50.0f, startY));
        m_entryTexts.push_back(std::move(header));
        
        for (size_t i = 0; i < combinedEntries.size() && i < MAX_ENTRIES; ++i) {
            const auto& entry = combinedEntries[i];
            
            std::string rankStr = "#" + std::to_string(i + 1);
            std::string nameStr = entry.playerName;
            if (nameStr.length() > 20) nameStr = nameStr.substr(0, 17) + "...";
            
            std::string line = rankStr;
            line.resize(6, ' ');
            line += nameStr;
            line.resize(35, ' ');
            line += std::to_string(entry.score);
            line.resize(42, ' ');
            line += entry.getAgentTypeString();
            line.resize(54, ' ');
            line += std::to_string(entry.efficiency).substr(0, 5);
            
            auto entryText = std::make_unique<sf::Text>(m_font);
            entryText->setString(line);
            entryText->setCharacterSize(16);
            
            sf::Color color = sf::Color(47, 79, 47);
            if (i == 0) color = sf::Color(255, 215, 0);
            else if (i == 1) color = sf::Color(192, 192, 192);
            else if (i == 2) color = sf::Color(205, 127, 50);
            
            entryText->setFillColor(color);
            entryText->setPosition(sf::Vector2f(50.0f, startY + 40.0f + i * 25.0f));
            
            m_entryTexts.push_back(std::move(entryText));
        }
    }
    
    if (combinedEntries.empty()) {
        auto emptyText = std::make_unique<sf::Text>(m_font);
        emptyText->setString("No scores yet! Be the first to play!");
        emptyText->setCharacterSize(24);
        emptyText->setFillColor(sf::Color(70, 130, 180));
        emptyText->setPosition(sf::Vector2f(400.0f, 300.0f));
        m_entryTexts.push_back(std::move(emptyText));
    }
}

void Leaderboard::updateSectionDisplay() {
    // Update button highlighting for stats section
    for (size_t i = 0; i < m_analysisImages.size(); ++i) {
        if (i == m_selectedStatsButton) {
            m_analysisImages[i].button.setFillColor(sf::Color(255, 255, 0));
        } else {
            m_analysisImages[i].button.setFillColor(sf::Color(173, 216, 230));
        }
    }
}

void Leaderboard::initializeAnalysisImages() {
    float buttonY = 150.0f;
    float buttonHeight = 60.0f;
    float buttonSpacing = 80.0f;
    
    std::vector<std::tuple<std::string, std::string, std::string>> imageData = {
        {"Q-Learning Analysis", "models/analysis_qtable_balanced.png", "Performance analysis for Q-Learning balanced model"},
        {"DQN Analysis", "models/analysis_dqn_balanced_best_fixed.png", "Performance analysis for DQN balanced model"},
        {"PPO Analysis", "models/analysis_ppo_balanced_best_fixed.png", "Performance analysis for PPO balanced model"},
        {"Actor-Critic Analysis", "models/analysis_ac_balanced_best_fixed.png", "Performance analysis for Actor-Critic balanced model"},
        {"Model Comparison", "models/enhanced_comparison_fixed.png", "Comprehensive comparison of all trained models"}
    };
    
    for (size_t i = 0; i < imageData.size(); ++i) {
        AnalysisImage image;
        image.name = std::get<0>(imageData[i]);
        image.path = std::get<1>(imageData[i]);
        image.description = std::get<2>(imageData[i]);
        
        // Button setup
        image.button.setSize(sf::Vector2f(300.0f, buttonHeight));
        image.button.setPosition(sf::Vector2f(50.0f, buttonY + i * buttonSpacing));
        image.button.setFillColor(sf::Color(173, 216, 230));
        image.button.setOutlineThickness(2.0f);
        image.button.setOutlineColor(sf::Color(70, 130, 180));
        
        // Button text
        image.buttonText = std::make_unique<sf::Text>(m_font);
        image.buttonText->setString(image.name);
        image.buttonText->setCharacterSize(18);
        image.buttonText->setFillColor(sf::Color(47, 79, 47));
        image.buttonText->setPosition(sf::Vector2f(60.0f, buttonY + i * buttonSpacing + 15.0f));
        
        m_analysisImages.push_back(std::move(image));
    }
}