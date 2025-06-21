#include "StatsGallery.hpp"
#include <spdlog/spdlog.h>
#include <filesystem>

ImageViewer::ImageViewer() : m_imageSprite(m_imageTexture), m_visible(false) {}
void ImageViewer::initialize(sf::RenderWindow& window) {
    std::vector<std::string> fontPaths = {
        "assets/fonts/ARIAL.TTF", "assets/fonts/arial.ttf", 
        "assets/fonts/ArialCE.ttf", "assets/fonts/Roboto.ttf"
    };
    
    for (const auto& path : fontPaths) {
        if (m_font.openFromFile(path)) break;
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Full screen background
    m_background.setSize(sf::Vector2f(windowSize.x, windowSize.y));
    m_background.setFillColor(sf::Color(0, 0, 0, 240));
    
    // Full screen frame with small margins
    m_imageFrame.setSize(sf::Vector2f(windowSize.x - 60, windowSize.y - 120));
    m_imageFrame.setPosition(sf::Vector2f(30, 60));
    m_imageFrame.setFillColor(sf::Color::White);
    m_imageFrame.setOutlineThickness(3.0f);
    m_imageFrame.setOutlineColor(sf::Color(218, 165, 32));
    
    // Title text
    m_titleText = std::make_unique<sf::Text>(m_font);
    m_titleText->setCharacterSize(24);
    m_titleText->setFillColor(sf::Color::White);
    m_titleText->setStyle(sf::Text::Bold);
    
    // Download button
    m_downloadButton.setSize(sf::Vector2f(120, 40));
    m_downloadButton.setPosition(sf::Vector2f(windowSize.x - 150, 15));
    m_downloadButton.setFillColor(sf::Color(255, 215, 0));
    m_downloadButton.setOutlineThickness(2.0f);
    m_downloadButton.setOutlineColor(sf::Color(218, 165, 32));
    
    m_downloadText = std::make_unique<sf::Text>(m_font);
    m_downloadText->setString("Download");
    m_downloadText->setCharacterSize(16);
    m_downloadText->setFillColor(sf::Color(139, 69, 19));
    m_downloadText->setStyle(sf::Text::Bold);
    m_downloadText->setPosition(sf::Vector2f(windowSize.x - 135, 25));
    
    // Instruction text
    m_instructionText = std::make_unique<sf::Text>(m_font);
    m_instructionText->setString("ESC: Close | CLICK DOWNLOAD: Save Image");
    m_instructionText->setCharacterSize(16);
    m_instructionText->setFillColor(sf::Color(200, 200, 200));
    m_instructionText->setPosition(sf::Vector2f(30, windowSize.y - 40));
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

    // Recreate sprite with loaded texture
    m_imageSprite = sf::Sprite(m_imageTexture);
    m_currentTitle = title;
    m_currentImagePath = imagePath;
    m_titleText->setString(title);
    
    // Get actual window size from frame
    sf::Vector2f frameSize = m_imageFrame.getSize();
    sf::Vector2f framePos = m_imageFrame.getPosition();
    sf::Vector2u imageSize = m_imageTexture.getSize();
    
    // Scale to fit frame with padding
    float frameWidth = frameSize.x - 40;  // 20px padding each side
    float frameHeight = frameSize.y - 40; // 20px padding top/bottom
    
    float scaleX = frameWidth / imageSize.x;
    float scaleY = frameHeight / imageSize.y;
    float scale = std::min(scaleX, scaleY);
    
    m_imageSprite.setScale(sf::Vector2f(scale, scale));
    
    // Center image in frame
    sf::Vector2f scaledSize(imageSize.x * scale, imageSize.y * scale);
    float offsetX = (frameWidth - scaledSize.x) / 2;
    float offsetY = (frameHeight - scaledSize.y) / 2;
    
    m_imageSprite.setPosition(sf::Vector2f(framePos.x + 20 + offsetX, framePos.y + 20 + offsetY));
    
    // Position title at actual window center
    sf::Vector2f windowSize = sf::Vector2f(1200, 800); // Or get from somewhere
    auto titleBounds = m_titleText->getLocalBounds();
    m_titleText->setPosition(sf::Vector2f((windowSize.x - titleBounds.size.x) / 2.0f, 20));
    
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
        
        // Check download button click
        if (m_downloadButton.getGlobalBounds().contains(mousePos)) {
            downloadImage();
        } else if (!m_imageFrame.getGlobalBounds().contains(mousePos)) {
            close();
        }
    }
}

void ImageViewer::downloadImage() {
    if (m_currentImagePath.empty()) return;
    
    try {
        // Simple file copy to Downloads or current directory
        std::filesystem::path sourcePath(m_currentImagePath);
        std::string filename = sourcePath.filename().string();
        std::string destPath = "downloaded_" + filename;
        
        std::filesystem::copy_file(sourcePath, destPath, std::filesystem::copy_options::overwrite_existing);
        spdlog::info("ImageViewer: Image downloaded to: {}", destPath);
    } catch (const std::exception& e) {
        spdlog::error("ImageViewer: Download failed: {}", e.what());
    }
}

void ImageViewer::render(sf::RenderWindow& window) {
    if (!m_visible) return;
    
    window.draw(m_background);
    window.draw(m_imageFrame);
    window.draw(m_downloadButton);
    if (m_titleText) window.draw(*m_titleText);
    if (m_downloadText) window.draw(*m_downloadText);
    window.draw(m_imageSprite);
    if (m_instructionText) window.draw(*m_instructionText);
}

// Enhanced StatsGallery Implementation
StatsGallery::StatsGallery() : m_state(GalleryState::BROWSING), m_selectedButton(0) {
    m_imageViewer = std::make_unique<ImageViewer>();
}

void StatsGallery::initialize(sf::RenderWindow& window) {
    std::vector<std::string> fontPaths = {
        "assets/fonts/ARIAL.TTF", "assets/fonts/arial.ttf", 
        "assets/fonts/ArialCE.ttf", "assets/fonts/Roboto.ttf"
    };
    
    for (const auto& path : fontPaths) {
        if (m_font.openFromFile(path)) break;
    }
    
    sf::Vector2u windowSize = window.getSize();
    
    // Yellow theme background
    m_background.setSize(sf::Vector2f(windowSize.x, windowSize.y));
    m_background.setFillColor(sf::Color(255, 253, 208));
    
    // Title panel
    m_titlePanel.setSize(sf::Vector2f(windowSize.x - 100.0f, 100.0f));
    m_titlePanel.setPosition(sf::Vector2f(50.0f, 20.0f));
    m_titlePanel.setFillColor(sf::Color(255, 248, 220, 150));
    m_titlePanel.setOutlineThickness(2.0f);
    m_titlePanel.setOutlineColor(sf::Color(218, 165, 32, 100));
    
    // Content panel
    m_contentPanel.setSize(sf::Vector2f(windowSize.x - 100.0f, windowSize.y - 200.0f));
    m_contentPanel.setPosition(sf::Vector2f(50.0f, 140.0f));
    m_contentPanel.setFillColor(sf::Color(255, 250, 205, 180));
    m_contentPanel.setOutlineThickness(2.0f);
    m_contentPanel.setOutlineColor(sf::Color(218, 165, 32, 150));
    
    // Title
    m_title = std::make_unique<sf::Text>(m_font);
    m_title->setString("STATS & ANALYSIS");
    m_title->setCharacterSize(48);
    m_title->setFillColor(sf::Color(139, 69, 19));
    m_title->setStyle(sf::Text::Bold);
    
    auto titleBounds = m_title->getLocalBounds();
    m_title->setPosition(sf::Vector2f((windowSize.x - titleBounds.size.x) / 2.0f, 35.0f));
    
    // Section title with "Enter to display" hint
    m_sectionTitle = std::make_unique<sf::Text>(m_font);
    m_sectionTitle->setString("Model Performance Analysis - Press ENTER to display charts");
    m_sectionTitle->setCharacterSize(20);
    m_sectionTitle->setFillColor(sf::Color(160, 82, 45));
    m_sectionTitle->setPosition(sf::Vector2f(70.0f, 100.0f));
    
    // Instructions
    m_instructions = std::make_unique<sf::Text>(m_font);
    m_instructions->setCharacterSize(18);
    m_instructions->setFillColor(sf::Color(101, 67, 33));
    m_instructions->setPosition(sf::Vector2f(70.0f, windowSize.y - 60.0f));
    
    m_imageViewer->initialize(window);
    m_evaluationData = m_modelManager.getEvaluationData();
    initializeAnalysisImages();
    updateButtonSelection();
}

void StatsGallery::initializeAnalysisImages() {
    float buttonY = 170.0f;
    float buttonHeight = 50.0f;
    float buttonSpacing = 65.0f;
    
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
        
        // Yellow theme buttons
        image.button.setSize(sf::Vector2f(400.0f, buttonHeight));
        image.button.setPosition(sf::Vector2f(80.0f, buttonY + i * buttonSpacing));
        image.button.setFillColor(sf::Color(255, 248, 220, 240));
        image.button.setOutlineThickness(2.0f);
        image.button.setOutlineColor(sf::Color(218, 165, 32));
        
        // Button text
        image.buttonText = std::make_unique<sf::Text>(m_font);
        image.buttonText->setString(image.name);
        image.buttonText->setCharacterSize(18);
        image.buttonText->setFillColor(sf::Color(139, 69, 19));
        image.buttonText->setStyle(sf::Text::Bold);
        image.buttonText->setPosition(sf::Vector2f(100.0f, buttonY + i * buttonSpacing + 12.0f));
        
        m_analysisImages.push_back(std::move(image));
    }
}

void StatsGallery::updateButtonSelection() {
    for (size_t i = 0; i < m_analysisImages.size(); ++i) {
        if (i == m_selectedButton) {
            m_analysisImages[i].button.setFillColor(sf::Color(255, 215, 0, 250));
            m_analysisImages[i].button.setOutlineThickness(3.0f);
            m_analysisImages[i].button.setOutlineColor(sf::Color(255, 140, 0));
        } else {
            m_analysisImages[i].button.setFillColor(sf::Color(255, 248, 220, 240));
            m_analysisImages[i].button.setOutlineThickness(2.0f);
            m_analysisImages[i].button.setOutlineColor(sf::Color(218, 165, 32));
        }
    }
}

void StatsGallery::renderAnalysisCharts(sf::RenderWindow& window) {
    for (const auto& image : m_analysisImages) {
        window.draw(image.button);
        if (image.buttonText) window.draw(*image.buttonText);
    }
    
    // Description panel
    if (m_selectedButton < m_analysisImages.size()) {
        sf::RectangleShape descPanel(sf::Vector2f(450.0f, 150.0f));
        descPanel.setPosition(sf::Vector2f(520.0f, 200.0f));
        descPanel.setFillColor(sf::Color(255, 248, 220, 200));
        descPanel.setOutlineThickness(2.0f);
        descPanel.setOutlineColor(sf::Color(218, 165, 32));
        window.draw(descPanel);
        
        sf::Text descText(m_font);
        descText.setString(m_analysisImages[m_selectedButton].description);
        descText.setCharacterSize(16);
        descText.setFillColor(sf::Color(101, 67, 33));
        descText.setPosition(sf::Vector2f(540.0f, 220.0f));
        window.draw(descText);
        
        // "Press ENTER" hint
        sf::Text enterHint(m_font);
        enterHint.setString("Press ENTER to view full screen");
        enterHint.setCharacterSize(16);
        enterHint.setFillColor(sf::Color(160, 82, 45));
        enterHint.setStyle(sf::Text::Bold);
        enterHint.setPosition(sf::Vector2f(540.0f, 310.0f));
        window.draw(enterHint);
    }
    
    renderModelStats(window);
    
    m_instructions->setString("UP/DOWN: Navigate | ENTER: View Full Screen | ESC: Back to Menu");
}

void StatsGallery::render(sf::RenderWindow& window) {
    window.draw(m_background);
    window.draw(m_titlePanel);
    window.draw(m_contentPanel);
    window.draw(*m_title);
    window.draw(*m_sectionTitle);
    
    switch (m_state) {
        case GalleryState::BROWSING:
            renderAnalysisCharts(window);
            break;
        case GalleryState::IMAGE_VIEWING:
            renderAnalysisCharts(window); // Background
            m_imageViewer->render(window);
            break;
    }
    
    window.draw(*m_instructions);
}

// Rest of methods remain the same...
void StatsGallery::handleEvent(const sf::Event& event) {
    if (m_state == GalleryState::IMAGE_VIEWING) {
        m_imageViewer->handleEvent(event);
        if (!m_imageViewer->isVisible()) {
            m_state = GalleryState::BROWSING;
        }
        return;
    }
    
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if (keyPressed->code == sf::Keyboard::Key::Escape) {
            if (m_backCallback) m_backCallback();
        } else {
            handleNavigation(event);
        }
    }
}

void StatsGallery::handleNavigation(const sf::Event& event) {
    if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        switch (keyPressed->code) {
            case sf::Keyboard::Key::Up:
                m_selectedButton = (m_selectedButton - 1 + m_analysisImages.size()) % m_analysisImages.size();
                updateButtonSelection();
                break;
            case sf::Keyboard::Key::Down:
                m_selectedButton = (m_selectedButton + 1) % m_analysisImages.size();
                updateButtonSelection();
                break;
            case sf::Keyboard::Key::Enter:
                if (m_selectedButton < m_analysisImages.size()) {
                    m_imageViewer->loadImage(m_analysisImages[m_selectedButton].path, 
                                           m_analysisImages[m_selectedButton].name);
                    m_state = GalleryState::IMAGE_VIEWING;
                }
                break;
        }
    }
}

void StatsGallery::update() {}

void StatsGallery::renderModelStats(sf::RenderWindow& window) {
    sf::Text statsTitle(m_font);
    statsTitle.setString("Model Performance Summary");
    statsTitle.setCharacterSize(18);
    statsTitle.setFillColor(sf::Color(160, 82, 45));
    statsTitle.setStyle(sf::Text::Bold);
    statsTitle.setPosition(sf::Vector2f(520.0f, 400.0f));
    window.draw(statsTitle);
    
    std::string summary = m_modelManager.getPerformanceSummary();
    if (!summary.empty()) {
        sf::Text summaryText(m_font);
        summaryText.setString(summary);
        summaryText.setCharacterSize(14);
        summaryText.setFillColor(sf::Color(101, 67, 33));
        summaryText.setPosition(sf::Vector2f(520.0f, 430.0f));
        window.draw(summaryText);
    }
}