// Enhanced StatsGallery.hpp with full screen and download features

#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <functional>
#include <memory>
#include "MLAgents.hpp"

// Enhanced ImageViewer with full screen and download
class ImageViewer {
public:
    ImageViewer();
    void initialize(sf::RenderWindow& window);
    void loadImage(const std::string& imagePath, const std::string& title);
    void handleEvent(const sf::Event& event);
    void render(sf::RenderWindow& window);
    bool isVisible() const { return m_visible; }
    void close() { m_visible = false; }

private:
    sf::Font m_font;
    sf::Texture m_imageTexture;
    sf::Sprite m_imageSprite;
    std::unique_ptr<sf::Text> m_titleText;
    std::unique_ptr<sf::Text> m_instructionText;
    std::unique_ptr<sf::Text> m_downloadText;        // NEW: Download button text
    sf::RectangleShape m_background;
    sf::RectangleShape m_imageFrame;
    sf::RectangleShape m_downloadButton;             // NEW: Download button
    bool m_visible;
    std::string m_currentTitle;
    std::string m_currentImagePath;                  // NEW: For download functionality
    
    void downloadImage();                            // NEW: Download functionality
};

class StatsGallery {
public:
    StatsGallery();
    void initialize(sf::RenderWindow& window);
    void handleEvent(const sf::Event& event);
    void update();
    void render(sf::RenderWindow& window);
    
    void setBackCallback(std::function<void()> callback) { m_backCallback = callback; }

private:
    enum class GalleryState {
        BROWSING,
        IMAGE_VIEWING
    };
    
    struct AnalysisImage {
        std::string name;
        std::string path;
        std::string description;
        sf::RectangleShape button;
        std::unique_ptr<sf::Text> buttonText;
    };
    
    GalleryState m_state;
    sf::Font m_font;
    int m_selectedButton;
    
    // UI Elements - Yellow theme
    std::unique_ptr<sf::Text> m_title;
    std::unique_ptr<sf::Text> m_instructions;
    std::unique_ptr<sf::Text> m_sectionTitle;
    sf::RectangleShape m_background;
    sf::RectangleShape m_titlePanel;         // NEW: Title panel
    sf::RectangleShape m_contentPanel;       // NEW: Content panel
    
    // Analysis images
    std::vector<AnalysisImage> m_analysisImages;
    std::unique_ptr<ImageViewer> m_imageViewer;
    
    // Model data
    TrainedModelManager m_modelManager;
    EvaluationReportData m_evaluationData;
    
    std::function<void()> m_backCallback;
    
    void initializeAnalysisImages();
    void handleNavigation(const sf::Event& event);
    void updateButtonSelection();
    void renderAnalysisCharts(sf::RenderWindow& window);
    void renderModelStats(sf::RenderWindow& window);
};