#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <memory>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

// Include project headers
#include "MLAgents.hpp"
#include "TorchInference.hpp"
#include "GameState.hpp"

class ModelDebugger {
private:
    std::string m_modelsDirectory;
    
public:
    ModelDebugger(const std::string& modelsDir = "models/") : m_modelsDirectory(modelsDir) {
        setupLogging();
    }
    
    void setupLogging() {
        // Set up comprehensive logging
        spdlog::set_level(spdlog::level::debug);
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
        
        // Create console and file sinks
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/model_debug.log", true);
        
        std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
        auto logger = std::make_shared<spdlog::logger>("ModelDebugger", sinks.begin(), sinks.end());
        spdlog::set_default_logger(logger);
        
        spdlog::info("ModelDebugger: Logging initialized");
    }
    
    void printHeader() {
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "             SnakeAI-MLOps Model Debugger v2.0\n";
        std::cout << "             LibTorch Integration Testing Tool\n";
        std::cout << "================================================================\n";
        std::cout << "\n";
    }
    
    void scanModelsDirectory() {
        std::cout << "🔍 Scanning models directory: " << m_modelsDirectory << "\n\n";
        
        if (!std::filesystem::exists(m_modelsDirectory)) {
            std::cout << "❌ Models directory does not exist!\n";
            std::cout << "💡 Create the directory and add your trained models:\n";
            std::cout << "   mkdir -p models/{qlearning,dqn,ppo,actor_critic}\n";
            return;
        }
        
        std::vector<std::filesystem::path> foundModels;
        
        // Recursively scan for model files
        for (const auto& entry : std::filesystem::recursive_directory_iterator(m_modelsDirectory)) {
            if (entry.is_regular_file()) {
                const auto& path = entry.path();
                std::string extension = path.extension().string();
                
                if (extension == ".json" || extension == ".pth" || extension == ".pt") {
                    foundModels.push_back(path);
                }
            }
        }
        
        if (foundModels.empty()) {
            std::cout << "❌ No model files found in " << m_modelsDirectory << "\n";
            std::cout << "💡 Expected file types: .json (Q-Learning), .pth/.pt (Neural Networks)\n";
            return;
        }
        
        std::cout << "📁 Found " << foundModels.size() << " model files:\n\n";
        
        // Group by type
        std::vector<std::filesystem::path> qlearningModels, neuralModels;
        
        for (const auto& model : foundModels) {
            if (model.extension() == ".json") {
                qlearningModels.push_back(model);
            } else {
                neuralModels.push_back(model);
            }
        }
        
        // Display Q-Learning models
        if (!qlearningModels.empty()) {
            std::cout << "🧠 Q-Learning Models (.json):\n";
            for (const auto& model : qlearningModels) {
                auto size = std::filesystem::file_size(model);
                std::cout << "  ✅ " << model.string() << " (" << size << " bytes)\n";
            }
            std::cout << "\n";
        }
        
        // Display Neural Network models
        if (!neuralModels.empty()) {
            std::cout << "🤖 Neural Network Models (.pth/.pt):\n";
            for (const auto& model : neuralModels) {
                auto size = std::filesystem::file_size(model);
                std::string sizeStr = formatFileSize(size);
                std::cout << "  📦 " << model.string() << " (" << sizeStr << ")\n";
            }
            std::cout << "\n";
        }
    }
    
    void testLibTorchInstallation() {
        std::cout << "🔧 Testing LibTorch Installation:\n\n";
        
        try {
            // Test basic LibTorch functionality
            torch::Tensor tensor = torch::ones({2, 3});
            std::cout << "  ✅ Basic tensor creation: SUCCESS\n";
            
            // Test tensor operations
            torch::Tensor result = tensor * 2;
            std::cout << "  ✅ Tensor operations: SUCCESS\n";
            
            // Test device availability
            if (torch::cuda::is_available()) {
                std::cout << "  🚀 CUDA support: AVAILABLE (using CPU for compatibility)\n";
            } else {
                std::cout << "  💻 CUDA support: NOT AVAILABLE (CPU only)\n";
            }
            
            // Test JIT compilation
            try {
                // Create a simple scripted function
                auto simple_module = torch::jit::compile(R"(
                    def forward(input):
                        return input + 1
                )");
                std::cout << "  ✅ JIT compilation: SUCCESS\n";
            } catch (const std::exception& e) {
                std::cout << "  ⚠️  JIT compilation: FAILED (" << e.what() << ")\n";
            }
            
            std::cout << "  ✅ LibTorch installation: FUNCTIONAL\n\n";
            
        } catch (const std::exception& e) {
            std::cout << "  ❌ LibTorch installation: FAILED\n";
            std::cout << "     Error: " << e.what() << "\n\n";
            std::cout << "💡 Troubleshooting:\n";
            std::cout << "   1. Ensure LibTorch is properly installed\n";
            std::cout << "   2. Check CMakeLists.txt includes find_package(Torch REQUIRED)\n";
            std::cout << "   3. Verify LibTorch DLLs are in PATH (Windows)\n\n";
        }
    }
    
    void testTrainedModelManager() {
        std::cout << "📊 Testing TrainedModelManager:\n\n";
        
        try {
            TrainedModelManager manager(m_modelsDirectory);
            auto models = manager.getAvailableModels();
            
            std::cout << "  ✅ TrainedModelManager initialization: SUCCESS\n";
            std::cout << "  📈 Found " << models.size() << " registered models\n\n";
            
            if (models.empty()) {
                std::cout << "  ⚠️  No models registered by TrainedModelManager\n";
                std::cout << "     This might indicate:\n";
                std::cout << "     - Missing evaluation report (enhanced_evaluation_report_fixed.json)\n";
                std::cout << "     - Models not in expected directory structure\n\n";
            } else {
                std::cout << "  📋 Registered Models:\n";
                for (const auto& model : models) {
                    std::cout << "    🔹 " << model.name << " (" << model.modelType << ")\n";
                    std::cout << "       Path: " << model.modelPath << "\n";
                    std::cout << "       Exists: " << (std::filesystem::exists(model.modelPath) ? "✅" : "❌") << "\n";
                    if (model.performance.bestScore > 0) {
                        std::cout << "       Best Score: " << model.performance.bestScore << "\n";
                    }
                    std::cout << "\n";
                }
            }
            
        } catch (const std::exception& e) {
            std::cout << "  ❌ TrainedModelManager: FAILED\n";
            std::cout << "     Error: " << e.what() << "\n\n";
        }
    }
    
    void testModelLoading() {
        std::cout << "🧪 Testing Individual Model Loading:\n\n";
        
        TrainedModelManager manager(m_modelsDirectory);
        auto models = manager.getAvailableModels();
        
        if (models.empty()) {
            std::cout << "  ⚠️  No models available for testing\n\n";
            return;
        }
        
        for (const auto& modelInfo : models) {
            std::cout << "🎯 Testing: " << modelInfo.name << "\n";
            std::cout << "   Type: " << modelInfo.modelType << "\n";
            std::cout << "   Path: " << modelInfo.modelPath << "\n";
            
            if (!std::filesystem::exists(modelInfo.modelPath)) {
                std::cout << "   ❌ File does not exist\n\n";
                continue;
            }
            
            try {
                std::unique_ptr<IAgent> agent;
                
                if (modelInfo.modelType == "qlearning") {
                    agent = std::make_unique<QLearningAgentEnhanced>(modelInfo);
                } else if (modelInfo.modelType == "dqn") {
                    agent = std::make_unique<DQNAgent>(modelInfo);
                } else if (modelInfo.modelType == "ppo") {
                    agent = std::make_unique<PPOAgent>(modelInfo);
                } else if (modelInfo.modelType == "actor_critic") {
                    agent = std::make_unique<ActorCriticAgent>(modelInfo);
                } else {
                    std::cout << "   ⚠️  Unknown model type\n\n";
                    continue;
                }
                
                if (agent) {
                    std::cout << "   ✅ Agent creation: SUCCESS\n";
                    std::cout << "   ℹ️  Agent info: " << agent->getAgentInfo() << "\n";
                    std::cout << "   📄 Model info: " << agent->getModelInfo() << "\n";
                    
                    // Test inference
                    EnhancedState testState = createTestState();
                    Direction action = agent->getAction(testState, false);
                    std::cout << "   🎮 Test inference: SUCCESS (action: " << static_cast<int>(action) << ")\n";
                    
                } else {
                    std::cout << "   ❌ Agent creation: FAILED\n";
                }
                
            } catch (const std::exception& e) {
                std::cout << "   ❌ Exception: " << e.what() << "\n";
            }
            
            std::cout << "\n";
        }
    }
    
    void testLibTorchInference() {
        std::cout << "🚀 Testing LibTorch Inference Directly:\n\n";
        
        // Find neural network models
        std::vector<std::filesystem::path> neuralModels;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(m_modelsDirectory)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".pth" || ext == ".pt") {
                    neuralModels.push_back(entry.path());
                }
            }
        }
        
        if (neuralModels.empty()) {
            std::cout << "  ⚠️  No neural network model files found\n\n";
            return;
        }
        
        for (const auto& modelPath : neuralModels) {
            std::cout << "🎯 Testing direct LibTorch loading: " << modelPath.filename() << "\n";
            
            try {
                TorchInference inference;
                bool loaded = inference.loadModel(modelPath.string());
                
                if (loaded) {
                    std::cout << "   ✅ Model loading: SUCCESS\n";
                    
                    // Run comprehensive tests
                    bool testsPassed = inference.runModelTests();
                    if (testsPassed) {
                        std::cout << "   ✅ Model tests: ALL PASSED\n";
                    } else {
                        std::cout << "   ⚠️  Model tests: SOME FAILED\n";
                    }
                    
                } else {
                    std::cout << "   ❌ Model loading: FAILED\n";
                }
                
            } catch (const std::exception& e) {
                std::cout << "   ❌ Exception: " << e.what() << "\n";
            }
            
            std::cout << "\n";
        }
    }
    
    void generateReport() {
        std::cout << "📝 Generating Comprehensive Report:\n\n";
        
        // System Information
        std::cout << "🖥️  System Information:\n";
        std::cout << "   Operating System: ";
        #ifdef _WIN32
            std::cout << "Windows\n";
        #elif __linux__
            std::cout << "Linux\n";
        #elif __APPLE__
            std::cout << "macOS\n";
        #else
            std::cout << "Unknown\n";
        #endif
        
        std::cout << "   Compiler: ";
        #ifdef _MSC_VER
            std::cout << "MSVC " << _MSC_VER << "\n";
        #elif __GNUC__
            std::cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
        #elif __clang__
            std::cout << "Clang " << __clang_major__ << "." << __clang_minor__ << "\n";
        #else
            std::cout << "Unknown\n";
        #endif
        
        std::cout << "   C++ Standard: " << __cplusplus << "\n";
        std::cout << "\n";
        
        // LibTorch Information
        std::cout << "🔧 LibTorch Information:\n";
        try {
            std::cout << "   Version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << "\n";
            std::cout << "   CUDA Available: " << (torch::cuda::is_available() ? "Yes" : "No") << "\n";
            if (torch::cuda::is_available()) {
                std::cout << "   CUDA Devices: " << torch::cuda::device_count() << "\n";
            }
        } catch (const std::exception& e) {
            std::cout << "   Error getting LibTorch info: " << e.what() << "\n";
        }
        std::cout << "\n";
        
        // File System Analysis
        std::cout << "📁 File System Analysis:\n";
        analyzeFileSystem();
        
        std::cout << "📋 Summary and Recommendations:\n";
        generateRecommendations();
    }
    
private:
    EnhancedState createTestState() {
        EnhancedState state;
        state.basic.dangerStraight = false;
        state.basic.dangerLeft = true;
        state.basic.dangerRight = false;
        state.basic.currentDirection = Direction::RIGHT;
        state.basic.foodLeft = true;
        state.basic.foodRight = false;
        state.basic.foodUp = false;
        state.basic.foodDown = false;
        
        // Fill enhanced state with reasonable values
        state.distanceToFood = 5.0f;
        state.distanceToWall[0] = 3.0f; // up
        state.distanceToWall[1] = 7.0f; // down
        state.distanceToWall[2] = 2.0f; // left
        state.distanceToWall[3] = 8.0f; // right
        state.snakeLength = 3;
        state.emptySpaces = 97;
        state.pathToFood = 5.5f;
        
        return state;
    }
    
    std::string formatFileSize(std::uintmax_t size) {
        if (size < 1024) return std::to_string(size) + " B";
        if (size < 1024 * 1024) return std::to_string(size / 1024) + " KB";
        return std::to_string(size / (1024 * 1024)) + " MB";
    }
    
    void analyzeFileSystem() {
        std::vector<std::string> expectedDirectories = {
            "models/qlearning",
            "models/dqn", 
            "models/ppo",
            "models/actor_critic"
        };
        
        for (const auto& dir : expectedDirectories) {
            if (std::filesystem::exists(dir)) {
                int fileCount = 0;
                for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                    if (entry.is_regular_file()) fileCount++;
                }
                std::cout << "   ✅ " << dir << " (" << fileCount << " files)\n";
            } else {
                std::cout << "   ❌ " << dir << " (missing)\n";
            }
        }
        
        std::cout << "\n";
    }
    
    void generateRecommendations() {
        std::cout << "💡 Recommendations:\n\n";
        
        std::cout << "1. 📦 If models fail to load:\n";
        std::cout << "   - Ensure models were exported correctly from Python\n";
        std::cout << "   - Check torch.jit.save() was used for .pt files\n";
        std::cout << "   - Verify model input dimensions match (8D expected)\n\n";
        
        std::cout << "2. 🔧 If LibTorch errors occur:\n";
        std::cout << "   - Verify LibTorch version compatibility\n";
        std::cout << "   - Check DLL paths on Windows\n";
        std::cout << "   - Ensure C++ standard is 17 or higher\n\n";
        
        std::cout << "3. 📁 For missing models:\n";
        std::cout << "   - Run Python training: python train_models.py --technique all\n";
        std::cout << "   - Check models directory structure\n";
        std::cout << "   - Verify file permissions\n\n";
        
        std::cout << "4. 🚀 For optimal performance:\n";
        std::cout << "   - Use Release build for production\n";
        std::cout << "   - Enable GPU support if available\n";
        std::cout << "   - Monitor model inference latency\n\n";
    }
};

int main(int argc, char* argv[]) {
    std::string modelsDir = "models/";
    
    // Parse command line arguments
    if (argc > 1) {
        modelsDir = argv[1];
    }
    
    try {
        ModelDebugger debugger(modelsDir);
        
        debugger.printHeader();
        debugger.scanModelsDirectory();
        debugger.testLibTorchInstallation();
        debugger.testTrainedModelManager();
        debugger.testModelLoading();
        debugger.testLibTorchInference();
        debugger.generateReport();
        
        std::cout << "✅ Model debugging complete! Check logs/model_debug.log for detailed output.\n";
        std::cout << "\n💡 Usage in game:\n";
        std::cout << "   1. Run SnakeAI-MLOps.exe\n";
        std::cout << "   2. Select 'Agent vs System' mode\n";
        std::cout << "   3. Choose your preferred AI model\n";
        std::cout << "   4. Watch the AI play!\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}