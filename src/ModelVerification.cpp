#include "MLAgents.hpp"
#include <iostream>
#include <filesystem>

void testModelLoading() {
    std::cout << "🔍 SnakeAI-MLOps Model Verification" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Initialize model manager
    TrainedModelManager manager;
    auto models = manager.getAvailableModels();
    
    std::cout << "\n📊 Found " << models.size() << " trained models:" << std::endl;
    
    if (models.empty()) {
        std::cout << "❌ No trained models found!" << std::endl;
        std::cout << "💡 Run Python training first:" << std::endl;
        std::cout << "   python src/train_models.py --technique all" << std::endl;
        return;
    }
    
    // Display all found models
    for (const auto& model : models) {
        std::cout << "\n🤖 " << model.name << std::endl;
        std::cout << "   Type: " << model.modelType << std::endl;
        std::cout << "   Profile: " << model.profile << std::endl;
        std::cout << "   Path: " << model.modelPath << std::endl;
        std::cout << "   Exists: " << (std::filesystem::exists(model.modelPath) ? "✅" : "❌") << std::endl;
    }
    
    // Test Q-Learning model loading
    std::cout << "\n🧪 Testing Q-Learning Model Loading:" << std::endl;
    for (const auto& model : models) {
        if (model.modelType == "qlearning" && std::filesystem::exists(model.modelPath)) {
            std::cout << "\n🎯 Testing: " << model.name << std::endl;
            
            try {
                auto agent = std::make_unique<QLearningAgentEnhanced>(model);
                
                if (agent->isPreTrained()) {
                    std::cout << "   ✅ Model loaded successfully" << std::endl;
                    std::cout << "   📈 Q-table size: " << agent->getQTableSize() << " states" << std::endl;
                    std::cout << "   🎛️  Epsilon: " << agent->getEpsilon() << std::endl;
                    std::cout << "   ℹ️  Info: " << agent->getModelInfo() << std::endl;
                    
                    // Test a simple inference
                    EnhancedState testState;
                    testState.basic.dangerStraight = false;
                    testState.basic.dangerLeft = false;
                    testState.basic.dangerRight = true;
                    testState.basic.currentDirection = Direction::RIGHT;
                    testState.basic.foodLeft = true;
                    testState.basic.foodRight = false;
                    testState.basic.foodUp = false;
                    testState.basic.foodDown = false;
                    
                    Direction action = agent->getAction(testState, false);
                    std::cout << "   🎮 Test action: " << static_cast<int>(action) << std::endl;
                    
                } else {
                    std::cout << "   ❌ Failed to load pre-trained model" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "   ❌ Exception: " << e.what() << std::endl;
            }
            
            break; // Test only one Q-Learning model
        }
    }
    
    // Test Neural Network Placeholders
    std::cout << "\n🧠 Testing Neural Network Placeholders:" << std::endl;
    
    std::vector<std::pair<std::string, AgentType>> neuralTypes = {
        {"DQN", AgentType::DEEP_Q_NETWORK},
        {"Policy Gradient", AgentType::POLICY_GRADIENT},
        {"Actor-Critic", AgentType::ACTOR_CRITIC}
    };
    
    for (const auto& [name, type] : neuralTypes) {
        std::cout << "\n🎭 Testing: " << name << std::endl;
        
        AgentConfig config;
        config.type = type;
        config.name = name + " Test";
        config.isImplemented = true;
        
        auto agent = AgentFactory::createAgent(config);
        
        if (agent) {
            std::cout << "   ✅ Agent created successfully" << std::endl;
            std::cout << "   ℹ️  Info: " << agent->getAgentInfo() << std::endl;
            
            // Test inference
            EnhancedState testState;
            testState.basic.dangerStraight = false;
            testState.basic.dangerLeft = true;
            testState.basic.dangerRight = false;
            testState.basic.currentDirection = Direction::UP;
            testState.basic.foodLeft = false;
            testState.basic.foodRight = true;
            testState.basic.foodUp = false;
            testState.basic.foodDown = true;
            
            Direction action = agent->getAction(testState, false);
            std::cout << "   🎮 Test action: " << static_cast<int>(action) << std::endl;
        } else {
            std::cout << "   ❌ Failed to create agent" << std::endl;
        }
    }
    
    // Test Agent Factory with trained models
    std::cout << "\n🏭 Testing Agent Factory with Trained Models:" << std::endl;
    for (const auto& model : models) {
        if (std::filesystem::exists(model.modelPath)) {
            std::cout << "\n🔧 Creating agent: " << model.name << std::endl;
            
            auto agent = AgentFactory::createTrainedAgent(model.name);
            if (agent) {
                std::cout << "   ✅ Agent created from trained model" << std::endl;
                std::cout << "   ℹ️  Model info: " << agent->getModelInfo() << std::endl;
            } else {
                std::cout << "   ❌ Failed to create agent from trained model" << std::endl;
            }
        }
    }
    
    std::cout << "\n🎉 Model Verification Complete!" << std::endl;
    std::cout << "\n💡 Usage Tips:" << std::endl;
    std::cout << "   • Run the game and select 'Agent vs System' mode" << std::endl;
    std::cout << "   • Choose from available trained models" << std::endl;
    std::cout << "   • Watch the AI play automatically!" << std::endl;
    std::cout << "   • Use +/- keys to adjust game speed" << std::endl;
    std::cout << "   • Press F2 during gameplay to switch agents" << std::endl;
}

int main() {
    try {
        testModelLoading();
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during verification: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}