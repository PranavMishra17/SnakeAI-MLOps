#pragma once

#ifdef TORCH_AVAILABLE
#include <torch/torch.h>
#include <torch/script.h>
#endif

#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <spdlog/spdlog.h>
#include "GameState.hpp"

class TorchInference {
protected:
#ifdef TORCH_AVAILABLE
    torch::jit::script::Module m_model;
    torch::Device m_device;
#endif
    bool m_isLoaded;
    std::string m_modelPath;
    std::string m_lastError;
    
    // Convert EnhancedState to 8D vector matching Python training
    std::vector<float> convertToTrainingState(const EnhancedState& state) {
        const auto& basic = state.basic;
        
        std::vector<float> result = {
            basic.dangerStraight ? 1.0f : 0.0f,
            basic.dangerLeft ? 1.0f : 0.0f,
            basic.dangerRight ? 1.0f : 0.0f,
            static_cast<float>(static_cast<int>(basic.currentDirection)) / 3.0f, // Normalize 0-3 to 0-1
            basic.foodLeft ? 1.0f : 0.0f,
            basic.foodRight ? 1.0f : 0.0f,
            basic.foodUp ? 1.0f : 0.0f,
            basic.foodDown ? 1.0f : 0.0f
        };
        
        // Log the state conversion for debugging
        spdlog::debug("TorchInference: State conversion - danger:[{},{},{}], dir:{:.3f}, food:[{},{},{},{}]",
                     result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]);
        
        return result;
    }
    
public:
    TorchInference() : 
#ifdef TORCH_AVAILABLE
        m_device(torch::kCPU),
#endif
        m_isLoaded(false) {
        
        spdlog::debug("TorchInference: Base class constructor");
        
#ifdef TORCH_AVAILABLE
        spdlog::debug("TorchInference: LibTorch available - using CPU device");
        
        // Check if CUDA is available
        if (torch::cuda::is_available()) {
            spdlog::info("TorchInference: CUDA is available, but using CPU for compatibility");
            // Note: Could enable CUDA here if needed: m_device = torch::kCUDA;
        } else {
            spdlog::info("TorchInference: CUDA not available, using CPU");
        }
#else
        spdlog::warn("TorchInference: LibTorch not available - neural network features disabled");
        m_lastError = "LibTorch not available in this build";
#endif
    }
    
    virtual ~TorchInference() {
        spdlog::debug("TorchInference: Destructor called for model: {}", m_modelPath);
    }
    
    // Check if LibTorch is available at compile time
    bool isTorchAvailable() const {
#ifdef TORCH_AVAILABLE
        return true;
#else
        return false;
#endif
    }
    
    virtual bool loadModel(const std::string& modelPath);
    virtual std::vector<float> predict(const std::vector<float>& input);
    int predictAction(const std::vector<float>& input);
    std::vector<float> predictValues(const std::vector<float>& input);
    
    virtual bool isLoaded() const { return m_isLoaded; }
    const std::string& getModelPath() const { return m_modelPath; }
    const std::string& getLastError() const { return m_lastError; }
    
    // Debug utilities
    void logModelInfo() const {
        if (m_isLoaded) {
            spdlog::info("TorchInference: Model loaded from {}", m_modelPath);
#ifdef TORCH_AVAILABLE
            spdlog::info("TorchInference: Device: {}", m_device.str());
#endif
        } else {
            spdlog::warn("TorchInference: No model loaded");
        }
    }
    
    // Test model with a variety of inputs
    bool runModelTests() {
        if (!isTorchAvailable()) {
            spdlog::error("TorchInference: Cannot run tests - LibTorch not available");
            return false;
        }
        
        if (!m_isLoaded) {
            spdlog::error("TorchInference: Cannot run tests - model not loaded");
            return false;
        }
        
        spdlog::info("TorchInference: Running comprehensive model tests...");
        
        std::vector<std::vector<float>> testCases = {
            {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, // Safe, food left
            {1.0f, 0.0f, 0.0f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f}, // Danger straight, food right
            {0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}, // Danger left/right, food up
            {1.0f, 1.0f, 1.0f, 0.25f, 0.0f, 0.0f, 0.0f, 1.0f} // All danger, food down
        };
        
        for (size_t i = 0; i < testCases.size(); ++i) {
            auto output = predict(testCases[i]);
            if (output.empty()) {
                spdlog::error("TorchInference: Test case {} failed - empty output", i);
                return false;
            }
            spdlog::info("TorchInference: Test case {} passed - output size: {}", i, output.size());
        }
        
        spdlog::info("TorchInference: All model tests passed!");
        return true;
    }
};

// DQN-specific inference
class DQNInference : public TorchInference {
public:
    DQNInference() : TorchInference() {
        spdlog::debug("DQNInference: Constructor called");
    }
    
    Direction getAction(const EnhancedState& state, float epsilon = 0.0f);
    
    // Override to handle DQN-specific model paths
    bool loadModel(const std::string& modelPath) override {
        spdlog::info("DQNInference: Attempting to load DQN model from: {}", modelPath);
        
        if (!isTorchAvailable()) {
            spdlog::error("DQNInference: LibTorch not available - cannot load neural network models");
            m_lastError = "LibTorch not available in this build";
            return false;
        }
        
        // Try original path first
        if (TorchInference::loadModel(modelPath)) {
            spdlog::info("DQNInference: Successfully loaded DQN model");
            return true;
        }
        
        // Try .pt variant if .pth failed
        if (modelPath.find(".pth") != std::string::npos) {
            std::string ptPath = modelPath;
            size_t pos = ptPath.find(".pth");
            ptPath.replace(pos, 4, ".pt");
            
            spdlog::info("DQNInference: Trying .pt variant: {}", ptPath);
            if (TorchInference::loadModel(ptPath)) {
                spdlog::info("DQNInference: Successfully loaded DQN model from .pt file");
                return true;
            }
        }
        
        spdlog::error("DQNInference: Failed to load DQN model from any variant");
        return false;
    }
    
private:
    Direction fallbackAction(const EnhancedState& state);
};

// PPO-specific inference (loads policy network)
class PPOInference : public TorchInference {
private:
#ifdef TORCH_AVAILABLE
    torch::jit::script::Module m_policyModel;
#endif
    bool m_policyLoaded = false;
    
public:
    PPOInference() : TorchInference() {
        spdlog::debug("PPOInference: Constructor called");
    }
    
    bool loadModel(const std::string& modelPath) override;
    std::vector<float> predict(const std::vector<float>& input) override;
    bool isLoaded() const override { return m_policyLoaded; }
    Direction getAction(const EnhancedState& state);
    
private:
    Direction fallbackAction(const EnhancedState& state);
};

// Actor-Critic inference (loads both actor and critic)
class ActorCriticInference : public TorchInference {
private:
#ifdef TORCH_AVAILABLE
    torch::jit::script::Module m_actorModel;
    torch::jit::script::Module m_criticModel;
#endif
    bool m_actorLoaded = false;
    bool m_criticLoaded = false;
    
public:
    ActorCriticInference() : TorchInference() {
        spdlog::debug("ActorCriticInference: Constructor called");
    }
    
    bool loadModel(const std::string& modelPath) override;
    std::vector<float> predict(const std::vector<float>& input) override;
    bool isLoaded() const override { return m_actorLoaded; }
    Direction getAction(const EnhancedState& state);
    float getValue(const EnhancedState& state);
    
private:
    Direction fallbackAction(const EnhancedState& state);
};