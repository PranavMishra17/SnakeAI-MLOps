#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <spdlog/spdlog.h>
#include "GameState.hpp"

class TorchInference {
protected:
    torch::jit::script::Module m_model;
    torch::Device m_device;
    bool m_isLoaded;
    std::string m_modelPath;
    
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
    TorchInference() : m_device(torch::kCPU), m_isLoaded(false) {
        spdlog::debug("TorchInference: Base class constructor - using CPU device");
        
        // Check if CUDA is available
        if (torch::cuda::is_available()) {
            spdlog::info("TorchInference: CUDA is available, but using CPU for compatibility");
            // Note: Could enable CUDA here if needed: m_device = torch::kCUDA;
        } else {
            spdlog::info("TorchInference: CUDA not available, using CPU");
        }
    }
    
    virtual ~TorchInference() {
        spdlog::debug("TorchInference: Destructor called for model: {}", m_modelPath);
    }
    
    virtual bool loadModel(const std::string& modelPath);
    virtual std::vector<float> predict(const std::vector<float>& input);
    int predictAction(const std::vector<float>& input);
    std::vector<float> predictValues(const std::vector<float>& input);
    
    virtual bool isLoaded() const { return m_isLoaded; }
    const std::string& getModelPath() const { return m_modelPath; }
    
    // Debug utilities
    void logModelInfo() const {
        if (m_isLoaded) {
            spdlog::info("TorchInference: Model loaded from {}", m_modelPath);
            spdlog::info("TorchInference: Device: {}", m_device.str());
        } else {
            spdlog::warn("TorchInference: No model loaded");
        }
    }
    
    // Test model with a variety of inputs
    bool runModelTests() {
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
    torch::jit::script::Module m_policyModel;
    bool m_policyLoaded = false;
    
public:
    PPOInference() : TorchInference() {
        spdlog::debug("PPOInference: Constructor called");
    }
    
    bool loadModel(const std::string& modelPath) override {
        spdlog::info("PPOInference: Attempting to load PPO model from: {}", modelPath);
        
        // PPO creates separate _policy.pt and _value.pt files
        std::vector<std::string> possiblePaths;
        
        // Try original path first
        possiblePaths.push_back(modelPath);
        
        // Convert .pth to _policy.pt
        if (modelPath.find(".pth") != std::string::npos) {
            std::string policyPath = modelPath.substr(0, modelPath.find(".pth")) + "_policy.pt";
            possiblePaths.push_back(policyPath);
        }
        
        // Convert .pt to _policy.pt if not already
        if (modelPath.find(".pt") != std::string::npos && modelPath.find("_policy") == std::string::npos) {
            std::string policyPath = modelPath.substr(0, modelPath.find(".pt")) + "_policy.pt";
            possiblePaths.push_back(policyPath);
        }
        
        // Try base name + _policy.pt
        std::filesystem::path basePath(modelPath);
        std::string basePathStr = basePath.parent_path().string() + "/" + basePath.stem().string() + "_policy.pt";
        possiblePaths.push_back(basePathStr);
        
        for (const auto& path : possiblePaths) {
            spdlog::debug("PPOInference: Trying policy path: {}", path);
            
            if (!std::filesystem::exists(path)) {
                spdlog::debug("PPOInference: Path does not exist: {}", path);
                continue;
            }
            
            try {
                torch::NoGradGuard no_grad;
                m_policyModel = torch::jit::load(path, m_device);
                m_policyModel.eval();
                m_policyLoaded = true;
                m_modelPath = path;
                m_isLoaded = true;
                
                spdlog::info("PPOInference: Successfully loaded policy model: {}", path);
                
                // Test with 8D input
                std::vector<float> testInput(8, 0.0f);
                auto testOutput = predict(testInput);
                
                if (!testOutput.empty()) {
                    spdlog::info("PPOInference: Policy model validation successful");
                    return true;
                } else {
                    spdlog::error("PPOInference: Policy model test failed");
                    m_policyLoaded = false;
                    m_isLoaded = false;
                }
                
            } catch (const std::exception& e) {
                spdlog::error("PPOInference: Failed to load policy model {}: {}", path, e.what());
                m_policyLoaded = false;
                m_isLoaded = false;
            }
        }
        
        spdlog::error("PPOInference: Failed to load PPO policy model from any path");
        return false;
    }
    
    std::vector<float> predict(const std::vector<float>& input) override {
        if (!m_policyLoaded) {
            spdlog::error("PPOInference: Policy model not loaded");
            return {};
        }
        
        try {
            torch::NoGradGuard no_grad;
            
            if (input.size() != 8) {
                spdlog::error("PPOInference: Expected 8D input, got {}D", input.size());
                return {};
            }
            
            torch::Tensor inputTensor = torch::from_blob(
                const_cast<float*>(input.data()), 
                {1, static_cast<long>(input.size())}, 
                torch::kFloat32
            ).to(m_device);
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(inputTensor);
            
            at::Tensor output = m_policyModel.forward(inputs).toTensor();
            output = output.to(torch::kCPU).contiguous();
            
            std::vector<float> result;
            float* data_ptr = output.data_ptr<float>();
            result.assign(data_ptr, data_ptr + output.numel());
            
            return result;
            
        } catch (const std::exception& e) {
            spdlog::error("PPOInference: Prediction failed: {}", e.what());
            return {};
        }
    }
    
    bool isLoaded() const override { return m_policyLoaded; }
    
    Direction getAction(const EnhancedState& state);
    
private:
    Direction fallbackAction(const EnhancedState& state);
};

// Actor-Critic inference (loads both actor and critic)
class ActorCriticInference : public TorchInference {
private:
    torch::jit::script::Module m_actorModel;
    torch::jit::script::Module m_criticModel;
    bool m_actorLoaded = false;
    bool m_criticLoaded = false;
    
public:
    ActorCriticInference() : TorchInference() {
        spdlog::debug("ActorCriticInference: Constructor called");
    }
    
    bool loadModel(const std::string& modelPath) override {
        spdlog::info("ActorCriticInference: Attempting to load Actor-Critic model from: {}", modelPath);
        
        // Actor-Critic creates separate _actor.pt and _critic.pt files
        std::vector<std::pair<std::string, std::string>> possiblePaths;
        
        // Try original path variants
        std::filesystem::path basePath(modelPath);
        std::string basePathStr = basePath.parent_path().string() + "/" + basePath.stem().string();
        
        possiblePaths.push_back({basePathStr + "_actor.pt", basePathStr + "_critic.pt"});
        
        if (modelPath.find(".pth") != std::string::npos) {
            std::string base = modelPath.substr(0, modelPath.find(".pth"));
            possiblePaths.push_back({base + "_actor.pt", base + "_critic.pt"});
        }
        
        if (modelPath.find(".pt") != std::string::npos) {
            std::string base = modelPath.substr(0, modelPath.find(".pt"));
            possiblePaths.push_back({base + "_actor.pt", base + "_critic.pt"});
        }
        
        bool success = false;
        
        for (const auto& [actorPath, criticPath] : possiblePaths) {
            spdlog::debug("ActorCriticInference: Trying actor: {}, critic: {}", actorPath, criticPath);
            
            // Load actor model (required)
            if (std::filesystem::exists(actorPath)) {
                try {
                    torch::NoGradGuard no_grad;
                    m_actorModel = torch::jit::load(actorPath, m_device);
                    m_actorModel.eval();
                    m_actorLoaded = true;
                    m_modelPath = actorPath;
                    m_isLoaded = true;
                    spdlog::info("ActorCriticInference: Loaded actor model: {}", actorPath);
                    success = true;
                    break;
                } catch (const std::exception& e) {
                    spdlog::error("ActorCriticInference: Failed to load actor {}: {}", actorPath, e.what());
                    m_actorLoaded = false;
                }
            } else {
                spdlog::debug("ActorCriticInference: Actor file does not exist: {}", actorPath);
            }
        }
        
        if (!success) {
            spdlog::error("ActorCriticInference: Failed to load actor model from any path");
            return false;
        }
        
        // Load critic model (optional for action selection)
        for (const auto& [actorPath, criticPath] : possiblePaths) {
            if (std::filesystem::exists(criticPath)) {
                try {
                    torch::NoGradGuard no_grad;
                    m_criticModel = torch::jit::load(criticPath, m_device);
                    m_criticModel.eval();
                    m_criticLoaded = true;
                    spdlog::info("ActorCriticInference: Loaded critic model: {}", criticPath);
                    break;
                } catch (const std::exception& e) {
                    spdlog::error("ActorCriticInference: Failed to load critic {}: {}", criticPath, e.what());
                    m_criticLoaded = false;
                }
            }
        }
        
        if (success) {
            // Test with 8D input
            std::vector<float> testInput(8, 0.0f);
            auto testOutput = predict(testInput);
            
            if (!testOutput.empty()) {
                spdlog::info("ActorCriticInference: Actor-Critic validation successful");
            } else {
                spdlog::warn("ActorCriticInference: Actor model test failed, but model loaded");
            }
        }
        
        return success;
    }
    
    std::vector<float> predict(const std::vector<float>& input) override {
        if (!m_actorLoaded) {
            spdlog::error("ActorCriticInference: Actor model not loaded");
            return {};
        }
        
        try {
            torch::NoGradGuard no_grad;
            
            if (input.size() != 8) {
                spdlog::error("ActorCriticInference: Expected 8D input, got {}D", input.size());
                return {};
            }
            
            torch::Tensor inputTensor = torch::from_blob(
                const_cast<float*>(input.data()), 
                {1, static_cast<long>(input.size())}, 
                torch::kFloat32
            ).to(m_device);
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(inputTensor);
            
            at::Tensor output = m_actorModel.forward(inputs).toTensor();
            output = output.to(torch::kCPU).contiguous();
            
            std::vector<float> result;
            float* data_ptr = output.data_ptr<float>();
            result.assign(data_ptr, data_ptr + output.numel());
            
            return result;
            
        } catch (const std::exception& e) {
            spdlog::error("ActorCriticInference: Actor prediction failed: {}", e.what());
            return {};
        }
    }
    
    bool isLoaded() const override { return m_actorLoaded; }
    
    Direction getAction(const EnhancedState& state);
    float getValue(const EnhancedState& state);
    
private:
    Direction fallbackAction(const EnhancedState& state);
};