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
        
        return {
            basic.dangerStraight ? 1.0f : 0.0f,
            basic.dangerLeft ? 1.0f : 0.0f,
            basic.dangerRight ? 1.0f : 0.0f,
            static_cast<float>(static_cast<int>(basic.currentDirection)) / 3.0f, // Normalize 0-3 to 0-1
            basic.foodLeft ? 1.0f : 0.0f,
            basic.foodRight ? 1.0f : 0.0f,
            basic.foodUp ? 1.0f : 0.0f,
            basic.foodDown ? 1.0f : 0.0f
        };
    }
    
public:
    TorchInference() : m_device(torch::kCPU), m_isLoaded(false) {}
    virtual ~TorchInference() = default;
    
    virtual bool loadModel(const std::string& modelPath);
    virtual std::vector<float> predict(const std::vector<float>& input);
    int predictAction(const std::vector<float>& input);
    std::vector<float> predictValues(const std::vector<float>& input);
    
    virtual bool isLoaded() const { return m_isLoaded; }
    const std::string& getModelPath() const { return m_modelPath; }
};

// DQN-specific inference
class DQNInference : public TorchInference {
public:
    Direction getAction(const EnhancedState& state, float epsilon = 0.0f);
    
private:
    Direction fallbackAction(const EnhancedState& state);
};

// PPO-specific inference (loads policy network)
class PPOInference : public TorchInference {
private:
    torch::jit::script::Module m_policyModel;
    bool m_policyLoaded = false;
    
public:
    bool loadModel(const std::string& modelPath) override {
        // PPO creates separate _policy.pt and _value.pt files
        std::string policyPath = modelPath;
        
        // Convert .pth to _policy.pt
        if (policyPath.find(".pth") != std::string::npos) {
            policyPath = policyPath.substr(0, policyPath.find(".pth")) + "_policy.pt";
        } else if (policyPath.find(".pt") != std::string::npos) {
            // If already .pt, check if it needs _policy suffix
            if (policyPath.find("_policy") == std::string::npos) {
                policyPath = policyPath.substr(0, policyPath.find(".pt")) + "_policy.pt";
            }
        }
        
        if (!std::filesystem::exists(policyPath)) {
            spdlog::warn("PPOInference: Policy model not found: {}", policyPath);
            return false;
        }
        
        try {
            torch::NoGradGuard no_grad;
            m_policyModel = torch::jit::load(policyPath, m_device);
            m_policyModel.eval();
            m_policyLoaded = true;
            m_modelPath = policyPath;
            
            spdlog::info("PPOInference: Loaded policy model: {}", policyPath);
            
            // Test with 8D input
            std::vector<float> testInput(8, 0.0f);
            auto testOutput = predict(testInput);
            
            if (!testOutput.empty()) {
                spdlog::info("PPOInference: Policy model validation successful");
                return true;
            }
            
        } catch (const std::exception& e) {
            spdlog::error("PPOInference: Failed to load policy model: {}", e.what());
            m_policyLoaded = false;
        }
        
        return false;
    }
    
    std::vector<float> predict(const std::vector<float>& input) override {
        if (!m_policyLoaded) return {};
        
        try {
            torch::NoGradGuard no_grad;
            
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
    bool loadModel(const std::string& modelPath) override {
        // Actor-Critic creates separate _actor.pt and _critic.pt files
        std::string actorPath = modelPath;
        std::string criticPath = modelPath;
        
        // Convert .pth to _actor.pt and _critic.pt
        if (actorPath.find(".pth") != std::string::npos) {
            std::string basePath = actorPath.substr(0, actorPath.find(".pth"));
            actorPath = basePath + "_actor.pt";
            criticPath = basePath + "_critic.pt";
        } else if (actorPath.find(".pt") != std::string::npos) {
            std::string basePath = actorPath.substr(0, actorPath.find(".pt"));
            actorPath = basePath + "_actor.pt";
            criticPath = basePath + "_critic.pt";
        }
        
        bool success = false;
        
        // Load actor model
        if (std::filesystem::exists(actorPath)) {
            try {
                torch::NoGradGuard no_grad;
                m_actorModel = torch::jit::load(actorPath, m_device);
                m_actorModel.eval();
                m_actorLoaded = true;
                spdlog::info("ActorCriticInference: Loaded actor model: {}", actorPath);
                success = true;
            } catch (const std::exception& e) {
                spdlog::error("ActorCriticInference: Failed to load actor: {}", e.what());
                m_actorLoaded = false;
            }
        }
        
        // Load critic model (optional for action selection)
        if (std::filesystem::exists(criticPath)) {
            try {
                torch::NoGradGuard no_grad;
                m_criticModel = torch::jit::load(criticPath, m_device);
                m_criticModel.eval();
                m_criticLoaded = true;
                spdlog::info("ActorCriticInference: Loaded critic model: {}", criticPath);
            } catch (const std::exception& e) {
                spdlog::error("ActorCriticInference: Failed to load critic: {}", e.what());
                m_criticLoaded = false;
            }
        }
        
        if (success) {
            m_modelPath = actorPath;
            
            // Test with 8D input
            std::vector<float> testInput(8, 0.0f);
            auto testOutput = predict(testInput);
            
            if (!testOutput.empty()) {
                spdlog::info("ActorCriticInference: Actor-Critic validation successful");
            }
        }
        
        return success;
    }
    
    std::vector<float> predict(const std::vector<float>& input) override {
        if (!m_actorLoaded) return {};
        
        try {
            torch::NoGradGuard no_grad;
            
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