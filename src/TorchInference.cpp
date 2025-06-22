#include "TorchInference.hpp"
#include <filesystem>
#include <random>
#include <spdlog/spdlog.h>

// Helper function to normalize paths for LibTorch (only when available)
std::string normalizePath(const std::string& path) {
    std::filesystem::path fsPath(path);
    std::string normalized = fsPath.string();
    
    // Convert backslashes to forward slashes for LibTorch compatibility
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    
    // Ensure path is absolute
    if (!fsPath.is_absolute()) {
        fsPath = std::filesystem::absolute(fsPath);
        normalized = fsPath.string();
        std::replace(normalized.begin(), normalized.end(), '\\', '/');
    }
    
    return normalized;
}

// TorchInference Implementation
bool TorchInference::loadModel(const std::string& modelPath) {
    spdlog::info("TorchInference: Attempting to load model from: {}", modelPath);
    
    if (!isTorchAvailable()) {
        m_lastError = "LibTorch not available in this build - cannot load neural network models";
        spdlog::error("TorchInference: {}", m_lastError);
        return false;
    }
    
    if (modelPath.empty()) {
        m_lastError = "Empty model path provided";
        spdlog::error("TorchInference: {}", m_lastError);
        return false;
    }
    
    if (!std::filesystem::exists(modelPath)) {
        m_lastError = "Model file not found: " + modelPath;
        spdlog::error("TorchInference: {}", m_lastError);
        return false;
    }
    
    auto fileSize = std::filesystem::file_size(modelPath);
    spdlog::info("TorchInference: Model file exists, size: {} bytes", fileSize);
    
    if (fileSize == 0) {
        m_lastError = "Model file is empty: " + modelPath;
        spdlog::error("TorchInference: {}", m_lastError);
        return false;
    }
    
#ifdef TORCH_AVAILABLE
    // Normalize the path for LibTorch
    std::string normalizedPath = normalizePath(modelPath);
    spdlog::info("TorchInference: Normalized path: {}", normalizedPath);
    
    try {
        spdlog::info("TorchInference: Initializing LibTorch...");
        torch::NoGradGuard no_grad;
        
        spdlog::info("TorchInference: Loading JIT module from: {}", normalizedPath);
        
        // Use normalized path with LibTorch
        m_model = torch::jit::load(normalizedPath, m_device);
        
        spdlog::info("TorchInference: Setting model to evaluation mode...");
        m_model.eval();
        
        m_isLoaded = true;
        m_modelPath = normalizedPath;
        m_lastError.clear();
        
        spdlog::info("TorchInference: Successfully loaded model from {}", normalizedPath);
        
        // Test inference with 8D input (matching Python training)
        spdlog::info("TorchInference: Testing model with sample 8D input...");
        std::vector<float> testInput(8, 0.5f);
        testInput[0] = 0.0f; // dangerStraight
        testInput[1] = 1.0f; // dangerLeft  
        testInput[2] = 0.0f; // dangerRight
        testInput[3] = 0.33f; // currentDirection (1/3 = DOWN)
        testInput[4] = 1.0f; // foodLeft
        testInput[5] = 0.0f; // foodRight
        testInput[6] = 0.0f; // foodUp
        testInput[7] = 0.0f; // foodDown
        
        spdlog::debug("TorchInference: Test input: [{}, {}, {}, {}, {}, {}, {}, {}]", 
                     testInput[0], testInput[1], testInput[2], testInput[3],
                     testInput[4], testInput[5], testInput[6], testInput[7]);
        
        auto testOutput = predict(testInput);
        
        if (!testOutput.empty()) {
            spdlog::info("TorchInference: Model validation successful, output size: {}", testOutput.size());
            spdlog::debug("TorchInference: Test output: [{}]", 
                         testOutput.size() >= 4 ? 
                         fmt::format("{}, {}, {}, {}", testOutput[0], testOutput[1], testOutput[2], testOutput[3]) :
                         "insufficient output");
            
            // Validate output has expected structure
            if (testOutput.size() >= 4) {
                float sum = 0.0f;
                for (size_t i = 0; i < 4; ++i) {
                    sum += testOutput[i];
                }
                spdlog::debug("TorchInference: Output sum: {:.4f}", sum);
                
                // Check if outputs are reasonable (not all zero, not all NaN)
                bool hasValidOutput = false;
                for (size_t i = 0; i < 4; ++i) {
                    if (!std::isnan(testOutput[i]) && !std::isinf(testOutput[i])) {
                        hasValidOutput = true;
                        break;
                    }
                }
                
                if (!hasValidOutput) {
                    m_lastError = "Model outputs invalid values (NaN/Inf)";
                    spdlog::error("TorchInference: {}", m_lastError);
                    m_isLoaded = false;
                    return false;
                }
                
                spdlog::info("TorchInference: Model outputs appear valid");
            } else {
                spdlog::warn("TorchInference: Model output size {} < expected 4", testOutput.size());
            }
        } else {
            m_lastError = "Model test inference failed - empty output";
            spdlog::error("TorchInference: {}", m_lastError);
            m_isLoaded = false;
            return false;
        }
        
        return true;
        
    } catch (const c10::Error& e) {
        m_lastError = "LibTorch c10::Error: " + std::string(e.what());
        spdlog::error("TorchInference: LibTorch c10::Error loading {}: {}", normalizedPath, e.what());
        spdlog::error("TorchInference: Error type: c10::Error - this usually indicates model format issues");
        spdlog::error("TorchInference: Possible causes:");
        spdlog::error("  - Model was not saved with torch.jit.save()");
        spdlog::error("  - Model architecture mismatch");
        spdlog::error("  - LibTorch version incompatibility");
        m_isLoaded = false;
        return false;
    } catch (const torch::jit::ErrorReport& e) {
        m_lastError = "LibTorch JIT ErrorReport: " + std::string(e.what());
        spdlog::error("TorchInference: LibTorch JIT ErrorReport loading {}: {}", normalizedPath, e.what());
        spdlog::error("TorchInference: Error type: JIT ErrorReport - model compilation/format issue");
        m_isLoaded = false;
        return false;
    } catch (const std::runtime_error& e) {
        m_lastError = "Runtime error: " + std::string(e.what());
        spdlog::error("TorchInference: Runtime error loading {}: {}", normalizedPath, e.what());
        spdlog::error("TorchInference: Error type: Runtime error - check model compatibility");
        m_isLoaded = false;
        return false;
    } catch (const std::exception& e) {
        m_lastError = "Standard exception: " + std::string(e.what());
        spdlog::error("TorchInference: Standard exception loading {}: {}", normalizedPath, e.what());
        spdlog::error("TorchInference: Error type: std::exception");
        m_isLoaded = false;
        return false;
    } catch (...) {
        m_lastError = "Unknown exception occurred while loading model";
        spdlog::error("TorchInference: Unknown exception loading {}", normalizedPath);
        spdlog::error("TorchInference: Error type: Unknown exception");
        m_isLoaded = false;
        return false;
    }
#else
    m_lastError = "LibTorch not available in this build";
    spdlog::error("TorchInference: {}", m_lastError);
    return false;
#endif
}

std::vector<float> TorchInference::predict(const std::vector<float>& input) {
    if (!isTorchAvailable()) {
        m_lastError = "LibTorch not available in this build";
        spdlog::error("TorchInference: predict() called but LibTorch not available");
        return {};
    }
    
    if (!m_isLoaded) {
        m_lastError = "Model not loaded";
        spdlog::error("TorchInference: predict() called but model not loaded");
        return {};
    }
    
    spdlog::debug("TorchInference: predict() called with input size: {}", input.size());
    
#ifdef TORCH_AVAILABLE
    try {
        torch::NoGradGuard no_grad;
        
        // Validate input size (should be 8D)
        if (input.size() != 8) {
            m_lastError = "Expected 8D input, got " + std::to_string(input.size()) + "D";
            spdlog::error("TorchInference: {}", m_lastError);
            return {};
        }
        
        spdlog::debug("TorchInference: Input validated, creating tensor...");
        
        // Create tensor from input vector
        torch::Tensor inputTensor = torch::from_blob(
            const_cast<float*>(input.data()), 
            {1, static_cast<long>(input.size())}, 
            torch::kFloat32
        ).to(m_device);
        
        spdlog::debug("TorchInference: Input tensor created, shape: [{}, {}]", 
                     inputTensor.size(0), inputTensor.size(1));
        
        // Forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);
        
        spdlog::debug("TorchInference: Performing forward pass...");
        at::Tensor output = m_model.forward(inputs).toTensor();
        
        spdlog::debug("TorchInference: Forward pass complete, output shape: [{}, {}]", 
                     output.size(0), output.numel());
        
        // Ensure output is on CPU and contiguous
        output = output.to(torch::kCPU).contiguous();
        
        spdlog::debug("TorchInference: Output moved to CPU, converting to vector...");
        
        // Convert to vector
        std::vector<float> result;
        float* data_ptr = output.data_ptr<float>();
        result.assign(data_ptr, data_ptr + output.numel());
        
        spdlog::debug("TorchInference: Conversion complete, result size: {}", result.size());
        
        // Log output values for debugging
        if (result.size() >= 4) {
            spdlog::debug("TorchInference: Output values: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]", 
                         result[0], result[1], result[2], result[3]);
        }
        
        m_lastError.clear();
        return result;
        
    } catch (const c10::Error& e) {
        m_lastError = "LibTorch c10::Error during prediction: " + std::string(e.what());
        spdlog::error("TorchInference: {}", m_lastError);
        return {};
    } catch (const std::runtime_error& e) {
        m_lastError = "Runtime error during prediction: " + std::string(e.what());
        spdlog::error("TorchInference: {}", m_lastError);
        return {};
    } catch (const std::exception& e) {
        m_lastError = "Exception during prediction: " + std::string(e.what());
        spdlog::error("TorchInference: {}", m_lastError);
        return {};
    } catch (...) {
        m_lastError = "Unknown exception during prediction";
        spdlog::error("TorchInference: {}", m_lastError);
        return {};
    }
#else
    m_lastError = "LibTorch not available in this build";
    spdlog::error("TorchInference: {}", m_lastError);
    return {};
#endif
}

// The fixed section for TorchInference.cpp from around line 948
// Replace the problematic function with this version

int TorchInference::predictAction(const std::vector<float>& input) {
    spdlog::debug("TorchInference: predictAction() called");
    
    auto output = predict(input);
    if (output.empty() || output.size() < 4) {
        spdlog::warn("TorchInference: predictAction() - invalid output, returning default action 0");
        return 0; // Default to UP
    }
    
    // Find action with highest value/probability
    int action = 0;
    float maxValue = output[0];
    
    // Manually find the max element index
    for (int i = 1; i < 4 && i < output.size(); ++i) {
        if (output[i] > maxValue) {
            maxValue = output[i];
            action = i;
        }
    }
    
    spdlog::debug("TorchInference: predictAction() - selected action: {} (value: {:.4f})", 
                 action, maxValue);
    
    return action;
}

std::vector<float> TorchInference::predictValues(const std::vector<float>& input) {
    spdlog::debug("TorchInference: predictValues() called");
    return predict(input);
}

// DQNInference Implementation
Direction DQNInference::getAction(const EnhancedState& state, float epsilon) {
    spdlog::debug("DQNInference: getAction() called with epsilon: {:.4f}", epsilon);
    
    // Epsilon-greedy exploration
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    
    if (epsilon > 0.0f && dis(gen) < epsilon) {
        std::uniform_int_distribution<> actionDis(0, 3);
        Direction randomAction = static_cast<Direction>(actionDis(gen));
        spdlog::debug("DQNInference: Epsilon exploration - random action: {}", static_cast<int>(randomAction));
        return randomAction;
    }
    
    if (!isLoaded() || !isTorchAvailable()) {
        spdlog::debug("DQNInference: Model not loaded or LibTorch not available, using fallback");
        return fallbackAction(state);
    }
    
    // Use 8D state representation
    auto stateVector = convertToTrainingState(state);
    spdlog::debug("DQNInference: State converted to 8D vector");
    
    int action = predictAction(stateVector);
    
    // Clamp action to valid range
    action = std::max(0, std::min(3, action));
    Direction result = static_cast<Direction>(action);
    
    spdlog::debug("DQNInference: Predicted action: {} -> Direction: {}", action, static_cast<int>(result));
    
    return result;
}

Direction DQNInference::fallbackAction(const EnhancedState& state) {
    spdlog::debug("DQNInference: Using fallback heuristic action selection");
    
    const auto& basic = state.basic;
    std::vector<Direction> safeActions;
    
    // Check safe directions
    if (!basic.dangerStraight) {
        safeActions.push_back(basic.currentDirection);
        spdlog::debug("DQNInference: Straight is safe");
    }
    if (!basic.dangerLeft) {
        Direction leftDir = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 3) % 4);
        safeActions.push_back(leftDir);
        spdlog::debug("DQNInference: Left is safe");
    }
    if (!basic.dangerRight) {
        Direction rightDir = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 1) % 4);
        safeActions.push_back(rightDir);
        spdlog::debug("DQNInference: Right is safe");
    }
    
    if (safeActions.empty()) {
        // Last resort - opposite direction
        Direction opposite = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 2) % 4);
        spdlog::warn("DQNInference: No safe actions, using opposite direction: {}", static_cast<int>(opposite));
        return opposite;
    }
    
    // Prefer actions that move toward food
    for (Direction action : safeActions) {
        bool movesTowardFood = false;
        switch (action) {
            case Direction::UP: movesTowardFood = basic.foodUp; break;
            case Direction::DOWN: movesTowardFood = basic.foodDown; break;
            case Direction::LEFT: movesTowardFood = basic.foodLeft; break;
            case Direction::RIGHT: movesTowardFood = basic.foodRight; break;
        }
        if (movesTowardFood) {
            spdlog::debug("DQNInference: Selected food-seeking action: {}", static_cast<int>(action));
            return action;
        }
    }
    
    Direction result = safeActions[0];
    spdlog::debug("DQNInference: Selected first safe action: {}", static_cast<int>(result));
    return result;
}

// PPOInference Implementation
bool PPOInference::loadModel(const std::string& modelPath) {
    spdlog::info("PPOInference: Attempting to load PPO model from: {}", modelPath);
    
    if (!isTorchAvailable()) {
        m_lastError = "LibTorch not available in this build";
        spdlog::error("PPOInference: {}", m_lastError);
        return false;
    }
    
#ifdef TORCH_AVAILABLE
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
            m_lastError.clear();
            
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
            m_lastError = "Failed to load policy model: " + std::string(e.what());
            spdlog::error("PPOInference: Failed to load policy model {}: {}", path, e.what());
            m_policyLoaded = false;
            m_isLoaded = false;
        }
    }
    
    m_lastError = "Failed to load PPO policy model from any path";
    spdlog::error("PPOInference: {}", m_lastError);
    return false;
#else
    m_lastError = "LibTorch not available in this build";
    spdlog::error("PPOInference: {}", m_lastError);
    return false;
#endif
}

std::vector<float> PPOInference::predict(const std::vector<float>& input) {
    if (!isTorchAvailable()) {
        m_lastError = "LibTorch not available in this build";
        spdlog::error("PPOInference: predict() called but LibTorch not available");
        return {};
    }
    
    if (!m_policyLoaded) {
        m_lastError = "Policy model not loaded";
        spdlog::error("PPOInference: {}", m_lastError);
        return {};
    }
    
#ifdef TORCH_AVAILABLE
    try {
        torch::NoGradGuard no_grad;
        
        if (input.size() != 8) {
            m_lastError = "Expected 8D input, got " + std::to_string(input.size()) + "D";
            spdlog::error("PPOInference: {}", m_lastError);
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
        
        m_lastError.clear();
        return result;
        
    } catch (const std::exception& e) {
        m_lastError = "Prediction failed: " + std::string(e.what());
        spdlog::error("PPOInference: {}", m_lastError);
        return {};
    }
#else
    m_lastError = "LibTorch not available in this build";
    spdlog::error("PPOInference: {}", m_lastError);
    return {};
#endif
}

Direction PPOInference::getAction(const EnhancedState& state) {
    spdlog::debug("PPOInference: getAction() called");
    
    if (!isLoaded() || !isTorchAvailable()) {
        spdlog::debug("PPOInference: Model not loaded or LibTorch not available, using fallback");
        return fallbackAction(state);
    }
    
    // Use 8D state representation
    auto stateVector = convertToTrainingState(state);
    spdlog::debug("PPOInference: State converted to 8D vector");
    
    auto actionProbs = predict(stateVector);
    
    if (actionProbs.empty() || actionProbs.size() < 4) {
        spdlog::warn("PPOInference: Invalid action probabilities, using fallback");
        return fallbackAction(state);
    }
    
    spdlog::debug("PPOInference: Raw action probabilities: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]",
                 actionProbs[0], actionProbs[1], actionProbs[2], actionProbs[3]);
    
    // Ensure probabilities are valid and sum to 1
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        actionProbs[i] = std::max(0.0f, actionProbs[i]); // Clamp to positive
        sum += actionProbs[i];
    }
    
    if (sum <= 0.0f) {
        spdlog::warn("PPOInference: Invalid probability sum: {:.4f}, using fallback", sum);
        return fallbackAction(state);
    }
    
    // Normalize probabilities
    for (int i = 0; i < 4; ++i) {
        actionProbs[i] /= sum;
    }
    
    spdlog::debug("PPOInference: Normalized probabilities: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]",
                 actionProbs[0], actionProbs[1], actionProbs[2], actionProbs[3]);
    
    // Sample from probability distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<> dist(actionProbs.begin(), actionProbs.begin() + 4);
    
    int action = dist(gen);
    Direction result = static_cast<Direction>(action);
    
    spdlog::debug("PPOInference: Sampled action: {} (prob: {:.4f})", action, actionProbs[action]);
    
    return result;
}

Direction PPOInference::fallbackAction(const EnhancedState& state) {
    spdlog::debug("PPOInference: Using fallback probabilistic action selection");
    
    const auto& basic = state.basic;
    std::array<float, 4> actionProbs = {0.25f, 0.25f, 0.25f, 0.25f};
    
    // Reduce probability of dangerous actions
    if (basic.dangerStraight) {
        actionProbs[static_cast<int>(basic.currentDirection)] *= 0.1f;
        spdlog::debug("PPOInference: Reduced probability for dangerous straight direction");
    }
    if (basic.dangerLeft) {
        int leftIdx = (static_cast<int>(basic.currentDirection) + 3) % 4;
        actionProbs[leftIdx] *= 0.1f;
        spdlog::debug("PPOInference: Reduced probability for dangerous left direction");
    }
    if (basic.dangerRight) {
        int rightIdx = (static_cast<int>(basic.currentDirection) + 1) % 4;
        actionProbs[rightIdx] *= 0.1f;
        spdlog::debug("PPOInference: Reduced probability for dangerous right direction");
    }
    
    // Increase probability of food-seeking actions
    if (basic.foodUp) {
        actionProbs[0] *= 2.0f;
        spdlog::debug("PPOInference: Increased probability for food-seeking UP action");
    }
    if (basic.foodDown) {
        actionProbs[1] *= 2.0f;
        spdlog::debug("PPOInference: Increased probability for food-seeking DOWN action");
    }
    if (basic.foodLeft) {
        actionProbs[2] *= 2.0f;
        spdlog::debug("PPOInference: Increased probability for food-seeking LEFT action");
    }
    if (basic.foodRight) {
        actionProbs[3] *= 2.0f;
        spdlog::debug("PPOInference: Increased probability for food-seeking RIGHT action");
    }
    
    // Normalize
    float sum = actionProbs[0] + actionProbs[1] + actionProbs[2] + actionProbs[3];
    for (auto& prob : actionProbs) {
        prob /= sum;
    }
    
    spdlog::debug("PPOInference: Fallback probabilities: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]",
                 actionProbs[0], actionProbs[1], actionProbs[2], actionProbs[3]);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<> dist(actionProbs.begin(), actionProbs.end());
    
    int action = dist(gen);
    Direction result = static_cast<Direction>(action);
    
    spdlog::debug("PPOInference: Fallback selected action: {}", action);
    
    return result;
}

// ActorCriticInference Implementation
bool ActorCriticInference::loadModel(const std::string& modelPath) {
    spdlog::info("ActorCriticInference: Attempting to load Actor-Critic model from: {}", modelPath);
    
    if (!isTorchAvailable()) {
        m_lastError = "LibTorch not available in this build";
        spdlog::error("ActorCriticInference: {}", m_lastError);
        return false;
    }
    
#ifdef TORCH_AVAILABLE
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
                m_lastError.clear();
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
        m_lastError = "Failed to load actor model from any path";
        spdlog::error("ActorCriticInference: {}", m_lastError);
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
#else
    m_lastError = "LibTorch not available in this build";
    spdlog::error("ActorCriticInference: {}", m_lastError);
    return false;
#endif
}

std::vector<float> ActorCriticInference::predict(const std::vector<float>& input) {
    if (!isTorchAvailable()) {
        m_lastError = "LibTorch not available in this build";
        spdlog::error("ActorCriticInference: predict() called but LibTorch not available");
        return {};
    }
    
    if (!m_actorLoaded) {
        m_lastError = "Actor model not loaded";
        spdlog::error("ActorCriticInference: {}", m_lastError);
        return {};
    }
    
#ifdef TORCH_AVAILABLE
    try {
        torch::NoGradGuard no_grad;
        
        if (input.size() != 8) {
            m_lastError = "Expected 8D input, got " + std::to_string(input.size()) + "D";
            spdlog::error("ActorCriticInference: {}", m_lastError);
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
        
        m_lastError.clear();
        return result;
        
    } catch (const std::exception& e) {
        m_lastError = "Actor prediction failed: " + std::string(e.what());
        spdlog::error("ActorCriticInference: {}", m_lastError);
        return {};
    }
#else
    m_lastError = "LibTorch not available in this build";
    spdlog::error("ActorCriticInference: {}", m_lastError);
    return {};
#endif
}

Direction ActorCriticInference::getAction(const EnhancedState& state) {
    spdlog::debug("ActorCriticInference: getAction() called");
    
    if (!isLoaded() || !isTorchAvailable()) {
        spdlog::debug("ActorCriticInference: Model not loaded or LibTorch not available, using fallback");
        return fallbackAction(state);
    }
    
    // Use 8D state representation
    auto stateVector = convertToTrainingState(state);
    spdlog::debug("ActorCriticInference: State converted to 8D vector");
    
    auto output = predict(stateVector);
    
    if (output.empty() || output.size() < 4) {
        spdlog::warn("ActorCriticInference: Invalid output, using fallback");
        return fallbackAction(state);
    }
    
    // Extract action probabilities (first 4 elements)
    std::vector<float> actionProbs(output.begin(), output.begin() + 4);
    
    spdlog::debug("ActorCriticInference: Raw action logits: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]",
                 actionProbs[0], actionProbs[1], actionProbs[2], actionProbs[3]);
    
    // Apply softmax to ensure valid probability distribution
    float maxLogit = *std::max_element(actionProbs.begin(), actionProbs.end());
    float sum = 0.0f;
    
    for (auto& prob : actionProbs) {
        prob = std::exp(prob - maxLogit); // Subtract max for numerical stability
        sum += prob;
    }
    
    if (sum <= 0.0f) {
        spdlog::warn("ActorCriticInference: Invalid softmax sum: {:.4f}, using fallback", sum);
        return fallbackAction(state);
    }
    
    for (auto& prob : actionProbs) {
        prob /= sum;
    }
    
    spdlog::debug("ActorCriticInference: Softmax probabilities: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]",
                 actionProbs[0], actionProbs[1], actionProbs[2], actionProbs[3]);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<> dist(actionProbs.begin(), actionProbs.end());
    
    int action = dist(gen);
    Direction result = static_cast<Direction>(action);
    
    spdlog::debug("ActorCriticInference: Selected action: {} (prob: {:.4f})", action, actionProbs[action]);
    
    return result;
}

float ActorCriticInference::getValue(const EnhancedState& state) {
    spdlog::debug("ActorCriticInference: getValue() called");
    
    if (!isTorchAvailable()) {
        spdlog::debug("ActorCriticInference: LibTorch not available, returning 0.0");
        return 0.0f;
    }
    
    if (!m_criticLoaded) {
        spdlog::debug("ActorCriticInference: Critic not loaded, returning 0.0");
        return 0.0f;
    }
    
#ifdef TORCH_AVAILABLE
    try {
        // Use 8D state representation
        auto stateVector = convertToTrainingState(state);
        spdlog::debug("ActorCriticInference: State converted for value estimation");
        
        torch::NoGradGuard no_grad;
        
        torch::Tensor inputTensor = torch::from_blob(
            const_cast<float*>(stateVector.data()), 
            {1, static_cast<long>(stateVector.size())}, 
            torch::kFloat32
        ).to(m_device);
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);
        
        at::Tensor output = m_criticModel.forward(inputs).toTensor();
        output = output.to(torch::kCPU).contiguous();
        
        float value = output.item<float>();
        spdlog::debug("ActorCriticInference: State value: {:.4f}", value);
        
        return value;
        
    } catch (const std::exception& e) {
        spdlog::error("ActorCriticInference: Value prediction failed: {}", e.what());
        return 0.0f;
    }
#else
    spdlog::debug("ActorCriticInference: LibTorch not available, returning 0.0");
    return 0.0f;
#endif
}

Direction ActorCriticInference::fallbackAction(const EnhancedState& state) {
    spdlog::debug("ActorCriticInference: Using fallback value-based action selection");
    
    const auto& basic = state.basic;
    std::array<float, 4> actionValues = {0.5f, 0.5f, 0.5f, 0.5f};
    
    // Penalize dangerous actions
    if (basic.dangerStraight) {
        actionValues[static_cast<int>(basic.currentDirection)] = -1.0f;
        spdlog::debug("ActorCriticInference: Penalized dangerous straight direction");
    }
    if (basic.dangerLeft) {
        int leftIdx = (static_cast<int>(basic.currentDirection) + 3) % 4;
        actionValues[leftIdx] = -1.0f;
        spdlog::debug("ActorCriticInference: Penalized dangerous left direction");
    }
    if (basic.dangerRight) {
        int rightIdx = (static_cast<int>(basic.currentDirection) + 1) % 4;
        actionValues[rightIdx] = -1.0f;
        spdlog::debug("ActorCriticInference: Penalized dangerous right direction");
    }
    
    // Reward food-seeking actions
    if (basic.foodUp) {
        actionValues[0] += 0.8f;
        spdlog::debug("ActorCriticInference: Rewarded food-seeking UP action");
    }
    if (basic.foodDown) {
        actionValues[1] += 0.8f;
        spdlog::debug("ActorCriticInference: Rewarded food-seeking DOWN action");
    }
    if (basic.foodLeft) {
        actionValues[2] += 0.8f;
        spdlog::debug("ActorCriticInference: Rewarded food-seeking LEFT action");
    }
    if (basic.foodRight) {
        actionValues[3] += 0.8f;
        spdlog::debug("ActorCriticInference: Rewarded food-seeking RIGHT action");
    }
    
    // Add exploration noise
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    for (auto& value : actionValues) {
        value += noise(gen);
    }
    
    spdlog::debug("ActorCriticInference: Action values: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]",
                 actionValues[0], actionValues[1], actionValues[2], actionValues[3]);
    
    int bestAction = std::distance(actionValues.begin(), 
                                 std::max_element(actionValues.begin(), actionValues.end()));
    
    Direction result = static_cast<Direction>(bestAction);
    spdlog::debug("ActorCriticInference: Fallback selected action: {} (value: {:.4f})", 
                 bestAction, actionValues[bestAction]);
    
    return result;
}