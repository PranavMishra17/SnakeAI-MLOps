#include "TorchInference.hpp"
#include <filesystem>
#include <random>

// TorchInference Implementation
bool TorchInference::loadModel(const std::string& modelPath) {
    if (!std::filesystem::exists(modelPath)) {
        spdlog::warn("TorchInference: Model file not found: {}", modelPath);
        return false;
    }
    
    try {
        torch::NoGradGuard no_grad;
        
        m_model = torch::jit::load(modelPath, m_device);
        m_model.eval();
        m_isLoaded = true;
        m_modelPath = modelPath;
        
        spdlog::info("TorchInference: Successfully loaded model from {}", modelPath);
        
        // Test inference with 8D input (matching Python training)
        std::vector<float> testInput(8, 0.0f);
        auto testOutput = predict(testInput);
        
        if (!testOutput.empty()) {
            spdlog::info("TorchInference: Model validation successful, output size: {}", testOutput.size());
        } else {
            spdlog::warn("TorchInference: Model loaded but test inference failed");
        }
        
        return true;
        
    } catch (const c10::Error& e) {
        spdlog::error("TorchInference: LibTorch error loading {}: {}", modelPath, e.what());
        m_isLoaded = false;
        return false;
    } catch (const std::exception& e) {
        spdlog::error("TorchInference: Failed to load model {}: {}", modelPath, e.what());
        m_isLoaded = false;
        return false;
    }
}

std::vector<float> TorchInference::predict(const std::vector<float>& input) {
    if (!m_isLoaded) {
        return {};
    }
    
    try {
        torch::NoGradGuard no_grad;
        
        // Validate input size (should be 8D)
        if (input.size() != 8) {
            spdlog::error("TorchInference: Expected 8D input, got {}D", input.size());
            return {};
        }
        
        // Create tensor from input vector
        torch::Tensor inputTensor = torch::from_blob(
            const_cast<float*>(input.data()), 
            {1, static_cast<long>(input.size())}, 
            torch::kFloat32
        ).to(m_device);
        
        // Forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);
        
        at::Tensor output = m_model.forward(inputs).toTensor();
        
        // Ensure output is on CPU and contiguous
        output = output.to(torch::kCPU).contiguous();
        
        // Convert to vector
        std::vector<float> result;
        float* data_ptr = output.data_ptr<float>();
        result.assign(data_ptr, data_ptr + output.numel());
        
        return result;
        
    } catch (const c10::Error& e) {
        spdlog::error("TorchInference: LibTorch prediction error: {}", e.what());
        return {};
    } catch (const std::exception& e) {
        spdlog::error("TorchInference: Prediction failed: {}", e.what());
        return {};
    }
}

int TorchInference::predictAction(const std::vector<float>& input) {
    auto output = predict(input);
    if (output.empty() || output.size() < 4) {
        return 0; // Default to UP
    }
    
    // Find action with highest value/probability
    auto maxElement = std::max_element(output.begin(), output.begin() + 4);
    return std::distance(output.begin(), maxElement);
}

std::vector<float> TorchInference::predictValues(const std::vector<float>& input) {
    return predict(input);
}

// DQNInference Implementation
Direction DQNInference::getAction(const EnhancedState& state, float epsilon) {
    // Epsilon-greedy exploration
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    
    if (epsilon > 0.0f && dis(gen) < epsilon) {
        std::uniform_int_distribution<> actionDis(0, 3);
        return static_cast<Direction>(actionDis(gen));
    }
    
    if (!isLoaded()) {
        return fallbackAction(state);
    }
    
    // CRITICAL FIX: Use 8D state representation
    auto stateVector = convertToTrainingState(state);
    int action = predictAction(stateVector);
    
    // Clamp action to valid range
    action = std::max(0, std::min(3, action));
    return static_cast<Direction>(action);
}

Direction DQNInference::fallbackAction(const EnhancedState& state) {
    const auto& basic = state.basic;
    std::vector<Direction> safeActions;
    
    // Check safe directions
    if (!basic.dangerStraight) {
        safeActions.push_back(basic.currentDirection);
    }
    if (!basic.dangerLeft) {
        Direction leftDir = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 3) % 4);
        safeActions.push_back(leftDir);
    }
    if (!basic.dangerRight) {
        Direction rightDir = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 1) % 4);
        safeActions.push_back(rightDir);
    }
    
    if (safeActions.empty()) {
        // Last resort - opposite direction
        Direction opposite = static_cast<Direction>((static_cast<int>(basic.currentDirection) + 2) % 4);
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
            return action;
        }
    }
    
    return safeActions[0];
}

// PPOInference Implementation
Direction PPOInference::getAction(const EnhancedState& state) {
    if (!isLoaded()) {
        return fallbackAction(state);
    }
    
    // CRITICAL FIX: Use 8D state representation
    auto stateVector = convertToTrainingState(state);
    auto actionProbs = predict(stateVector);
    
    if (actionProbs.empty() || actionProbs.size() < 4) {
        return fallbackAction(state);
    }
    
    // Ensure probabilities are valid and sum to 1
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        actionProbs[i] = std::max(0.0f, actionProbs[i]); // Clamp to positive
        sum += actionProbs[i];
    }
    
    if (sum <= 0.0f) {
        return fallbackAction(state);
    }
    
    // Normalize probabilities
    for (int i = 0; i < 4; ++i) {
        actionProbs[i] /= sum;
    }
    
    // Sample from probability distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<> dist(actionProbs.begin(), actionProbs.begin() + 4);
    
    return static_cast<Direction>(dist(gen));
}

Direction PPOInference::fallbackAction(const EnhancedState& state) {
    const auto& basic = state.basic;
    std::array<float, 4> actionProbs = {0.25f, 0.25f, 0.25f, 0.25f};
    
    // Reduce probability of dangerous actions
    if (basic.dangerStraight) {
        actionProbs[static_cast<int>(basic.currentDirection)] *= 0.1f;
    }
    if (basic.dangerLeft) {
        int leftIdx = (static_cast<int>(basic.currentDirection) + 3) % 4;
        actionProbs[leftIdx] *= 0.1f;
    }
    if (basic.dangerRight) {
        int rightIdx = (static_cast<int>(basic.currentDirection) + 1) % 4;
        actionProbs[rightIdx] *= 0.1f;
    }
    
    // Increase probability of food-seeking actions
    if (basic.foodUp) actionProbs[0] *= 2.0f;
    if (basic.foodDown) actionProbs[1] *= 2.0f;
    if (basic.foodLeft) actionProbs[2] *= 2.0f;
    if (basic.foodRight) actionProbs[3] *= 2.0f;
    
    // Normalize
    float sum = actionProbs[0] + actionProbs[1] + actionProbs[2] + actionProbs[3];
    for (auto& prob : actionProbs) {
        prob /= sum;
    }
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<> dist(actionProbs.begin(), actionProbs.end());
    
    return static_cast<Direction>(dist(gen));
}

// ActorCriticInference Implementation
Direction ActorCriticInference::getAction(const EnhancedState& state) {
    if (!isLoaded()) {
        return fallbackAction(state);
    }
    
    // CRITICAL FIX: Use 8D state representation
    auto stateVector = convertToTrainingState(state);
    auto output = predict(stateVector);
    
    if (output.empty() || output.size() < 4) {
        return fallbackAction(state);
    }
    
    // Extract action probabilities (first 4 elements)
    std::vector<float> actionProbs(output.begin(), output.begin() + 4);
    
    // Apply softmax to ensure valid probability distribution
    float maxLogit = *std::max_element(actionProbs.begin(), actionProbs.end());
    float sum = 0.0f;
    
    for (auto& prob : actionProbs) {
        prob = std::exp(prob - maxLogit); // Subtract max for numerical stability
        sum += prob;
    }
    
    if (sum <= 0.0f) {
        return fallbackAction(state);
    }
    
    for (auto& prob : actionProbs) {
        prob /= sum;
    }
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<> dist(actionProbs.begin(), actionProbs.end());
    
    return static_cast<Direction>(dist(gen));
}

float ActorCriticInference::getValue(const EnhancedState& state) {
    if (!m_criticLoaded) {
        return 0.0f;
    }
    
    try {
        // CRITICAL FIX: Use 8D state representation
        auto stateVector = convertToTrainingState(state);
        
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
        
        return output.item<float>();
        
    } catch (const std::exception& e) {
        spdlog::error("ActorCriticInference: Value prediction failed: {}", e.what());
        return 0.0f;
    }
}

Direction ActorCriticInference::fallbackAction(const EnhancedState& state) {
    const auto& basic = state.basic;
    std::array<float, 4> actionValues = {0.5f, 0.5f, 0.5f, 0.5f};
    
    // Penalize dangerous actions
    if (basic.dangerStraight) {
        actionValues[static_cast<int>(basic.currentDirection)] = -1.0f;
    }
    if (basic.dangerLeft) {
        int leftIdx = (static_cast<int>(basic.currentDirection) + 3) % 4;
        actionValues[leftIdx] = -1.0f;
    }
    if (basic.dangerRight) {
        int rightIdx = (static_cast<int>(basic.currentDirection) + 1) % 4;
        actionValues[rightIdx] = -1.0f;
    }
    
    // Reward food-seeking actions
    if (basic.foodUp) actionValues[0] += 0.8f;
    if (basic.foodDown) actionValues[1] += 0.8f;
    if (basic.foodLeft) actionValues[2] += 0.8f;
    if (basic.foodRight) actionValues[3] += 0.8f;
    
    // Add exploration noise
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    for (auto& value : actionValues) {
        value += noise(gen);
    }
    
    int bestAction = std::distance(actionValues.begin(), 
                                 std::max_element(actionValues.begin(), actionValues.end()));
    
    return static_cast<Direction>(bestAction);
}