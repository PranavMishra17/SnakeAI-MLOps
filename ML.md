# Machine Learning Implementation Guide - SnakeAI-MLOps

## Current Implementation: Tabular Q-Learning

### Mathematical Foundation

**Q-Learning** is a model-free, off-policy temporal difference learning algorithm that learns the optimal action-value function Q*(s,a).

#### Bellman Equation
```
Q*(s,a) = E[R(s,a) + γ max_{a'} Q*(s',a')]
```

#### Q-Learning Update Rule
```
Q(s,a) ← Q(s,a) + α[R + γ max_{a'} Q(s',a') - Q(s,a)]
```

Where:
- `α ∈ (0,1]`: Learning rate
- `γ ∈ [0,1]`: Discount factor  
- `R`: Immediate reward
- `s,s'`: Current and next states
- `a,a'`: Current and next actions

### Current State Representation

**State Space Dimensionality**: 8-dimensional discrete state vector

```cpp
struct AgentState {
    bool dangerStraight;     // Binary collision detection
    bool dangerLeft;         // Binary collision detection  
    bool dangerRight;        // Binary collision detection
    Direction currentDir;    // 4 discrete values {UP, DOWN, LEFT, RIGHT}
    bool foodLeft;           // Binary food direction
    bool foodRight;          // Binary food direction
    bool foodUp;             // Binary food direction
    bool foodDown;           // Binary food direction
};
```

**State Space Size**: 2³ × 4 × 2⁴ = 8 × 4 × 16 = **512 possible states**

**Action Space**: 4 discrete actions {UP, DOWN, LEFT, RIGHT}

**Q-Table Size**: 512 × 4 = **2,048 Q-values**

### Reward Function

Current reward structure implements shaped rewards for faster convergence:

```cpp
R(s,a,s') = {
    +10.0,  if apple eaten
    -10.0,  if death (collision)
    +1.0,   if moving toward food
    -1.0,   if moving away from food
    0.0,    otherwise
}
```

### Exploration Strategy

**ε-greedy Policy**:
```
π(a|s) = {
    1-ε + ε/|A|,  if a = argmax Q(s,a')
    ε/|A|,        otherwise
}
```

**Epsilon Decay**: 
```
ε(t) = ε₀ × decay_rate^t
```
Current: `ε₀ = 0.1`, `decay_rate = 0.995`

### Convergence Properties

**Theoretical Guarantees**: Q-Learning converges to Q* under conditions:
1. All state-action pairs visited infinitely often
2. Learning rate satisfies: `Σα = ∞` and `Σα² < ∞`
3. Rewards are bounded

**Current Implementation Issues**:
- Fixed learning rate (α = 0.1) may not satisfy convergence conditions
- Limited exploration due to simple ε-greedy

---

## Data Structures for Agent Training

### Q-Learning Data Requirements

**State Representation**:
```cpp
struct AgentState {
    bool dangerStraight, dangerLeft, dangerRight;  // 3 bits
    Direction currentDirection;                     // 2 bits (4 values)
    bool foodLeft, foodRight, foodUp, foodDown;    // 4 bits
    // Total: 8 dimensions, 512 discrete states
};
```

**Training Data Format**:
```cpp
struct QLearningTransition {
    std::string stateKey;      // State encoded as string
    int actionIndex;           // 0-3 for UP,DOWN,LEFT,RIGHT
    float reward;              // Immediate reward
    std::string nextStateKey;  // Next state encoded
    bool terminal;             // Episode termination flag
    float epsilon;             // Exploration rate when action taken
};
```

**Q-Table Storage**:
```cpp
std::map<std::string, std::array<float, 4>> qTable;
// Key: state.toString() -> "001203101" (9 chars)
// Value: [Q(s,UP), Q(s,DOWN), Q(s,LEFT), Q(s,RIGHT)]
```

**Training Batch Requirements**:
- Minimum: 100 transitions for stable updates
- Optimal: 1,000+ transitions per training batch
- Experience replay: Store last 10,000 transitions

### Deep Q-Network (DQN) Data Requirements

**State Representation**:
```cpp
struct EnhancedState {
    // Neural network input: 20-dimensional vector
    std::vector<float> features = {
        // Basic danger detection (3)
        dangerStraight, dangerLeft, dangerRight,
        // Direction encoding (4) - one-hot
        dirUp, dirDown, dirLeft, dirRight,
        // Food direction (4)
        foodLeft, foodRight, foodUp, foodDown,
        // Distance features (5)
        distanceToFood, distToWallUp, distToWallDown, 
        distToWallLeft, distToWallRight,
        // Density features (4)
        bodyDensityQ1, bodyDensityQ2, bodyDensityQ3, bodyDensityQ4
    };
};
```

**Training Data Format**:
```cpp
struct DQNTransition {
    std::vector<float> state;      // 20-dimensional state vector
    int action;                    // Action index 0-3
    float reward;                  // Immediate reward
    std::vector<float> nextState;  // Next state vector
    bool terminal;                 // Episode end flag
    float priority;                // For prioritized replay (default 1.0)
    double timestamp;              // For temporal difference
};
```

**Experience Replay Buffer**:
```cpp
class ExperienceReplayBuffer {
    std::deque<DQNTransition> buffer;
    size_t maxCapacity = 100000;     // 100K transitions
    size_t batchSize = 32;           // Training batch size
    
    std::vector<DQNTransition> sampleBatch(size_t size);
    void addTransition(const DQNTransition& transition);
};
```

**Network Architecture Data**:
```cpp
struct DQNModelData {
    std::vector<std::vector<float>> weights;  // Layer weights
    std::vector<float> biases;                // Layer biases
    std::vector<int> layerSizes = {20, 128, 128, 4}; // Input->Hidden->Output
    float learningRate = 0.001f;
    float targetUpdateFreq = 1000;            // Target network update frequency
};
```

**Training Requirements**:
- Minimum buffer size: 10,000 transitions before training
- Batch size: 32-128 transitions per update
- Target network update: Every 1,000 steps
- Training frequency: Every 4 steps

### Policy Gradient Data Requirements

**Episode-Based Collection**:
```cpp
struct PolicyGradientEpisode {
    std::vector<EnhancedState> states;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<float> logProbabilities;  // log π(a|s)
    std::vector<float> advantages;        // A(s,a) = G_t - V(s)
    std::vector<float> returns;           // G_t = Σ γ^k r_{t+k}
    float episodeReturn;                  // Total episode reward
    int episodeLength;
};
```

**Training Data Format**:
```cpp
struct PolicyTransition {
    std::vector<float> state;     // 20-dimensional state
    int action;                   // Action taken
    float reward;                 // Immediate reward
    float logProb;                // log π(a|s) when action taken
    float advantage;              // Advantage estimate A(s,a)
    float return_;                // Discounted return G_t
    bool terminal;                // End of episode flag
};
```

**Batch Training Requirements**:
- Collect full episodes before training
- Batch size: 2,048-8,192 transitions
- Multiple epochs per batch: 4-10 iterations
- Advantage calculation: GAE (λ=0.95) or Monte Carlo

### Actor-Critic Data Requirements

**Dual Network Storage**:
```cpp
struct ActorCriticData {
    // Actor network (policy)
    std::vector<std::vector<float>> actorWeights;
    std::vector<float> actorBiases;
    
    // Critic network (value function)
    std::vector<std::vector<float>> criticWeights;
    std::vector<float> criticBiases;
    
    float actorLearningRate = 0.001f;
    float criticLearningRate = 0.002f;
};
```

**Training Data Format**:
```cpp
struct ActorCriticTransition {
    std::vector<float> state;        // Current state
    int action;                      // Action taken
    float reward;                    // Immediate reward
    std::vector<float> nextState;    // Next state
    bool terminal;                   // Episode termination
    float valueEstimate;             // V(s) from critic
    float nextValueEstimate;         // V(s') from critic
    float tdError;                   // δ = r + γV(s') - V(s)
    float logProbability;            // log π(a|s) from actor
};
```

**Training Requirements**:
- Online training: Update after each step
- Critic target: r + γV(s') for non-terminal, r for terminal
- Actor update: Policy gradient with advantage
- Learning rates: Actor < Critic (typically 1:2 ratio)

### Genetic Algorithm Data Requirements

**Population Structure**:
```cpp
struct GeneticIndividual {
    std::vector<std::vector<float>> neuralWeights;  // Network weights
    std::vector<float> biases;                      // Network biases
    float fitness;                                  // Performance score
    int gamesPlayed;                               // Evaluation count
    float averageScore;                            // Average game score
    std::vector<int> gameScores;                   // Individual game results
};

struct GeneticPopulation {
    std::vector<GeneticIndividual> individuals;
    int generation;
    int populationSize = 50;
    float mutationRate = 0.1f;
    float crossoverRate = 0.7f;
    float eliteRatio = 0.2f;                       // Top 20% survive
};
```

**Evaluation Data**:
```cpp
struct GeneticEvaluation {
    int individualId;
    int generation;
    std::vector<int> gameScores;     // Scores from multiple games
    float meanScore;                 // Average performance
    float scoreVariance;             // Consistency measure
    int totalSteps;                  // Total steps across all games
    float efficiency;                // Score per step ratio
};
```

**Training Requirements**:
- Population size: 50-200 individuals
- Evaluation games: 10-50 games per individual
- Selection pressure: Top 20-50% survive
- Mutation: Gaussian noise (σ=0.1) on weights

### Unified Data Collection Format

**Universal Transition Structure**:
```cpp
struct UnifiedTransition {
    // Episode metadata
    int episode;
    int step;
    std::chrono::milliseconds timestamp;
    
    // State representations (multiple formats)
    AgentState basicState;           // 8D discrete for Q-Learning
    EnhancedState enhancedState;     // 20D continuous for neural nets
    std::vector<float> rawState;     // Raw state vector
    
    // Action information
    int actionIndex;                 // 0-3 action encoding
    Direction actionDirection;       // Enum action
    
    // Reward and termination
    float reward;                    // Immediate reward
    bool terminal;                   // Episode end flag
    
    // Agent-specific data
    float logProbability;            // For policy gradient methods
    float valueEstimate;             // For actor-critic methods
    float qValue;                    // For Q-learning methods
    float epsilon;                   // Exploration rate
    
    // Training metadata
    AgentType agentType;
    float learningRate;
    int batchId;                     // Training batch identifier
};
```

**Multi-Agent Training Pipeline**:
```cpp
class UnifiedDataCollector {
    // Storage for different agent types
    std::vector<QLearningTransition> qLearningBuffer;
    ExperienceReplayBuffer dqnBuffer;
    std::vector<PolicyGradientEpisode> policyEpisodes;
    std::vector<ActorCriticTransition> acBuffer;
    GeneticPopulation geneticPopulation;
    
    // Conversion methods
    QLearningTransition toQLearning(const UnifiedTransition& trans);
    DQNTransition toDQN(const UnifiedTransition& trans);
    PolicyTransition toPolicyGradient(const UnifiedTransition& trans);
    ActorCriticTransition toActorCritic(const UnifiedTransition& trans);
};
```

### Training Data Persistence

**File Formats by Agent Type**:

**Q-Learning**: JSON format
```json
{
  "metadata": {"episodes": 1000, "totalSteps": 50000},
  "qTable": {
    "00120310": [0.5, -0.2, 0.8, 0.1],
    "01021130": [0.3, 0.7, -0.1, 0.4]
  },
  "hyperparameters": {"lr": 0.1, "gamma": 0.95, "epsilon": 0.05}
}
```

**Neural Networks**: Binary format + metadata
```cpp
struct ModelCheckpoint {
    std::vector<float> flattenedWeights;  // All weights serialized
    std::vector<int> layerSizes;          // Network architecture
    float trainingLoss;                   // Current loss value
    int trainingSteps;                    // Steps trained
    float validationScore;                // Performance metric
};
```

**Training Data Requirements Summary**:

| Agent Type | Min. Data | Optimal Data | Update Frequency | Memory Usage |
|------------|-----------|--------------|------------------|--------------|
| Q-Learning | 100 transitions | 10K transitions | Every step | ~1-10 MB |
| DQN | 10K transitions | 100K transitions | Every 4 steps | ~100-500 MB |
| Policy Gradient | 10 episodes | 100 episodes | Per episode | ~50-200 MB |
| Actor-Critic | 1 transition | Continuous | Every step | ~10-100 MB |
| Genetic Algorithm | 500 evaluations | 5K evaluations | Per generation | ~10-50 MB |

---

## Alternative Reinforcement Learning Approaches

### 1. Deep Q-Networks (DQN)

#### Mathematical Foundation

Replace Q-table with neural network Q(s,a;θ) parameterized by θ.

**Loss Function**:
```
L(θ) = E[(y - Q(s,a;θ))²]
```

Where target value:
```
y = r + γ max_{a'} Q(s',a';θ⁻)
```

θ⁻ represents target network parameters (updated periodically).

#### Architecture for Snake

**Input Layer**: Raw game state representation
- Grid state: 20×20×3 tensor (empty, snake, food)
- Snake head position: (x,y) coordinates  
- Snake direction: one-hot encoded vector
- **Input dimension**: 20×20×3 + 2 + 4 = 1,206

**Hidden Layers**:
```
Conv2D(32, 3×3) → ReLU → 
Conv2D(64, 3×3) → ReLU →
Conv2D(64, 3×3) → ReLU →
Flatten() →
Dense(512) → ReLU →
Dense(256) → ReLU →
Dense(4)  # Output layer
```

**Data Requirements**:
- **Training samples**: ~10⁶ - 10⁷ state transitions
- **Memory buffer**: ~10⁵ - 10⁶ experiences for replay
- **Training time**: 10-50 hours on GPU
- **Convergence**: 10,000 - 100,000 episodes

#### DQN Improvements

**Double DQN**: Reduces overestimation bias
```
y = r + γ Q(s', argmax_{a'} Q(s',a';θ), θ⁻)
```

**Dueling DQN**: Separate value and advantage streams
```
Q(s,a;θ) = V(s;θ) + A(s,a;θ) - 1/|A| Σ_{a'} A(s,a';θ)
```

**Prioritized Experience Replay**: Sample experiences by TD-error magnitude

### 2. Policy Gradient Methods

#### REINFORCE Algorithm

**Objective**: Maximize expected cumulative reward
```
J(θ) = E_{τ~π_θ}[R(τ)]
```

**Policy Gradient Theorem**:
```
∇_θ J(θ) = E_{τ~π_θ}[Σ_t ∇_θ log π_θ(a_t|s_t) G_t]
```

Where G_t is the return from time t.

**Update Rule**:
```
θ ← θ + α ∇_θ J(θ)
```

#### Actor-Critic Methods

**Value Function Approximation**:
```
V^π(s) ≈ V(s;w)
```

**Policy Parameterization**:
```
π(a|s) ≈ π(a|s;θ)
```

**Actor Update**:
```
θ ← θ + α_θ ∇_θ log π_θ(a|s) δ
```

**Critic Update**:
```
w ← w + α_w δ ∇_w V(s;w)
```

**TD Error**:
```
δ = r + γV(s';w) - V(s;w)
```

### 3. Proximal Policy Optimization (PPO)

#### Clipped Objective Function

```
L^CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

Where:
- `r_t(θ) = π_θ(a_t|s_t)/π_θ_old(a_t|s_t)` (probability ratio)
- `Â_t`: Advantage estimate
- `ε`: Clipping parameter (typically 0.2)

**Data Requirements for Snake**:
- **Batch size**: 2,048 - 8,192 transitions
- **Training iterations**: 5,000 - 20,000
- **Episodes**: 50,000 - 200,000
- **Training time**: 5-20 hours

### 4. Advanced Methods

#### Soft Actor-Critic (SAC)

**Entropy-Regularized Objective**:
```
J(π) = E_{τ~π}[Σ_t r(s_t,a_t) + αH(π(·|s_t))]
```

**Q-Function Updates** (twin critics to reduce overestimation):
```
L(θ_i) = E[(y - Q_θ_i(s,a))²]
y = r + γ(min_{j=1,2} Q_θ'_j(s',a') - α log π_φ(a'|s'))
```

#### Multi-Agent Reinforcement Learning

**Self-Play Training**:
- Train multiple agents simultaneously
- Each agent learns against previous versions
- Enables complex emergent behaviors

**Population-Based Training**:
- Maintain diverse population of agents
- Select and mutate successful strategies
- Prevents convergence to local optima

---

## State Space Design Alternatives

### Current State (8D Discrete)
**Pros**: Fast training, interpretable, small memory footprint  
**Cons**: Limited expressiveness, no spatial relationships

### Raw Grid State (20×20×C)
**Channels**: 
- C=3: {empty, snake, food}
- C=4: Add direction channel
- C=5: Add distance fields

**Pros**: Complete information, spatial awareness  
**Cons**: Large state space, requires CNN, slow training

### Engineered Features (High-Dimensional)
```cpp
struct EnhancedState {
    // Current features (8D)
    bool danger[3];
    Direction currentDir;
    bool food[4];
    
    // Additional features (12D)
    float distanceToFood;        // Euclidean distance
    float distanceToWall[4];     // Distance in each direction
    float bodyLength;            // Snake length
    float pathToFood;            // A* path length
    bool pathBlocked;            // Whether direct path exists
    float bodyDensity[4];        // Body segment density by quadrant
    float recentMoves[4];        // Movement history
};
```

**State Space**: ~20-30 dimensions (continuous/discrete mix)

### Attention-Based State

**Spatial Attention**: Focus on relevant grid regions
```
attention_weights = softmax(Q(grid_patches) · K(head_position))
state = Σ attention_weights · V(grid_patches)
```

---

## Training Methodologies

### 1. Supervised Learning Approach

**Human Demonstration Data**:
- Collect expert gameplay episodes
- Extract (state, action) pairs
- Train policy via behavioral cloning

**Loss Function**:
```
L(θ) = -1/N Σ_i log π_θ(a_i|s_i)
```

**Data Requirements**:
- 10,000 - 100,000 expert transitions
- Multiple skill levels for robust learning
- Diverse scenarios (different food positions)

### 2. Imitation Learning

#### Dataset Aggregation (DAgger)

**Algorithm**:
1. Train policy π₁ on expert data D₁
2. Run π₁, query expert for labels → D₂  
3. Train π₂ on D₁ ∪ D₂
4. Repeat until convergence

**Benefits**: Addresses distribution shift problem

#### Generative Adversarial Imitation Learning (GAIL)

**Discriminator Loss**:
```
L_D = -E_{(s,a)~π_expert}[log D(s,a)] - E_{(s,a)~π_θ}[log(1-D(s,a))]
```

**Generator Loss**:
```
L_G = -E_{(s,a)~π_θ}[log D(s,a)]
```

### 3. Curriculum Learning

#### Progressive Difficulty

**Stage 1**: Small grid (10×10), static food
**Stage 2**: Medium grid (15×15), moving food  
**Stage 3**: Full grid (20×20), multiple foods
**Stage 4**: Obstacles, time pressure

**Transfer Learning**: Initialize new stage with previous stage weights

#### Self-Paced Learning

**Automatic Difficulty Adjustment**:
```
difficulty(t) = f(success_rate, episode_length, score_variance)
```

### 4. Multi-Objective Optimization

**Objective Function**:
```
J(θ) = w₁·J_score(θ) + w₂·J_length(θ) + w₃·J_efficiency(θ)
```

Where:
- J_score: Average game score
- J_length: Average episode length  
- J_efficiency: Food collection rate

---

## Evaluation Metrics

### Performance Metrics

**Primary Metrics**:
- Average score per episode
- Success rate (reaching score threshold)
- Episode length distribution
- Sample efficiency (performance vs. training time)

**Advanced Metrics**:
- Regret bounds: `Regret(T) = T·V* - Σ_t V^π_t`
- Sample complexity: Episodes to reach 90% optimal performance
- Generalization: Performance on unseen initial conditions

### Statistical Analysis

**Learning Curves**: Plot with confidence intervals
```
μ(t) ± 1.96·σ(t)/√n
```

**Significance Testing**: Mann-Whitney U test for algorithm comparison

**Effect Size**: Cohen's d for practical significance
```
d = (μ₁ - μ₂)/σ_pooled
```

---

## Implementation Considerations

### Computational Requirements

| Method | Memory | Training Time | Inference Time |
|--------|--------|---------------|----------------|
| Q-Learning | ~10 KB | 1-10 min | <1ms |
| DQN | ~100 MB | 5-50 hours | 1-10ms |
| PPO | ~50 MB | 2-20 hours | 1-5ms |
| SAC | ~100 MB | 10-100 hours | 5-20ms |

### Hyperparameter Sensitivity

**Critical Parameters**:
- Learning rate: α ∈ [10⁻⁵, 10⁻²]
- Discount factor: γ ∈ [0.9, 0.999]
- Exploration: ε ∈ [0.01, 0.3]
- Network architecture: Depth, width, activation functions

**Hyperparameter Optimization**:
- Grid search for <5 parameters
- Random search for >5 parameters  
- Bayesian optimization for expensive evaluations
- Population-based training for adaptive search

### Reproducibility Requirements

**Random Seed Control**:
```cpp
std::mt19937 rng(seed);
torch::manual_seed(seed);  // If using PyTorch
```

**Environment Determinism**:
- Fixed initial conditions
- Deterministic physics
- Reproducible random sequences

---

## Advanced Research Directions

### 1. Meta-Learning

**Model-Agnostic Meta-Learning (MAML)**:
Learn initialization that quickly adapts to new tasks
```
θ* = argmin_θ E_{T~p(T)}[L_{T}(f_{θ-α∇L_T(f_θ)})]
```

### 2. Hierarchical Reinforcement Learning

**Options Framework**:
- Macro-actions spanning multiple time steps
- Sub-policies for navigation, hunting, escaping
- Temporal abstraction for efficient learning

### 3. Model-Based Methods

**Forward Model Learning**:
```
ŝ_{t+1} = f_φ(s_t, a_t)
```

**Planning with Learned Models**:
- Model Predictive Control (MPC)
- Monte Carlo Tree Search (MCTS)
- Dyna-Q integration

### 4. Neurosymbolic Approaches

**Logic-Guided Learning**:
- Encode game rules as logical constraints
- Integrate with neural policy learning
- Improved sample efficiency and interpretability

---

## Recommended Implementation Sequence

### Phase 1: Enhanced Q-Learning (1-2 weeks)
1. Implement experience replay buffer
2. Add double Q-learning
3. Improve state representation (engineered features)
4. Hyperparameter optimization

### Phase 2: Deep Q-Network (2-4 weeks)  
1. Raw grid state representation
2. Convolutional neural network
3. Target network and replay buffer
4. Performance comparison with tabular Q-learning

### Phase 3: Policy Gradient Methods (3-6 weeks)
1. REINFORCE implementation
2. Actor-Critic with baseline
3. PPO for stable training
4. Multi-objective optimization

### Phase 4: Advanced Techniques (4-8 weeks)
1. Self-play training
2. Curriculum learning
3. Meta-learning for rapid adaptation
4. Comprehensive evaluation and analysis

**Total Estimated Timeline**: 10-20 weeks for complete implementation

---

## References & Further Reading

**Foundational Papers**:
- Watkins & Dayan (1992): Q-Learning convergence proof
- Mnih et al. (2015): Deep Q-Networks  
- Schulman et al. (2017): Proximal Policy Optimization
- Haarnoja et al. (2018): Soft Actor-Critic

**Implementation Resources**:
- Stable-Baselines3: Production-ready RL algorithms
- OpenAI Gym: Standard evaluation environments
- Ray RLlib: Distributed RL training framework

**Snake-Specific Studies**:
- Analysis of optimal Snake policies
- State representation effectiveness studies
- Multi-agent Snake environments