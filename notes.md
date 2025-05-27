# Reinforcement Fine-Tuning Powers Reasoning Capability of Multimodal Large Language Models

An overview of the groundbreaking paper by Sun et al. (2025) that explores how Reinforcement Fine-Tuning (RFT) revolutionizes multimodal reasoning in large language models.

## 📋 Table of Contents

- [Overview](#overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Core Algorithms](#core-algorithms)
- [Community Achievements](#community-achievements)
- [Future Directions](#future-directions)
- [Implementation Examples](#implementation-examples)
- [References](#references)

## 🎯 Overview

This paper positions **Reinforcement Fine-Tuning (RFT)** as the key technology powering reasoning capabilities in Multimodal Large Language Models (MLLMs). Following the success of models like OpenAI-o1 and DeepSeek-R1, the research community has rapidly adopted RFT techniques to enhance reasoning across diverse modalities.

### Key Position Statement
> **"Reinforcement Fine-Tuning (RFT) Powers Reasoning Capability of Multimodal Large Language Models (MLLMs)"**

## 🧮 Mathematical Foundations

### 1. Markov Decision Process (MDP)

The foundation of all RL algorithms is the MDP, formally defined as a tuple:

```
MDP = (S, A, P, R, ρ, γ)
```

Where:
- **S**: State space
- **A**: Action space  
- **P**: Transition function P(s_{t+1}|s_t, a_t)
- **R**: Reward function R(s_t, a_t) ∈ ℝ
- **ρ**: Initial state distribution
- **γ**: Discount factor (0 ≤ γ ≤ 1)

**Agent-Environment Interaction:**
```
At time t: Agent observes s_t → Takes action a_t ~ π(·|s_t) → 
Environment transitions to s_{t+1} ~ P(·|s_t, a_t) → Receives reward R(s_t, a_t)
```

**Objective:** Find optimal policy π* that maximizes cumulative discounted reward:
$$G_t = \sum_{k=t}^{T} \gamma^{k-t} R(s_k, a_k)$$

### 2. Value Functions

#### State Value Function
The expected return starting from state s under policy π:

$$V^π(s) = \mathbb{E}_π[G_t | S_t = s] = \mathbb{E}_π\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s\right]$$

#### Action Value Function (Q-Function)
The expected return taking action a in state s, then following policy π:

$$Q^π(s,a) = \mathbb{E}_π[G_t | S_t = s, A_t = a] = \mathbb{E}_π\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right]$$

### 3. Policy Optimization Algorithms

#### Trust Region Policy Optimization (TRPO)

TRPO solves the constrained optimization problem:

$$\max_θ \mathbb{E}_{s \sim υ^{π_{θ_{old}}}, a \sim π_{θ_{old}}} \left[\frac{π_θ(a|s)}{π_{θ_{old}}(a|s)} A^{π_{θ_{old}}}(s,a)\right]$$

**Subject to:** 
$$\mathbb{E}_{s \sim υ^{π_{θ_{old}}}} [D_{KL}(π_{θ_{old}}(·|s), π_θ(·|s))] ≤ δ$$

Where:
- **Advantage Function:** $A^π(s,a) = Q^π(s,a) - V^π(s)$
- **State Visitation Distribution:** υ^π
- **KL Divergence Constraint:** Ensures new policy stays close to old policy

#### Proximal Policy Optimization (PPO)

PPO simplifies TRPO with two variants:

**PPO-Penalty:**
$$\max_θ \mathbb{E}_{s,a} \left[\frac{π_θ(a|s)}{π_{θ_{old}}(a|s)} A^{π_{θ_{old}}}(s,a) - β D_{KL}(π_{θ_{old}}(·|s), π_θ(·|s))\right]$$

**PPO-Clip (Most Popular):**
$$\max_θ \mathbb{E}_{s,a} \left[\min\left(\frac{π_θ(a|s)}{π_{θ_{old}}(a|s)} A^{π_{θ_{old}}}(s,a), \text{clip}\left(\frac{π_θ(a|s)}{π_{θ_{old}}(a|s)}, 1-ε, 1+ε\right) A^{π_{θ_{old}}}(s,a)\right)\right]$$

**Clipping Mechanism:**
- If $A^{π_{θ_{old}}}(s,a) > 0$ (good action): Ratio clipped at $(1+ε)$
- If $A^{π_{θ_{old}}}(s,a) < 0$ (bad action): Ratio clipped at $(1-ε)$
- Prevents large policy updates

### 4. Multimodal PPO Extension

For MLLMs with input $(m,t)$ (multimodal content + text query):

$$\max_θ \mathbb{E}_{(m,t),a \sim D, \{o_i\}_{i=1}^G \sim π_{θ_{old}}} \left[\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left(r_t \hat{A}_{i,t}^{φ}, \text{clip}(r_t, 1-ε, 1+ε) \hat{A}_{i,t}^{φ}\right)\right]$$

Where:
- $r_t = \frac{π_θ(o_{i,t}|(m,t), o_{i,<t})}{π_{θ_{old}}(o_{i,t}|(m,t), o_{i,<t})}$ (probability ratio)
- $\hat{A}_{i,t}^{φ}$: Generalized Advantage Estimation from critic model $V_φ$

### 5. Group Relative Policy Optimization (GRPO)

GRPO eliminates the critic model by using group-relative rewards:

$$\max_θ \mathbb{E}_{(m,t),a \sim D, \{o_i\}_{i=1}^G \sim π_{θ_{old}}} \left[\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min(r_t \hat{A}_{i,t}, \text{clip}(r_t, 1-ε, 1+ε) \hat{A}_{i,t}) - β D_{KL}(π_θ, π_{ref})_{i,t}\right]$$

**Group Relative Advantage:**
$$\hat{A}_{i,t} = e^{r_i} = \frac{r(o_i, a) - \text{mean}(\{r(o_j, a)\}_{j=1}^G)}{\text{std}(\{r(o_j, a)\}_{j=1}^G)}$$

**Benefits:**
- ✅ No critic model required
- ✅ Lower memory consumption
- ✅ More stable training
- ✅ Faster convergence

## 🔧 Core Algorithms

### Critic-Model-Driven Algorithms

#### PPO for MLLMs
```python
def ppo_mllm_step(model, critic, batch):
    # Forward pass
    logits = model(batch['input'])
    values = critic(batch['input'])
    
    # Compute advantages using GAE
    advantages = compute_gae(values, batch['rewards'])
    
    # Probability ratios
    log_probs = compute_log_probs(logits, batch['actions'])
    old_log_probs = batch['old_log_probs']
    ratios = torch.exp(log_probs - old_log_probs)
    
    # PPO clipped objective
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1-eps, 1+eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    value_loss = F.mse_loss(values, batch['returns'])
    
    return policy_loss + value_loss
```

### Critic-Model-Free Algorithms

#### GRPO Implementation
```python
def grpo_step(model, batch):
    # Sample group of responses
    responses = model.generate(batch['prompts'], num_samples=G)
    
    # Compute rewards
    rewards = [reward_function(resp) for resp in responses]
    
    # Group relative normalization
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    advantages = [(r - mean_reward) / std_reward for r in rewards]
    
    # Compute policy loss
    log_probs = model.compute_log_probs(responses)
    old_log_probs = batch['old_log_probs']
    ratios = torch.exp(log_probs - old_log_probs)
    
    # GRPO objective
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1-eps, 1+eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # KL penalty
    kl_penalty = compute_kl_penalty(model, reference_model)
    
    return policy_loss + beta * kl_penalty
```

## 🏆 Community Achievements

The paper identifies **5 major successes** in applying RFT to MLLMs:

### Success 1: Diverse Modalities
- **Vision**: 50+ papers on visual reasoning
- **Audio**: Audio-Reasoner, R1-AQA, SARI
- **Omni-modal**: R1-Omni, EchoInk-R1
- **GUI Agents**: UI-R1, GUI-R1, InfiGUI-R1
- **3D Spatial**: MetaSpatial for metaverse reasoning

### Success 2: Diverse Tasks & Domains

#### Mathematical Visual Reasoning
- **Core Tasks**: Geometry, algebra, visual math problems
- **Key Models**: InternVL2-MPO, Vision-R1, LMM-R1, VisualPRM
- **Benchmarks**: MathVista, MathVerse, We-Math

#### Academic Multi-discipline
- **Scope**: Physics, chemistry, biology across education levels
- **Models**: MM-EUREKA, Virgo, MMR1
- **Benchmarks**: OlympiadBench, MMMU-Pro

#### Domain-Specific Applications
- **Medical**: MedVLM-R1, ChestX-Reasoner
- **Embodied AI**: Embodied-Reasoner, Embodied-R
- **Video Understanding**: Video-R1, TimeZero, VideoChat-R1

### Success 3: Better Training Algorithms

#### Advanced Training Paradigms
- **Curriculum Learning**: Curr-ReFT with difficulty-aware rewards
- **Online Filtering**: MM-EUREKA eliminates trivial examples
- **Iterative Improvement**: OpenVLThinker uses self-generated data

#### Algorithmic Innovations
- **Dynamic KL**: OThink-MR1 with ε-greedy inspired KL strategy
- **Step-wise Rewards**: StepGRPO addresses sparse reward problem
- **Data Filtering**: ThinkLite-VL uses MCTS-based sample selection

### Success 4: Abundant Benchmarks

The paper identifies **6 exciting trends** in benchmark development:

1. **Increasing Difficulty**: ZeroBench where all current MLLMs fail
2. **Human-like Reasoning**: V1-33K, GeoSense, MM-IQ with IQ tests
3. **Comprehensive Coverage**: MDK12-Bench, MV-MATH, Spatial457
4. **Realistic Applications**: Video-MMLU, GDI-Bench
5. **Visual-Centric Design**: VisuLogic (hard to describe in language)
6. **Interactive Elements**: iVISPAR for spatial reasoning agents

### Success 5: Thriving Engineering Frameworks

#### Key Frameworks
- **Open-R1-Multimodal**: Built on Open-R1 + TRL
- **R1-V**: Supports Qwen2.5-VL + vLLM acceleration
- **EasyR1**: Clean fork of veRL with extensive model support
- **MAYA**: Transparent, educational RL framework

## 🚀 Future Directions

The paper outlines **5 critical research areas**:

### TO DO 1: Better Generalization
**Challenge**: Current models limited to 2-3 tasks in specific domains

**Research Directions**:
- Cross-modality transfer (text + visual → audio + omni-modal)
- Broader task generalization (perception → temporal reasoning)  
- Domain adaptation (general → specialized like medical/embodied)

### TO DO 2: Combine Reward Paradigms
**Outcome Rewards**: High efficiency, sparse feedback
**Process Rewards**: Dense feedback, unstable training

**Integration Strategies**:
- Use outcome rewards to train Process Reward Models (PRMs)
- Develop dense rewards within outcome paradigm (like StepGRPO)
- Explore hybrid approaches across modalities

### TO DO 3: Safety of Reasoning MLLMs
**New Safety Challenges**:
- **Reward Hacking**: Gaming the reward function
- **Jailbreak Attacks**: Exploiting reasoning chains
- **Overthinking**: Unnecessary computational waste

**Research Needs**:
- Advanced detection/defense mechanisms
- Safety-aware reward design
- Reasoning-specific security protocols

### TO DO 4: Data Augmentation for Multimodality
**Current State**: Limited multimodal reasoning data

**Promising Approaches**:
- **NoisyRollout**: Gaussian noise during visual training
- **Diverse Augmentations**: RandomCrop, CenterCrop, RandomAffine
- **Cross-modal Extensions**: Apply visual techniques to audio/video

### TO DO 5: Advanced Algorithms & Applications
**Algorithm Development**:
- Automatic task-specific reward function design
- More efficient training paradigms
- Better optimization techniques

**Broader Applications**:
- Architecture, aerospace, electrical engineering
- Interdisciplinary collaborative research
- Real-world deployment scenarios

## 🔍 Implementation Examples

### Basic GRPO Training Loop
```python
class GRPOTrainer:
    def __init__(self, model, tokenizer, reward_fn):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.ref_model = copy.deepcopy(model)
        
    def train_step(self, prompts, group_size=8):
        # Generate multiple responses
        responses = []
        for prompt in prompts:
            batch_responses = self.model.generate(
                prompt, num_return_sequences=group_size
            )
            responses.append(batch_responses)
        
        # Compute rewards and advantages
        for prompt_responses in responses:
            rewards = [self.reward_fn(resp) for resp in prompt_responses]
            advantages = self.compute_group_relative_advantages(rewards)
            
            # Update policy
            loss = self.compute_grpo_loss(prompt_responses, advantages)
            loss.backward()
            
    def compute_group_relative_advantages(self, rewards):
        mean_r, std_r = np.mean(rewards), np.std(rewards)
        return [(r - mean_r) / (std_r + 1e-8) for r in rewards]
```

### Multimodal Reasoning Pipeline
```python
class MultimodalRFT:
    def __init__(self, vision_encoder, language_model):
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        
    def forward(self, image, text_query):
        # Encode visual information
        visual_features = self.vision_encoder(image)
        
        # Combine with text
        combined_input = self.combine_modalities(visual_features, text_query)
        
        # Generate reasoning chain
        reasoning_chain = self.language_model.generate(
            combined_input, max_length=512
        )
        
        return reasoning_chain
    
    def compute_reasoning_reward(self, response, ground_truth):
        # Rule-based reward for mathematical reasoning
        if self.extract_answer(response) == ground_truth:
            # Bonus for showing work
            if "step" in response.lower() or "because" in response.lower():
                return 1.0 + 0.2  # Process bonus
            return 1.0
        else:
            # Partial credit for correct reasoning steps
            return self.compute_partial_credit(response, ground_truth)
```

## 📊 Mathematical Formulation Summary

| Algorithm | Objective Function | Key Innovation |
|-----------|-------------------|----------------|
| **TRPO** | $\max_θ \mathbb{E}[\frac{π_θ}{π_{old}} A] \text{ s.t. } D_{KL} ≤ δ$ | Trust region constraint |
| **PPO** | $\max_θ \mathbb{E}[\min(r_t A, \text{clip}(r_t, 1±ε) A)]$ | Clipped surrogate objective |
| **GRPO** | $\max_θ \mathbb{E}[\text{PPO-objective}] - β D_{KL}$ | Group-relative advantages |

**Where:**
- $r_t = \frac{π_θ(a|s)}{π_{old}(a|s)}$ (probability ratio)
- $A$: Advantage function
- $D_{KL}$: KL divergence penalty

## 🔗 References

1. **Original Paper**: Sun, H., et al. (2025). "Reinforcement Fine-Tuning Powers Reasoning Capability of Multimodal Large Language Models." *arXiv preprint arXiv:2505.18536*.

2. **Key Models Referenced**:
   - OpenAI-o1: [OpenAI Blog](https://openai.com/o1/)
   - DeepSeek-R1: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
   - Kimi k1.5: [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)

3. **Foundational Papers**:
   - PPO: Schulman et al. (2017) *arXiv:1707.06347*
   - TRPO: Schulman et al. (2015) *ICML*
   - RLHF: Ouyang et al. (2022) *NeurIPS*

4. **Project Repository**: [Awesome-RL-based-Reasoning-MLLMs](https://github.com/Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs)