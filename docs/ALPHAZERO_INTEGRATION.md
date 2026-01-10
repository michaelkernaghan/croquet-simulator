# AlphaZero Integration in Croquet Simulator

## Project State Summary

The croquet simulator now incorporates key insights from DeepMind's AlphaZero, enabling two complementary training approaches:

1. **Enhanced DQN Training** (`train_neural.py`) - The original DQN approach, now augmented with:
   - Reward shaping annealing (gradual transition to sparse rewards)
   - Self-play training (both sides use neural network)

2. **Full AlphaZero Training** (`train_alphazero.py`) - A new training mode implementing:
   - PolicyValueNetwork with dual heads
   - Monte Carlo Tree Search (MCTS)
   - Self-play with past checkpoint as opponent
   - Policy gradient training from MCTS visit distributions

---

## AlphaZero: Historical Context

### The Journey from AlphaGo to AlphaZero

**AlphaGo (2016)** - DeepMind's first breakthrough in Go:
- Used **supervised learning** from 30 million expert moves
- Combined with reinforcement learning self-play refinement
- Defeated Lee Sedol 4-1, stunning the Go community
- Required domain-specific features and human knowledge

**AlphaGo Zero (October 2017)** - The "tabula rasa" approach:
- **No human games** - learned entirely from self-play
- **No hand-crafted features** - raw board state only
- Surpassed all previous AlphaGo versions in 40 days
- Discovered novel strategies humans had never conceived

**AlphaZero (December 2017)** - Generalization to multiple games:
- Same algorithm for Chess, Shogi, and Go
- Defeated Stockfish (chess) and Elmo (shogi) decisively
- Demonstrated that self-play + search could master any perfect information game

### Key Innovations

| Innovation | AlphaGo | AlphaGo Zero / AlphaZero |
|------------|---------|--------------------------|
| Training data | Expert games | Self-play only |
| Features | Hand-crafted | Raw board state |
| Search | MCTS + policy network | MCTS + policy/value network |
| Reward signal | Win/loss + intermediate | Win/loss only |

---

## How We're Adopting AlphaZero Insights

### 1. From Shaped Rewards to Sparse Rewards

**The Challenge**: Our original DQN used heavily shaped rewards based on croquet expertise (Aiton approach quality, Wylie rush tactics, Oxford Rules of Thumb). This accelerates learning but constrains what the AI can discover.

**AlphaZero Insight**: Sparse rewards (just win/loss) allow the network to discover optimal strategies without human bias.

**Our Implementation**: Reward shaping annealing
```bash
python train_neural.py --anneal-shaping --shaping-start 1.0 --shaping-end 0.1
```

The shaping weight gradually decays from 1.0 (full expert guidance) to 0.1 (mostly sparse), allowing:
- Fast initial learning with expert knowledge
- Eventual discovery of strategies beyond human understanding

### 2. Self-Play Training

**The Challenge**: Originally, one side used the neural network while the opponent used a heuristic. This created asymmetric learning and limited the difficulty curve.

**AlphaZero Insight**: Training against yourself creates a curriculum of progressively harder opponents.

**Our Implementation**: Both sides use neural network
```bash
python train_neural.py --self-play --self-play-opponent past
```

Options:
- `current`: Both sides use same network (true self-play)
- `past`: Opponent uses periodic checkpoint (more stable training)

### 3. PolicyValueNetwork Architecture

**The Challenge**: DQN outputs Q-values for each action. This doesn't integrate well with MCTS, which needs action probabilities and position evaluations.

**AlphaZero Insight**: A dual-head network outputting policy (action probabilities) and value (win probability) works synergistically with MCTS.

**Our Implementation**: New `PolicyValueNet` class
```python
policy_logits, value = network(state)
# policy_logits: [8] action logits -> softmax for probabilities
# value: [-1, 1] expected game outcome
```

### 4. Monte Carlo Tree Search

**The Challenge**: DQN makes single-step decisions based on Q-values. It can't look ahead to evaluate multi-move sequences.

**AlphaZero Insight**: MCTS provides multi-step lookahead, with the neural network guiding exploration and evaluating leaf nodes.

**Our Implementation**: Full MCTS module (`ai/neural/mcts.py`)
- **PUCT selection**: Balances exploitation (Q-value) with exploration (visit count bonus)
- **Neural guidance**: Policy prior from network guides which actions to explore
- **Value evaluation**: Network evaluates leaf positions (no random rollouts)
- **Dirichlet noise**: Added at root for exploration during training

### 5. Policy Gradient Training

**The Challenge**: DQN trains on TD-error (difference between predicted and target Q-values). This can be unstable and doesn't directly optimize action selection.

**AlphaZero Insight**: Train the policy head to match MCTS visit distributions, and the value head to predict game outcomes.

**Our Implementation**: AlphaZero trainer with combined loss
```python
# Policy loss: cross-entropy with MCTS visit distribution
policy_loss = -sum(mcts_policy * log(network_policy))

# Value loss: MSE with game outcome
value_loss = (predicted_value - game_outcome)^2

# Combined loss
total_loss = policy_loss + value_loss
```

---

## Training Mode Comparison

| Aspect | DQN (train_neural.py) | AlphaZero (train_alphazero.py) |
|--------|----------------------|-------------------------------|
| Network | DuelingCroquetNet | PolicyValueNet |
| Action selection | Epsilon-greedy | MCTS |
| Training signal | TD-error | MCTS policy + outcome |
| Rewards | Shaped (optionally annealed) | Sparse only |
| Exploration | Epsilon decay | MCTS + Dirichlet noise |
| Opponent | Configurable (self/heuristic) | Self-play only |

---

## Quick Start

### Option A: Enhanced DQN with AlphaZero Features
```bash
# Full feature set: dueling network, expert shaping with annealing, self-play
python train_neural.py --dueling --expert --anneal-shaping --self-play --episodes 1000
```

### Option B: Pure AlphaZero Training
```bash
# MCTS-based self-play with policy gradient
python train_alphazero.py --episodes 500 --simulations 50
```

### Option C: Gradual Transition
1. Train DQN with expert shaping until convergence
2. Enable annealing to reduce shaping reliance
3. Switch to AlphaZero mode for final refinement

---

## Current Limitations & Future Work

### Limitations
1. **Compute**: AlphaZero used 5000 TPUs. Our MCTS uses 50 simulations vs AlphaZero's 800.
2. **Simplified physics**: MCTS simulator uses heuristic state transitions, not full physics.
3. **No parallel self-play**: Games run sequentially, not in parallel actor-learner setup.

### Future Enhancements
1. **Parallel self-play**: Multiple game actors feeding shared replay buffer
2. **Full physics rollout**: MCTS with actual physics simulation
3. **Residual network**: Deeper network architecture like AlphaZero's ResNet
4. **Temperature scheduling**: Dynamic temperature based on training progress
5. **Arena evaluation**: Periodic evaluation against fixed baselines

---

## References

1. Silver, D., et al. (2016). "Mastering the game of Go with deep neural networks and tree search." *Nature*
2. Silver, D., et al. (2017). "Mastering the game of Go without human knowledge." *Nature*
3. Silver, D., et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." *arXiv*

---

*Last updated: January 2026*
