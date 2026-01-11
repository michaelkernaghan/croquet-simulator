# Ablation Study Design

## Overview

These experiments isolate the contribution of individual components to address reviewer concerns about empirical validation.

---

## Experiment 1: Reward Component Ablations

Test which reward components contribute most to learning.

### Configurations

| Config | Pioneer | Break | Peel | Leave | Hoop | Win/Loss |
|--------|---------|-------|------|-------|------|----------|
| **Full** (baseline) | 2.5 | 1.6 | 1.8 | 0.8 | 10 | ±100 |
| **No Pioneer** | 0 | 1.6 | 1.8 | 0.8 | 10 | ±100 |
| **No Break** | 2.5 | 0 | 1.8 | 0.8 | 10 | ±100 |
| **No Peel** | 2.5 | 1.6 | 0 | 0.8 | 10 | ±100 |
| **Sparse Only** | 0 | 0 | 0 | 0 | 10 | ±100 |
| **Clustering** (naive) | 0 | 0 | 0 | 0 | 10 | ±100 + cluster_reward |

### Metrics to Record
- Win rate vs baseline opponent (every 100 episodes)
- Average hoops per turn
- Pioneer placement rate (balls within 5y of next hoop)
- Break length distribution (1, 2, 3, 4+ consecutive hoops)
- Training loss curve

### Command Template
```bash
python train.py --config ablation_no_pioneer --episodes 2000 --eval_interval 100
python train.py --config ablation_no_break --episodes 2000 --eval_interval 100
python train.py --config ablation_sparse_only --episodes 2000 --eval_interval 100
python train.py --config ablation_clustering --episodes 2000 --eval_interval 100
```

### Hypothesis
- **No Pioneer** should show worst break-building (validates Aiton insight)
- **Sparse Only** should learn slowly but eventually converge
- **Clustering** should show bunching behavior (validates our critique)

---

## Experiment 2: Architecture Ablations

Test dueling architecture and n-step returns contribution.

### Configurations

| Config | Architecture | N-Step | Expected Effect |
|--------|--------------|--------|-----------------|
| **Full** | Dueling | 3 | Best (baseline) |
| **No Dueling** | Standard DQN | 3 | Slightly worse |
| **1-Step** | Dueling | 1 | Slower credit assignment |
| **5-Step** | Dueling | 5 | Higher variance |
| **Vanilla DQN** | Standard | 1 | Worst (baseline comparison) |

### Command Template
```bash
python train.py --architecture standard --n_step 3 --episodes 2000
python train.py --architecture dueling --n_step 1 --episodes 2000
python train.py --architecture dueling --n_step 5 --episodes 2000
python train.py --architecture standard --n_step 1 --episodes 2000
```

### Hypothesis
- Dueling should help in states where action choice matters less
- 3-step should balance bias/variance for croquet's medium horizons
- Vanilla DQN should be competitive but slower to converge

---

## Experiment 3: Expert Influence Decay

Test gradual reduction of expert shaping (proposed in future work).

### Configurations

| Config | Expert Weight Schedule |
|--------|----------------------|
| **Constant** | λ = 1.0 throughout |
| **Linear Decay** | λ = 1.0 → 0.0 over 2000 episodes |
| **Step Decay** | λ = 1.0 (0-1000), 0.5 (1000-1500), 0.0 (1500+) |
| **No Expert** | λ = 0.0 throughout |

### Reward Formula
```
r_total = r_game + λ * r_expert
```

### Command Template
```bash
python train.py --expert_decay none --episodes 2000
python train.py --expert_decay linear --episodes 2000
python train.py --expert_decay step --episodes 2000
python train.py --expert_weight 0.0 --episodes 2000
```

### Hypothesis
- Constant should learn fastest initially
- Linear decay may discover novel strategies late in training
- No expert should eventually converge but much slower

---

## Experiment 4: Baseline Comparisons

Address reviewer concern about weak baselines.

### Opponents to Implement

1. **Random Agent**
   - Uniform random action selection
   - Establishes lower bound

2. **Heuristic Agent** (rule-based)
   ```python
   def select_action(state):
       if can_run_hoop():
           return run_hoop()
       if can_roquet_nearby():
           return roquet_nearest()
       if partner_near_next_hoop():
           return rush_to_partner()
       return defensive_position()
   ```

3. **Greedy Hoop Agent**
   - Always attempts nearest hoop
   - No break-building concept

4. **Self-Play Agent**
   - Train without expert rewards
   - Pure game outcome signal

### Evaluation Matrix

| Our Agent vs | Games | Expected Win Rate |
|--------------|-------|-------------------|
| Random | 100 | >95% |
| Heuristic | 100 | >70% |
| Greedy Hoop | 100 | >80% |
| Self-Play (same episodes) | 100 | >60% |

---

## Experiment 5: Dataset Size Sensitivity

Test how much expert data is needed.

### Configurations

| Config | Training Examples | Turns |
|--------|------------------|-------|
| **Full** | 476 | 395 |
| **50%** | 238 | ~198 |
| **25%** | 119 | ~99 |
| **10%** | 48 | ~40 |
| **None** | 0 | 0 |

### Command Template
```bash
python train.py --expert_data_fraction 1.0 --episodes 2000
python train.py --expert_data_fraction 0.5 --episodes 2000
python train.py --expert_data_fraction 0.25 --episodes 2000
python train.py --expert_data_fraction 0.1 --episodes 2000
python train.py --expert_data_fraction 0.0 --episodes 2000
```

### Hypothesis
- Diminishing returns beyond ~50% of data
- 10% may be sufficient for basic break-building
- Validates whether more tournament data would help

---

## Summary: Minimum Viable Ablations

For paper revision, prioritize these 5 runs (can complete in parallel):

| Priority | Experiment | Purpose |
|----------|------------|---------|
| 1 | Full vs Sparse Only | Validates expert shaping value |
| 2 | Full vs Clustering | Validates clustering critique |
| 3 | Full vs No Pioneer | Validates Aiton insight |
| 4 | Dueling vs Standard DQN | Validates architecture choice |
| 5 | vs Heuristic Agent | Establishes meaningful baseline |

### Estimated Time
- 5 experiments × 2000 episodes × ~30s/episode = ~83 hours total
- Can run 2-3 in parallel depending on GPU memory

---

## Results Template

After running, fill in:

```
| Configuration | Win Rate | Hoops/Turn | Pioneer Rate | Notes |
|---------------|----------|------------|--------------|-------|
| Full          |          |            |              |       |
| Sparse Only   |          |            |              |       |
| Clustering    |          |            |              |       |
| No Pioneer    |          |            |              |       |
| Standard DQN  |          |            |              |       |
```

This table goes directly into paper Section 6 (Results).
