# DQN Training Notes

## Stability Pack v2 (January 2026)

Following analysis of training collapse (ep4000 peak → ep8000 degradation), implemented comprehensive stability fixes based on Peter Lau's recommendations.

### Configuration

**Network Architecture:**
- Double DQN with dueling architecture
- Hidden layers: [256, 128, 64] (standard) or [256, 128] + V/A streams (dueling)
- Dropout: 0.2

**Training Stability:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Loss function | Huber (SmoothL1) | Reduce outlier TD-error impact |
| Gradient clip | 10.0 (global norm) | Prevent gradient explosions |
| LR decay stage 1 | ε ≤ 0.10 → LR × 0.5 | Stabilize as exploration ends |
| LR decay stage 2 | ε ≤ 0.05 → LR × 0.5 | Final stabilization (→2.5e-5) |
| TD error clip | 10.0 | PER priority stability |
| Target updates | τ = 0.005 (soft Polyak) | Stable Q-targets |
| Epsilon floor | 0.05 | Maintain exploration |
| Next-action mask | -1e9 for invalid | Prevent invalid bootstrapping |

**Next-Action Masking (Critical Fix):**
- Before: ~50% of target Q-values bootstrapped from invalid actions
- After: All target argmax selections respect action legality
- Diagnostic: `mask_change_pct` logged every 100 episodes (should decrease as network learns)
- Empty fallback rate should be 0% (non-zero indicates upstream legality bug)

**Exploration (Updated per Peter's feedback on early collapse):**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Epsilon decay | 200,000 steps | Reach floor ~ep1000 (was 10k = ep46!) |
| Replay warmup | 10,000 transitions | Don't train until buffer has variety |
| Epsilon floor | 0.05 | Maintain exploration |

**Replay Buffer:**
- Size: 100,000 transitions
- Min before training: 10,000 (warmup)
- PER optional (α=0.6, β schedule 0.4→1.0 recommended)

### Plateau Detector

Monitors greedy evaluation metrics to detect:
1. **Plateau**: No improvement > 0.05 hoops for 10 evals (1000 episodes)
2. **Collapse**: 30% drop from best for 3 consecutive evals (only if best ≥ 0.5)
3. **Divergence**: Loss > 20 and increasing for 3 evals (after 5000-step warmup)

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Smoothing window | 5 evals |
| Improvement threshold | 0.05 hoops |
| Patience | 10 evals |
| Collapse drop | 30% |
| Collapse K | 3 evals |
| Collapse min best | 0.5 hoops |
| Loss threshold | 20.0 |
| Warmup steps | 5000 |

### Commands

```bash
# Training with stability fixes
python train_neural.py 10000 --eval-freq 100 --save-freq 100 --dueling

# Monitor for plateau/collapse (separate terminal)
python plateau_detector.py --watch

# Evaluate specific checkpoints
python eval_checkpoints.py ep4000 ep5000 --games 20

# Check best checkpoint
python plateau_detector.py --best
```

### Checkpoint History

| Checkpoint | Greedy Avg Hoops | Status |
|------------|------------------|--------|
| ep2500 | 0.5 | Learning |
| ep3000 | 0.9 | Learning |
| ep4000 | **1.1** | **Best** |
| ep5000 | 0.9 | Decline |
| ep8000 | 0.1 | Collapsed |
| ep10000 | 0.2 | Collapsed |

**Current best**: `checkpoint_best.pt` = ep4000

### Troubleshooting

If training still drifts after ε bottoms out:
1. **Reward scale**: Rescale large reward spikes (peg-out +20 may still be high)
2. **PER strength**: Adjust α (priority exponent) or β (IS correction)
3. **Learning rate**: Start lower (5e-5) or add more decay stages
4. **Batch size**: Increase to 128 for more stable gradients

### Phase 2 Options (when plateau detected)

Per Peter Lau's recommendations:
- **2A: Residual shot learning** - Keep intents + heuristics, learn Δpower/Δaim corrections
- **2B: Hierarchical RL** - Intent policy + continuous actor for aim/power
- **2C: Parameterized actions** - Unified discrete + continuous policy

Choose based on plateau symptoms:
- High ROQUET_NEAR% but weak hoop conversion → 2A (shot execution)
- Good execution but poor sequencing → 2B (strategy layer)
- Need end-to-end learning → 2C (clean slate)
