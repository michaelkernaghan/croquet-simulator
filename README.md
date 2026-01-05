# Association Croquet Simulator with Neural AI

A physics-based Association Croquet simulator featuring a Deep Q-Network (DQN) AI that learns break-building tactics through reinforcement learning.

## Overview

This project implements:
- **Full Association Croquet rules** - 12-hoop course, deadness, continuation strokes, peg-out
- **Physics simulation** - Ball collisions, croquet strokes, hoop running, boundary handling
- **DQN-based AI** - Neural network learns tactical play through self-play
- **Expert tactical shaping** - Reward function encodes croquet principles from Aiton, Wylie, and Oxford Croquet

## Current Training Status

**Episode 1100+ | ~2.9M steps**

### Performance Metrics
- **Greedy Hoops**: 20-21 per game (out of 24 possible per side)
- **Win Rate**: 70%
- **Epsilon**: 0.05 (mostly greedy, 5% exploration)

### Tactical KPIs
- Pilot ball at current hoop: 28%
- Pioneer at next hoop: 35%
- Rush available: 3%
- Cluster quality: 0.42
- Break balls in position: 2.7

### Budget Tracking
- Average tactical budget used: 2.29/turn
- Turns hitting bonus cap (+3): 39%
- Turns hitting penalty cap (-2): 0%

## Architecture

### Neural Network
- **Dueling DQN** with Double DQN target updates
- State encoder: Ball positions, hoops run, deadness, strokes remaining
- Action space: HOOP_RUN, ROQUET_*, APPROACH, DEFENSIVE, PEG_OUT
- Prioritized experience replay with n-step returns

### Reward Function Design

The reward function separates **base rewards** (never capped) from **tactical shaping** (budget-capped):

**Base Rewards:**
- Hoop run: +10 (with sublinear streak bonus)
- Peg out: +20
- Roquet: +0.5 to +1.5 (context-dependent)
- Miss: -1.5
- Rover dawdling: escalating penalty

**Tactical Shaping (capped at +3/-2 per turn):**
- Delta-based rewards for CREATING positions (not maintaining)
- Pilot creation: +1.0 (once per turn)
- Pioneer creation: +1.5 (once per turn)
- Rush creation: +1.2 (once per turn)
- Cluster improvement: continuous
- Opponent separation: phase-gated

**Safety Features:**
- Per-turn shaping budget prevents reward farming
- Hysteresis on thresholds prevents flip-flopping
- Loss penalty caps prevent over-punishment during repositioning
- Once-per-turn caps on binary features

## Key Files

```
train_neural.py          # Main DQN training loop
ai/neural/
  dqn_trainer.py         # DQN implementation with reward function
  croquet_net.py         # Neural network architecture
  replay_buffer.py       # Experience replay implementations
  state_encoder.py       # State representation
ai/
  ai_controller.py       # AI decision making
  break_strategy.py      # Break-building logic
  tactical_planner.py    # LLM-assisted planning (optional)
physics/
  physics_engine.py      # Ball physics simulation
  croquet_strokes.py     # Stroke mechanics
rules/
  rule_engine.py         # Association Croquet rules
models/
  ball.py, court.py      # Game state models
```

## Running Training

```bash
# Continue from checkpoint
python train_neural.py --checkpoint ai_data/neural/checkpoint_final.pt --episodes 5000

# Fresh start
python train_neural.py --episodes 5000

# With LLM tactical planner
python train_neural.py --checkpoint ai_data/neural/checkpoint_final.pt --use-planner
```

## Future Plans

### Short-term
- [ ] Improve rush detection (currently only 3% of positions)
- [ ] Add directional pioneer/pilot detection (approach angle, not just distance)
- [ ] Monitor budget cap calibration as training continues

### Medium-term
- [ ] Behavior cloning pretraining from expert demonstrations
- [ ] Leave quality detection and rewards (NSL, OSL patterns)
- [ ] Wiring detection for defensive play

### Long-term
- [ ] Self-play curriculum with opponent modeling
- [ ] Transfer learning to different court sizes
- [ ] Integration with physical croquet robot

## References

- Keith Aiton's break-building principles
- Wylie's "Expert Croquet Tactics"
- Oxford Croquet "Rules of Thumb"
- MacRobertson Shield tournament analysis

## Training Notes

See `TRAINING_NOTES.md` for detailed session logs and observations.

Key learnings:
1. Delta-based rewards prevent position farming
2. Per-turn budget caps keep hoop rewards dominant
3. Hysteresis prevents threshold flip-flopping
4. Phase-gating prevents conflicting objectives
