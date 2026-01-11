# Association Croquet Simulator with Neural AI

A physics-based Association Croquet simulator featuring a Deep Q-Network (DQN) AI that learns break-building tactics through reinforcement learning, trained on 2025 World Championship match data.

## Academic Paper

**[Read the Paper (PDF)](docs/paper/croquet_ai_paper.pdf)**

This research explores applying deep reinforcement learning to Association Croquet, encoding expert tactical knowledge from Aiton, Wylie, and Oxford Croquet into a reward function that teaches authentic 4-ball break play.

## Overview

This project implements:
- **Full Association Croquet rules** - 12-hoop course, deadness, continuation strokes, peg-out
- **Physics simulation** - Ball collisions, croquet strokes, hoop running, boundary handling
- **Dueling DQN AI** - Neural network learns tactical play through self-play
- **Expert-informed reward shaping** - Tactical principles encoded from championship-level coaching

## Quick Start

```bash
# Run with visualization
python main.py

# Train from checkpoint
python train_neural.py 5000 --checkpoint ai_data/neural/checkpoint.pt

# Fresh training run
python train_neural.py 5000

# Monitor training for plateau/collapse
python plateau_detector.py --watch
```

## Training Status

**Latest checkpoint**: `ai_data/neural/checkpoint_ep1500.pt`

| Metric | Value |
|--------|-------|
| Greedy Hoops | 20-21 per game |
| Win Rate | 70% |
| Epsilon | 0.05 |
| Pioneer at next hoop | 35% |
| Rush available | 3% |

## Architecture

### Neural Network
- **Dueling DQN** with Double DQN target updates
- State encoder: Ball positions, hoops run, deadness, strokes remaining
- Action space: HOOP_RUN, ROQUET_*, APPROACH, DEFENSIVE, PEG_OUT
- Prioritized experience replay with n-step returns

### Reward Design

The reward function separates **base rewards** (uncapped) from **tactical shaping** (budget-capped at +3/-2 per turn):

| Category | Reward | Notes |
|----------|--------|-------|
| Hoop run | +10 | Sublinear streak bonus |
| Peg out | +20 | Game completion |
| Roquet | +0.5 to +1.5 | Context-dependent |
| Miss | -1.5 | Penalty |
| Pioneer creation | +1.5 | Once per turn |
| Rush creation | +1.2 | Once per turn |
| Pilot creation | +1.0 | Once per turn |

## Project Structure

```
croquet-simulator/
├── main.py                 # Game visualization
├── train_neural.py         # DQN training loop
├── train_alphazero.py      # AlphaZero variant
├── config.py               # Training configuration
├── plateau_detector.py     # Training monitor
│
├── ai/                     # AI components
│   ├── neural/             # DQN implementation
│   ├── ai_controller.py    # Decision making
│   └── break_strategy.py   # Break-building logic
│
├── models/                 # Game state (Ball, Court)
├── physics/                # Physics engine
├── rules/                  # Association Croquet rules
│
├── ai_data/
│   └── neural/             # Saved checkpoints
│
├── docs/
│   ├── paper/              # Academic paper (LaTeX + PDF)
│   └── ablation_experiments.md
│
├── transcripts/            # 2025 World Championship data
│   └── parsed/             # Extracted training examples
│
└── reference_docs/         # Coaching materials
```

## Training Data

Training data extracted from:
- 2025 World Championship match transcripts (VTT format)
- CroquetScores.com game records
- Keith Aiton coaching transcripts

See `docs/aiton_transcript_mapping.md` for reward shaping derivation.

## Documentation

- **[TRAINING_NOTES.md](TRAINING_NOTES.md)** - Stability fixes, hyperparameters, troubleshooting
- **[SPEC.md](SPEC.md)** - Full technical specification
- **[SESSION_STATUS.md](SESSION_STATUS.md)** - Current development status
- **[docs/ablation_experiments.md](docs/ablation_experiments.md)** - Planned experiments

## Roadmap

### In Progress
- [ ] Ablation experiments for paper validation
- [ ] Baseline agent comparisons (Random, Heuristic, Greedy)

### Short-term
- [ ] Improve rush detection (currently 3%)
- [ ] Directional pioneer/pilot detection

### Medium-term
- [ ] Behavior cloning from expert demonstrations
- [ ] Leave quality detection (NSL, OSL patterns)
- [ ] Wiring detection for defensive play

### Long-term
- [ ] Self-play curriculum with opponent modeling
- [ ] Transfer learning to different court sizes
- [ ] Physical croquet robot integration

## References

- Keith Aiton's break-building principles
- Wylie's "Expert Croquet Tactics"
- Oxford Croquet "Rules of Thumb"
- 2025 World Championship match analysis
