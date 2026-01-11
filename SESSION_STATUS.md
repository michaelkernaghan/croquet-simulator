# Croquet Simulator - Session Status

**Last Updated**: 2026-01-10

## Project Overview

Association Croquet simulator with DQN-based AI that learns break-building through reinforcement learning. The AI is trained to play authentic 4-ball breaks using tactical knowledge from Keith Aiton, Wylie, and Oxford Croquet references.

## Current State

### Training Status
- **Latest checkpoints**: `ai_data/neural/checkpoint_ep1500.pt` (Jan 10, 17:44)
- **Best historical**: `best_checkpoint_34hoops.pt` (from earlier run)
- **Architecture**: Dueling DQN with Double DQN, n-step returns
- **Training data**: Parsed from 2025 World Championship transcripts

### Recent Work (Jan 10, 2026)
1. Training runs completing faster than expected (<1 hour vs overnight)
2. Paper in progress at `docs/paper/croquet_ai_paper.tex` (PDF generated)
3. Ablation experiments designed in `docs/ablation_experiments.md`
4. Transcript parsing from World Championship videos complete

### Key Metrics (from README)
- Greedy Hoops: 20-21 per game (out of 24)
- Win Rate: 70%
- Epsilon: 0.05 (mostly greedy)
- Pioneer at next hoop: 35%
- Rush available: 3% (needs improvement)

## Project Structure

```
croquet-simulator/
├── train_neural.py         # Main DQN training (81KB, primary file)
├── train_alphazero.py      # AlphaZero variant
├── main.py                 # Game UI/visualization
├── config.py               # Training configuration
├── ai/                     # AI components
│   └── neural/             # DQN implementation
├── ai_data/
│   └── neural/             # Checkpoints saved here
├── docs/
│   ├── paper/              # Academic paper (LaTeX)
│   └── ablation_experiments.md
├── models/                 # Ball, Court, game state
├── physics/                # Physics engine
├── rules/                  # Association Croquet rules
├── scripts/                # Data extraction tools
├── transcripts/            # World Championship VTT files
│   └── parsed/             # Extracted training data
└── reference_docs/         # Coaching materials
```

## Quick Start Commands

```bash
cd "C:/Users/Michael Kernaghan/croquet-simulator"

# Continue training from latest checkpoint
python train_neural.py 5000 --checkpoint ai_data/neural/checkpoint.pt

# Run with visualization
python main.py

# Monitor for plateau/collapse
python plateau_detector.py --watch

# Evaluate checkpoints
python eval_checkpoints.py --games 20
```

## Known Issues / TODO

### Short-term
- [ ] Rush detection only 3% - needs improvement
- [ ] Add directional pioneer/pilot detection (approach angle)
- [ ] Calibrate budget caps as training continues

### Medium-term
- [ ] Run ablation experiments for paper
- [ ] Behavior cloning from expert demos
- [ ] Leave quality detection (NSL, OSL patterns)

### Paper Status
- Draft complete at `docs/paper/croquet_ai_paper.tex`
- Ablation experiments designed, not yet run
- Need baseline comparisons (Random, Heuristic, Greedy agents)

## Training Notes

### Stability Pack v2 (Active)
- Huber loss, gradient clipping at 10.0
- LR decay stages at epsilon 0.10 and 0.05
- Next-action masking for valid actions only
- Epsilon floor: 0.05 for continued exploration

### Historical Collapse (Solved)
- ep4000 was best (1.1 hoops)
- Collapsed by ep8000 (0.1 hoops)
- Fixed with stability pack, next-action masking

## Reward Shaping (from training_data.txt)

Top patterns by average reward:
1. break_continuation: 2.17
2. triple_peel: 1.80
3. break_assembly: 1.51
4. rush_control: 1.43
5. peel_sequence: 1.35

## Data Sources

- 2025 World Championship transcripts (VTT format)
- CroquetScores.com game data
- Aiton transcript mapping in `docs/aiton_transcript_mapping.md`
