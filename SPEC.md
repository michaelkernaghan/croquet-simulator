# Association Croquet Simulator Specification

## Overview

A self-learning Association Croquet simulator that trains an AI to play at expert level through self-play, with comprehensive visualization tools for post-game analysis.

## Core Architecture

### Current Structure (Preserved)
```
croquet-simulator/
├── ai/                    # AI decision making and learning
│   └── learning/          # Self-play learning components
├── models/                # Ball, Court, Hoop models
├── physics/               # Physics engine, collision detection
├── rules/                 # Game rules enforcement
├── view/                  # Pygame rendering
└── ai_data/               # Persisted learning state
```

## Feature Dependency Graph

```
                    ┌─────────────────────┐
                    │   Core Mechanics    │
                    │  (Ball, Court, Physics) │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │ Yard Line     │  │ Wire Detection│  │ Ball Selection│
    │ Placement     │  │ (Hoop/Peg)    │  │ (Same-color)  │
    └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Risk Threshold    │
                    │     Learning        │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │ Leave Pattern │  │ Per-Hoop      │  │ Stroke Type   │
    │ Bootstrap     │  │ Strategies    │  │ Learning      │
    └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    MCTS Planner     │
                    │  (Action Abstraction)│
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │ Multi-Ball    │  │ Headless      │  │ Event         │
    │ Actions       │  │ Training      │  │ Recording     │
    └───────────────┘  └───────┬───────┘  └───────┬───────┘
                               │                  │
                               └────────┬─────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │  Post-Game Analysis │
                              │   Visualization     │
                              └─────────────────────┘
```

---

## Physics & Game Rules

### Boundary Collision (Implemented)
- Balls do NOT bounce off boundaries
- When a ball goes out of bounds, it STOPS and is placed on the yard line (1 yard from boundary)
- Position is where the ball exited

### Yard Line Placement (To Implement)
**Approach**: Full evaluation of all valid yard line positions using position evaluator

When a ball goes out of bounds:
1. Determine all valid yard line positions per croquet rules
2. Score each position using position evaluator weights
3. AI chooses strategically optimal placement
4. Consider: proximity to partner, distance from opponents, approach angles to hoops

### Wire Detection (Medium Priority)
**Scope**: Hoop uprights and peg only (not ball-on-ball blocking)

- Check line-of-sight from striker to target ball
- Detect if hoop legs or peg obstruct the direct path
- Set `is_wired` flag in PositionFeatures when blocked
- Wire detection affects shot selection and position evaluation

### Tunneling Detection
**Approach**: Monitor and fix if detected

- Log suspicious cases where ball appears to pass through hoop without triggering run
- At MAX_SHOT_POWER=15 yards/s and 60fps, movement is 0.25 yards/frame (should be safe)
- If issues detected, implement swept collision checking

### Rush Physics (Current is Good)
- RESTITUTION = 0.8 provides good balance
- Near-perfect energy transfer for realistic rushes
- Striker ball retains some momentum (doesn't fully stop)

---

## AI Learning System

### Risk Threshold Learning
**Approach**: Learn when to continue vs. take defensive leave

Risk factors to consider (weighted combination):
1. Deadness accumulation (not primary, one of several)
2. Distance to remaining live balls
3. Opponent ball positions
4. Quality of available approach angles
5. Current break length (don't abandon successful breaks)

### Play Style Variance
**Approach**: Randomize per game

- Assign different play style weights to teams each game
- Creates diverse training situations
- Prevents convergence to single strategy
- Styles: Aggressive, Defensive, Balanced, Position-focused

### Ball Selection (To Implement)
**Approach**: Same-color pair choice, extend position evaluator

Each turn, AI chooses between its two balls:
- Add "turn potential" score to position evaluator
- Consider: proximity to target hoop, available roquets, break continuation potential
- Learn which ball selection leads to better outcomes

### Stroke Type Learning
**Approach**: Hybrid - geometry suggests, preferences break ties

- Primary selection based on where balls need to go
- Learn success rate modifiers for each stroke type
- Track: stop shot, drive, half roll, full roll, pass roll, take-off, split
- Apply learned preferences when multiple strokes would achieve similar geometry

### Per-Hoop Strategy Learning
**Approach**: Modifiers to base weights

- Base position evaluator weights apply to all hoops
- Learn per-hoop adjustment factors (12 sets of small modifiers)
- Stored efficiently: `base_weight * (1 + hoop_modifier[hoop_num])`
- Allows hoops with different difficulties to develop distinct approaches

### Approach Pattern Learning (Enhanced)
**Approach**: Direct targeting based on learned optima

Current tracking:
- Distance to hoop at attempt
- Approach angle quality (0-1)
- Success/failure outcome
- Shot power used

Enhancement:
- When approaching a hoop, actively target learned optimal distance/angle
- Adjust aim to achieve optimal approach position
- Different optima per hoop based on accumulated data

### Learning Rate & Stability
**Approach**: Add momentum to prevent oscillation

- Implement exponential moving average for weight updates
- New weight = (1 - momentum) * current + momentum * update
- Suggested momentum: 0.9 (slow adaptation, stable convergence)
- Allows unbounded weight drift but smooths trajectory

### Weight Bounds
**Approach**: Unbounded learning

- No hard constraints on weight values
- Trust the learning process
- Extreme weights indicate strong signals from data

---

## Training Infrastructure

### Headless Training Mode
**Approach**: No-render mode for maximum throughput

- Disable pygame rendering entirely
- Run physics simulation only
- Maximum games per second for training

### Batch Learning
**Approach**: Aggregate experiences from N games, then update

- Collect experiences from batch of games (suggested: 10-50 games)
- Aggregate statistics before weight updates
- More stable learning than per-experience updates
- Reduces noise from individual game variance

### Checkpointing
**Approach**: Regular interval + milestone saves

Triggers:
- Every N games (suggested: 100 games)
- When new best metrics achieved:
  - New best break length
  - New high win rate (over window)
  - New high hoop success rate

Checkpoint content:
- All position evaluator weights
- Per-hoop modifiers
- Learning statistics
- Approach pattern data
- Timestamp and game count

### Bias Checking
**Issue**: Blue/black won all 6 recorded games

Action: Randomize starting baulk assignments
- Each game, randomly assign which team starts from A-baulk vs B-baulk
- Track win rates per starting position
- Alert if persistent imbalance after randomization

### Hoop 1 Tracking Bug
**Issue**: 0% success rate with 447 attempts is suspicious

Investigation needed:
- Review when hoop attempts are incremented
- Likely counting every shot when hoop 1 is target, not actual run attempts
- Fix: Only count when ball is within reasonable approach distance/angle

---

## MCTS Planning

### Action Abstraction
**Approach**: High-level actions with parameter refinement

**Comprehensive Action Vocabulary**:

Core Actions:
- `RUSH` - Hit ball to send it in specific direction
- `APPROACH_HOOP` - Position to run target hoop
- `PIONEER_PLACE` - Send croqueted ball to future position
- `DEFENSIVE_SHOT` - Safe position, give nothing away

Croquet Stroke Actions:
- `STOP_SHOT` - Croqueted goes far, striker stays
- `ROLL_SHOT` - Both balls move similar distance
- `TAKE_OFF` - Striker moves far, minimal croqueted movement
- `SPLIT_SHOT` - Balls diverge to different positions

Advanced Actions (Medium Priority):
- `CANNON` - Strike to hit two balls
- `PEEL` - Push partner through their hoop
- `PROMOTION` - Position partner ball for peeling

Each action has parameters (target position, power) refined after selection.

### MCTS Implementation
- Tree search with action abstraction at decision nodes
- Simulation/rollout uses simplified evaluation
- Track visit counts and value estimates per action
- Balance exploration vs exploitation

---

## Leave Patterns

### Bootstrap Approach
**Method**: Hybrid - start with standards, allow refinement

Standard Leaves to Implement:
1. **North-South Leave (NSL)**: Balls on north and south boundaries
2. **Old Standard Leave (OSL)**: Classic defensive positioning
3. **Diagonal Spread**: Balls in opposite corners

Features:
- Define target positions for each leave pattern
- Score how well current position matches pattern
- AI learns when each leave is appropriate
- Allow AI to discover variants through play

---

## Visualization & Analysis

### Post-Game Analysis Mode
**Approach**: Record games, analyze with full toolset afterward

Live mode: Simple rendering, no overlays
Analysis mode: Full suite of tools on recordings

### Event-Based Recording
**Format**: Minimal + extensible schema

Core fields:
```json
{
  "event_type": "shot|collision|hoop_run|boundary",
  "timestamp": 1234,
  "turn_number": 5,
  "ball_positions": {"blue": [x, y], ...},
  "shot_vector": [angle, power],
  "outcome": "roquet|hoop|miss|out_of_bounds",
  "resulting_positions": {"blue": [x, y], ...}
}
```

Extensible: Add fields as needed, ignore unknown fields in reader.

### Visualization Tools
All available in post-game analysis:

1. **Shot Predictions**: Show planned trajectory, expected final positions
2. **Decision Overlay**: Why AI chose this shot - rules applied, scores
3. **Heat Maps**: Hoop success zones, common positions, danger areas
4. **Break Analysis**: Highlight successful break sequences

### Audio Feedback
**Approach**: Simple effects for key events

Sounds for:
- Ball collision (volume scaled by impact force)
- Hoop run (satisfying click/ding)
- Boundary contact
- Shot power indicator

Lower priority but improves observation experience.

---

## Performance Goals

### AI Skill Target
**Goal**: Expert level play

Metrics:
- Regular 6+ hoop breaks
- Consistent break-building
- Effective leave positioning
- Strategic ball selection

### Learning Success Criteria

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Avg break length | 0.38 hoops | 4+ hoops | `break_stats.avg_break_length` |
| Best break | 3 hoops | 12 hoops (full round) | `break_stats.best_break` |
| Hoop success rate (avg) | ~20% | 60%+ | `hoop_success_rates` average |
| Games to convergence | - | <1000 | Weights stabilize |

### Future Enhancement: Triple Peels
**Status**: Future enhancement, not in initial scope

- Get solid single-ball breaks working first
- Peel mechanics require precise multi-ball planning
- Add after MCTS and break-building are robust

---

## Technical Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Yard line placement | Full evaluation | Strategic AI should optimize all choices |
| Wire detection | Hoop/peg only | Traditional definition, simpler geometry |
| Ball selection | Same-color pair | Authentic rules, adds decision complexity |
| Risk learning | Multi-factor | Deadness is one signal among several |
| Play styles | Randomize per game | Creates diverse training scenarios |
| Learning momentum | Add momentum | Prevents oscillation, smoother convergence |
| Weight bounds | Unbounded | Trust the data, extreme values are signals |
| Recording | Event-based | Compact, extensible, sufficient for analysis |
| Search method | MCTS with action abstraction | Enables multi-shot planning with manageable branching |
| Leave patterns | Hybrid bootstrap | Standards provide foundation, learning refines |
| Training mode | Headless option | Maximum throughput for learning |
| Batch size | N games | Stable updates, aggregate statistics |
| Checkpoints | Interval + milestones | Regular saves plus performance peaks |
| Starting positions | Randomize | Eliminate potential positional bias |
| Parallelization | Profile first | Measure bottlenecks before optimizing |
| Audio | Simple effects | Enhances observation, lower priority |
| Visualization | Post-game analysis | Full toolset on recordings, not live |
| Code structure | Current is good | Clear separation, easy to navigate |

---

## Known Issues to Address

1. **Hoop 1 tracking bug**: 0% success with 447 attempts - fix counting logic
2. **Blue/black win bias**: All 6 games won - randomize starting positions
3. **Break continuation**: AI should prioritize roquets over passive setup
4. **Power calibration**: Croquet shots and rushes needed more power (addressed)

---

## Version History

- **v0.1**: Initial specification based on interview
- Date: 2024
- Interviewer responses incorporated for all design decisions
