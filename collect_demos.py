#!/usr/bin/env python3
"""
Collect demonstration data from heuristic/expert policy.

Runs the TacticalDecisionMaker (heuristic policy) to generate
state-action pairs for behavior cloning pretraining.

Usage:
    python collect_demos.py 50000  # Collect 50k transitions
    python collect_demos.py 100000 --output demos_100k.pt
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch

import config
from models.ball import Ball, Vector2
from models.court import Court
from physics.physics_engine import PhysicsEngine
from rules.rule_engine import RuleEngine, TurnState
from ai.tactical_decision_maker import TacticalDecisionMaker, ShotType
from ai.neural.croquet_net import StateEncoder


@dataclass
class DemoTransition:
    """A single demonstration transition."""
    state: torch.Tensor
    valid_actions: List[int]
    expert_action: int
    shot_type: str  # For debugging/analysis


def collect_demos(
    num_transitions: int,
    verbose: bool = False
) -> List[DemoTransition]:
    """
    Collect demonstration transitions from heuristic policy.

    Args:
        num_transitions: Number of transitions to collect
        verbose: Print progress

    Returns:
        List of DemoTransition objects
    """
    court = Court()
    encoder = StateEncoder()
    demos = []

    episodes = 0
    start_time = time.time()

    while len(demos) < num_transitions:
        # Run one episode
        episode_demos = _run_demo_episode(court, encoder, verbose)
        demos.extend(episode_demos)
        episodes += 1

        if episodes % 100 == 0:
            elapsed = time.time() - start_time
            rate = len(demos) / elapsed
            remaining = (num_transitions - len(demos)) / rate if rate > 0 else 0
            print(f"  Episodes: {episodes} | Demos: {len(demos)}/{num_transitions} | "
                  f"Rate: {rate:.1f}/s | ETA: {remaining:.0f}s")

    # Trim to exact count
    demos = demos[:num_transitions]

    return demos


def _run_demo_episode(
    court: Court,
    encoder: StateEncoder,
    verbose: bool = False
) -> List[DemoTransition]:
    """Run one episode collecting demo transitions."""
    demos = []

    # Initialize game
    balls = _create_balls()
    physics = PhysicsEngine(court)
    rules = RuleEngine()
    dm = TacticalDecisionMaker()

    # Game state
    turn_count = 0
    max_turns = 200
    balls_in_play = {color: False for color in config.BALL_COLORS}
    last_ball_played = {"blue_black": None, "red_yellow": None}

    while turn_count < max_turns:
        # Check for game end
        bb_out = sum(1 for c in ["blue", "black"] if balls[c].has_pegged_out)
        ry_out = sum(1 for c in ["red", "yellow"] if balls[c].has_pegged_out)
        if bb_out == 2 or ry_out == 2:
            break

        # Determine current side and ball
        current_side = "blue_black" if turn_count % 2 == 0 else "red_yellow"
        side_colors = ["blue", "black"] if current_side == "blue_black" else ["red", "yellow"]

        # Alternate balls within side
        last_played = last_ball_played[current_side]
        if last_played is None:
            current_color = side_colors[0]
        else:
            current_color = side_colors[1] if last_played == side_colors[0] else side_colors[0]

        ball = balls[current_color]
        if ball.has_pegged_out:
            current_color = side_colors[0] if current_color == side_colors[1] else side_colors[1]
            ball = balls[current_color]
            if ball.has_pegged_out:
                turn_count += 1
                continue

        # Get active balls
        active_balls = {c: b for c, b in balls.items() if not b.has_pegged_out}

        # Turn loop
        rules.start_turn(current_color)

        while True:
            if rules.turn_info and rules.turn_info.strokes_remaining <= 0:
                break

            # Skip opening placements (first 4 balls entering)
            if not balls_in_play[current_color]:
                balls_in_play[current_color] = True
                # Simple placement near hoop 1 (at roughly (7, 7))
                import random
                ball.position = Vector2(
                    random.uniform(5, 9),  # Near hoop 1 x-position
                    random.uniform(2, 5)   # South of hoop 1
                )
                ball.velocity = Vector2(0, 0)
                break
            else:
                # Collect demo transition
                demo = _collect_transition(ball, active_balls, court, rules, dm, encoder)
                if demo:
                    demos.append(demo)

                # Execute the shot
                shot_option = dm._neural_action_to_shot(
                    demo.expert_action if demo else 5,  # Default to APPROACH
                    0.0, ball, active_balls, court, rules.deadness, {}
                )
                velocity, _ = _shot_option_to_velocity(ball, shot_option, court)
                ball.velocity = velocity

            # Simulate physics
            _simulate_physics(balls, physics)

            # Process rule events (collisions list is empty in this simplified simulation)
            turn_continues, events = rules.process_stroke_result(
                ball, active_balls, court, []
            )

            # Mark in balls
            physics.mark_in_all_balls(
                balls,
                striker_color=current_color,
                striker_has_strokes=turn_continues
            )

            if not turn_continues:
                break

        # Update tracking
        ball_side = config.BALL_TEAMS[current_color]
        last_ball_played[ball_side] = current_color
        turn_count += 1

    return demos


def _collect_transition(
    ball: Ball,
    balls: Dict[str, Ball],
    court: Court,
    rules: RuleEngine,
    dm: TacticalDecisionMaker,
    encoder: StateEncoder
) -> DemoTransition:
    """Collect a single demo transition."""
    deadness = rules.deadness
    strokes_remaining = rules.turn_info.strokes_remaining if rules.turn_info else 1
    is_continuation = rules.turn_info.state in [TurnState.CONTINUATION, TurnState.CROQUET_TAKEN] if rules.turn_info else False

    # Encode state
    state = encoder.encode(
        ball, balls, court, deadness, strokes_remaining, is_continuation
    )

    # Get valid actions
    valid_actions = dm._get_valid_neural_actions(ball, balls, court, deadness)

    if not valid_actions:
        return None

    # Get expert (heuristic) choice
    # Use the tactical decision maker's scoring to pick best action
    expert_action = _get_expert_action(ball, balls, court, deadness, dm, valid_actions)

    # Get shot type name for debugging
    shot_type_map = {
        0: "HOOP_RUN", 1: "ROQUET_NEAR", 2: "ROQUET_PARTNER",
        3: "ROQUET_OPP1", 4: "ROQUET_OPP2", 5: "APPROACH",
        6: "DEFENSIVE", 7: "PEG_OUT"
    }
    shot_type = shot_type_map.get(expert_action, "UNKNOWN")

    return DemoTransition(
        state=state,
        valid_actions=valid_actions,
        expert_action=expert_action,
        shot_type=shot_type
    )


def _get_expert_action(
    ball: Ball,
    balls: Dict[str, Ball],
    court: Court,
    deadness: Dict[str, set],
    dm: TacticalDecisionMaker,
    valid_actions: List[int]
) -> int:
    """
    Get the expert (heuristic) action choice.

    Uses TacticalDecisionMaker's select_best_shot to determine the expert choice,
    then maps it to the corresponding neural action index.
    """
    if not valid_actions:
        return 5  # Default to APPROACH

    # Get the best shot from the heuristic decision maker
    strokes_remaining = 1  # Conservative assumption
    best_shot = dm.select_best_shot(
        striker=ball,
        balls=balls,
        court=court,
        deadness=deadness,
        strokes_remaining=strokes_remaining,
        game_state={},
        use_expert_tactics=True
    )

    if best_shot is None:
        return valid_actions[0] if valid_actions else 5

    # Map shot type to neural action index
    shot_type_to_action = {
        ShotType.HOOP_RUN: 0,
        ShotType.ROQUET: 1,  # Will refine based on target
        ShotType.APPROACH: 5,
        ShotType.DEFENSIVE: 6,
        ShotType.PEG_OUT: 7,
    }

    # Start with base mapping
    action = shot_type_to_action.get(best_shot.shot_type, 5)

    # Refine ROQUET action based on target ball
    if best_shot.shot_type == ShotType.ROQUET and best_shot.target_ball:
        target_color = best_shot.target_ball.color if hasattr(best_shot.target_ball, 'color') else None
        if target_color:
            # Determine partner and opponents
            partner = "black" if ball.color == "blue" else "blue" if ball.color == "black" else \
                     "yellow" if ball.color == "red" else "red"
            opponents = [c for c in ["blue", "black", "red", "yellow"]
                        if c != ball.color and c != partner]

            if target_color == partner:
                action = 2  # ROQUET_PARTNER
            elif target_color in opponents:
                action = 3 if target_color == opponents[0] else 4  # ROQUET_OPP1 or ROQUET_OPP2
            else:
                action = 1  # ROQUET_NEAR (closest)
        else:
            action = 1  # ROQUET_NEAR

    # Ensure we return a valid action
    if action not in valid_actions:
        # Fall back to first valid action
        action = valid_actions[0]

    return action


def _create_balls() -> Dict[str, Ball]:
    """Create balls for a new game."""
    balls = {}
    for color in config.BALL_COLORS:
        balls[color] = Ball(color, (0, 0))  # Ball constructor expects tuple
        balls[color].hoops_run = 0
    return balls


def _shot_option_to_velocity(ball: Ball, option, court: Court) -> Tuple[Vector2, str]:
    """Convert shot option to velocity (simplified version)."""
    import math

    to_target = option.target - ball.position
    distance = to_target.magnitude()
    angle = math.atan2(to_target.y, to_target.x)

    # Physics-based power calculation
    friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY
    if distance > 0:
        base_power = math.sqrt(2 * friction_decel * distance)
    else:
        base_power = 0.0

    # Small multiplier for contact
    power = base_power * 1.01
    power = min(power, config.MAX_SHOT_POWER * 0.8)

    velocity = Vector2.from_angle(angle, power)
    return velocity, option.description


def _simulate_physics(balls: Dict[str, Ball], physics: PhysicsEngine, max_steps: int = 1000):
    """Simulate until all balls stop."""
    dt = 1.0 / 60.0  # 60 FPS physics timestep
    for _ in range(max_steps):
        any_moving = any(b.is_moving for b in balls.values())
        if not any_moving:
            break
        physics.update(balls, dt)


def save_demos(demos: List[DemoTransition], output_path: str):
    """Save demonstrations to file."""
    # Convert to tensors
    states = torch.stack([d.state for d in demos])

    # Create valid action masks
    num_actions = 8
    valid_masks = torch.zeros(len(demos), num_actions, dtype=torch.bool)
    for i, d in enumerate(demos):
        for a in d.valid_actions:
            valid_masks[i, a] = True

    expert_actions = torch.tensor([d.expert_action for d in demos], dtype=torch.long)

    # Analyze action distribution
    action_counts = {}
    for d in demos:
        action_counts[d.shot_type] = action_counts.get(d.shot_type, 0) + 1

    print("\nAction distribution in demos:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count} ({100*count/len(demos):.1f}%)")

    # Save
    torch.save({
        'states': states,
        'valid_masks': valid_masks,
        'expert_actions': expert_actions,
        'num_demos': len(demos),
        'action_distribution': action_counts,
    }, output_path)

    print(f"\nSaved {len(demos)} demos to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect demonstration data")
    parser.add_argument("num_transitions", type=int, help="Number of transitions to collect")
    parser.add_argument("--output", type=str, default="ai_data/neural/demos.pt",
                       help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.num_transitions} demonstration transitions...")
    print("Using TacticalDecisionMaker heuristic policy")
    print()

    start = time.time()
    demos = collect_demos(args.num_transitions, args.verbose)
    elapsed = time.time() - start

    print(f"\nCollection complete in {elapsed:.1f}s")
    print(f"Collected {len(demos)} transitions")

    save_demos(demos, args.output)


if __name__ == "__main__":
    main()
