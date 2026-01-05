#!/usr/bin/env python3
"""
Evaluate and compare different checkpoints to find peak performance.

Usage:
    python eval_checkpoints.py                    # Compare default checkpoints
    python eval_checkpoints.py ep5000 ep8000     # Compare specific checkpoints
"""
import sys
import argparse
from pathlib import Path

import torch

import config
from models.ball import Ball, Vector2
from models.court import Court
from physics.physics_engine import PhysicsEngine
from rules.rule_engine import RuleEngine, TurnState
from ai.tactical_decision_maker import TacticalDecisionMaker, ShotType

from ai.neural.croquet_net import CroquetNet, StateEncoder, get_device
from ai.neural.dqn_trainer import DQNTrainer, TrainingConfig


def load_checkpoint(checkpoint_path: str, trainer: DQNTrainer):
    """Load a checkpoint into the trainer."""
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    trainer.online_net.load_state_dict(checkpoint['online_net'])
    trainer.target_net.load_state_dict(checkpoint['target_net'])
    print(f"Loaded: {checkpoint_path}")


def run_greedy_game(trainer: DQNTrainer, court: Court, verbose: bool = False) -> dict:
    """
    Run a single game with greedy policy (no exploration).

    Returns:
        dict with hoops_run, turns, winner
    """
    import random

    # Create balls
    balls = {}
    if random.random() < 0.5:
        a_baulk = ["blue", "black"]
        b_baulk = ["red", "yellow"]
    else:
        a_baulk = ["red", "yellow"]
        b_baulk = ["blue", "black"]

    for color in ["blue", "black", "red", "yellow"]:
        if color in a_baulk:
            x = random.uniform(2, 12)
            y = 1
        else:
            x = random.uniform(16, 26)
            y = 34
        balls[color] = Ball(color, (x, y))

    physics = PhysicsEngine(court)
    rules = RuleEngine()

    balls_in_play = {c: False for c in ["blue", "black", "red", "yellow"]}
    turn_order = config.TURN_ORDER
    current_ball_index = 0
    current_side = "blue_black"
    opening_complete = False
    last_ball_played = {"blue_black": None, "red_yellow": None}

    hoops_run = 0
    turn_count = 0
    max_turns = 200

    while turn_count < max_turns:
        # Check game over
        bb_out = sum(1 for c in ["blue", "black"] if balls[c].has_pegged_out)
        ry_out = sum(1 for c in ["red", "yellow"] if balls[c].has_pegged_out)

        if bb_out == 2 or ry_out == 2:
            break

        # Determine current ball
        if not opening_complete:
            if current_ball_index < 4:
                current_color = turn_order[current_ball_index]
            else:
                opening_complete = True

        if opening_complete:
            if current_side == "blue_black":
                options = [c for c in ["blue", "black"] if not balls[c].has_pegged_out]
            else:
                options = [c for c in ["red", "yellow"] if not balls[c].has_pegged_out]

            if not options:
                current_side = "red_yellow" if current_side == "blue_black" else "blue_black"
                continue

            last = last_ball_played.get(current_side)
            if last in options and len(options) > 1:
                options = [c for c in options if c != last]
            current_color = options[0]

        ball = balls[current_color]
        rules.start_turn(current_color)

        # Process strokes in turn
        while rules.turn_info and rules.turn_info.strokes_remaining > 0:
            balls_in_play[current_color] = True
            active_balls = {c: b for c, b in balls.items() if balls_in_play[c]}

            if not opening_complete:
                # Opening placement
                ball.position = Vector2(random.uniform(5, 9), random.uniform(2, 5))
                ball.velocity = Vector2(0, 0)
            else:
                # Greedy neural decision
                deadness = rules.deadness
                strokes_remaining = rules.turn_info.strokes_remaining if rules.turn_info else 1
                is_continuation = rules.turn_info.state in [TurnState.CONTINUATION, TurnState.CROQUET_TAKEN] if rules.turn_info else False

                state = trainer.encoder.encode(
                    ball, active_balls, court, deadness, strokes_remaining, is_continuation
                )

                dm = TacticalDecisionMaker()
                valid_actions = dm._get_valid_neural_actions(ball, active_balls, court, deadness)

                # GREEDY selection (no exploration)
                action, q_value = trainer.select_action(state, valid_actions, training=False)

                shot_option = dm._neural_action_to_shot(
                    action, q_value, ball, active_balls, court, deadness, {}
                )

                # Execute shot
                to_target = shot_option.target - ball.position
                distance = to_target.magnitude()
                angle = to_target.normalize() if distance > 0.1 else Vector2(1, 0)

                if shot_option.shot_type == ShotType.HOOP_RUN:
                    power = min(distance * 1.5 + 2.0, 8.0)
                elif shot_option.shot_type == ShotType.ROQUET:
                    power = distance * 0.8 + 3.0
                else:
                    power = distance * 0.6 + 2.0

                velocity = angle * power
                physics.shoot_ball(ball, velocity)

                # Track hoops before
                hoops_before = ball.hoops_run

            # Simulate physics
            max_steps = 300
            dt = 1.0 / 60.0
            for _ in range(max_steps):
                physics.update(balls, dt)
                if physics.are_all_balls_stopped(balls):
                    break

            # Process result
            shot_collisions = []
            turn_continues, events = rules.process_stroke_result(
                ball, balls, court, shot_collisions
            )

            physics.mark_in_all_balls(
                balls,
                striker_color=current_color,
                striker_has_strokes=turn_continues
            )

            for event in events:
                if "runs hoop" in event.lower():
                    hoops_run += 1

            if not turn_continues:
                break

        # Update tracking
        ball_side = config.BALL_TEAMS[current_color]
        last_ball_played[ball_side] = current_color
        turn_count += 1

        if not opening_complete:
            current_ball_index += 1
            if current_ball_index >= 4:
                opening_complete = True
        else:
            current_side = "red_yellow" if current_side == "blue_black" else "blue_black"

    # Determine winner
    bb_score = balls["blue"].hoops_run + balls["black"].hoops_run
    ry_score = balls["red"].hoops_run + balls["yellow"].hoops_run

    if bb_score > ry_score:
        winner = "blue_black"
    elif ry_score > bb_score:
        winner = "red_yellow"
    else:
        winner = "draw"

    return {
        'hoops_run': hoops_run,
        'turns': turn_count,
        'winner': winner,
        'bb_score': bb_score,
        'ry_score': ry_score
    }


def evaluate_checkpoint(checkpoint_path: str, num_games: int = 20) -> dict:
    """Evaluate a checkpoint over multiple games."""
    # Load checkpoint to check if it used dueling architecture
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_config = checkpoint.get('config', {})
    use_dueling = saved_config.get('use_dueling', False)

    # If checkpoint has dueling-style keys, use dueling
    if 'online_net' in checkpoint:
        state_dict = checkpoint['online_net']
        if any('value_stream' in k or 'advantage_stream' in k for k in state_dict.keys()):
            use_dueling = True

    cfg = TrainingConfig(use_dueling=use_dueling)
    trainer = DQNTrainer(cfg)
    court = Court()

    load_checkpoint(checkpoint_path, trainer)

    total_hoops = 0
    total_turns = 0
    bb_wins = 0

    for i in range(num_games):
        result = run_greedy_game(trainer, court)
        total_hoops += result['hoops_run']
        total_turns += result['turns']
        if result['winner'] == 'blue_black':
            bb_wins += 1

        if (i + 1) % 5 == 0:
            print(f"  Game {i+1}/{num_games}: hoops={result['hoops_run']}, "
                  f"score={result['bb_score']}-{result['ry_score']}")

    return {
        'avg_hoops': total_hoops / num_games,
        'avg_turns': total_turns / num_games,
        'win_rate': bb_wins / num_games * 100,
        'checkpoint': checkpoint_path
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument(
        "checkpoints",
        nargs="*",
        default=["ep5000", "ep8000"],
        help="Checkpoint suffixes to compare (e.g., ep5000 ep8000)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of games per checkpoint (default: 20)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CHECKPOINT EVALUATION")
    print("=" * 60)
    print(f"Evaluating {len(args.checkpoints)} checkpoints with {args.games} games each")
    print()

    results = []
    base_path = Path("ai_data/neural")

    for suffix in args.checkpoints:
        checkpoint_path = base_path / f"checkpoint_{suffix}.pt"
        if not checkpoint_path.exists():
            print(f"Warning: {checkpoint_path} not found, skipping")
            continue

        print(f"\n--- Evaluating {suffix} ---")
        result = evaluate_checkpoint(str(checkpoint_path), args.games)
        results.append(result)

    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Checkpoint':<20} {'Avg Hoops':>12} {'Avg Turns':>12} {'Win Rate':>12}")
    print("-" * 60)

    best_hoops = max(r['avg_hoops'] for r in results)

    for r in results:
        suffix = Path(r['checkpoint']).stem.replace('checkpoint_', '')
        marker = " *BEST*" if r['avg_hoops'] == best_hoops else ""
        print(f"{suffix:<20} {r['avg_hoops']:>12.1f} {r['avg_turns']:>12.1f} "
              f"{r['win_rate']:>11.0f}%{marker}")

    print()
    print("Recommendation: Use the checkpoint with highest Avg Hoops for gameplay")


if __name__ == "__main__":
    main()
