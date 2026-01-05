#!/usr/bin/env python3
"""
Headless Training - Run multiple games without graphics for learning.

This script runs the croquet simulator in headless mode (no pygame display)
to quickly train the AI through many games and evaluate learning progress.

Usage:
    python train_headless.py [num_games]

Example:
    python train_headless.py 10    # Run 10 games
    python train_headless.py       # Default: 5 games
"""
import sys
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import config
from models.ball import Ball, Vector2
from models.court import Court
from physics.physics_engine import PhysicsEngine
from rules.rule_engine import RuleEngine, TurnState
from ai.ai_controller import AIController
from ai.learning_strategy import LearningStrategy
from ai.learning.learner import CroquetLearner
from ai.learning.position_evaluator import PositionEvaluator


@dataclass
class GameStats:
    """Statistics from a single game."""
    winner: str = ""
    blue_black_score: int = 0
    red_yellow_score: int = 0
    total_turns: int = 0
    total_shots: int = 0
    hoops_run: Dict[str, int] = field(default_factory=dict)
    longest_break: int = 0
    duration_seconds: float = 0.0


class HeadlessSimulator:
    """
    Runs croquet simulation without graphics for fast training.
    """

    def __init__(self, verbose: bool = False, quiet: bool = True, use_neural: bool = False):
        """
        Initialize headless simulator.

        Args:
            verbose: If True, print shot-by-shot details
            quiet: If True, suppress all print statements from AI modules
            use_neural: If True, use trained neural network for shot selection
        """
        self.verbose = verbose
        self.quiet = quiet
        self.use_neural = use_neural
        self.court = Court()
        self.learner = CroquetLearner()
        self.position_evaluator = PositionEvaluator()

    def run_game(self) -> GameStats:
        """
        Run a single game and return statistics.

        Returns:
            GameStats with game results
        """
        import random
        import io
        import sys

        start_time = time.time()

        # Suppress print statements if quiet mode
        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
        stats = GameStats()
        stats.hoops_run = {"blue": 0, "black": 0, "red": 0, "yellow": 0}

        # Initialize game components
        balls = self._create_balls()
        physics = PhysicsEngine(self.court)
        rules = RuleEngine()
        ai_controllers = self._create_ai_controllers(use_neural=self.use_neural)

        # Track game state
        balls_in_play = {c: False for c in ["blue", "black", "red", "yellow"]}
        turn_order = config.TURN_ORDER
        current_ball_index = 0
        current_side = "blue_black"
        opening_complete = False

        # Double-tap prevention
        last_ball_played = {"blue_black": None, "red_yellow": None}

        # Game loop
        max_turns = 500  # Safety limit
        current_break_length = 0

        while stats.total_turns < max_turns:
            # Check for game over
            bb_score = balls["blue"].hoops_run + balls["black"].hoops_run
            ry_score = balls["red"].hoops_run + balls["yellow"].hoops_run

            # Check for pegged out balls
            bb_out = sum(1 for c in ["blue", "black"] if balls[c].has_pegged_out)
            ry_out = sum(1 for c in ["red", "yellow"] if balls[c].has_pegged_out)

            if bb_out == 2:
                stats.winner = "blue_black"
                break
            if ry_out == 2:
                stats.winner = "red_yellow"
                break

            # Determine current ball
            if not opening_complete:
                if current_ball_index < 4:
                    current_color = turn_order[current_ball_index]
                else:
                    opening_complete = True

            if opening_complete:
                # Side chooses ball (simplified: just alternate, avoid double-tap)
                if current_side == "blue_black":
                    options = [c for c in ["blue", "black"] if not balls[c].has_pegged_out]
                else:
                    options = [c for c in ["red", "yellow"] if not balls[c].has_pegged_out]

                if not options:
                    # Side has no balls left
                    current_side = "red_yellow" if current_side == "blue_black" else "blue_black"
                    continue

                # Avoid double-tap
                last = last_ball_played.get(current_side)
                if last in options and len(options) > 1:
                    options = [c for c in options if c != last]

                current_color = options[0]  # Simple choice

            ball = balls[current_color]
            ai = ai_controllers[current_color]

            # Start turn
            rules.start_turn(current_color)
            self.learner.start_new_break()
            turn_hoops = 0

            if self.verbose:
                print(f"\n--- Turn {stats.total_turns + 1}: {current_color.capitalize()} ---")

            # Process strokes in this turn
            while rules.turn_info and rules.turn_info.strokes_remaining > 0:
                stats.total_shots += 1

                # Mark ball as in play
                if not balls_in_play[current_color]:
                    balls_in_play[current_color] = True

                # Get balls in play for AI
                active_balls = {c: b for c, b in balls.items() if balls_in_play[c]}

                # Check for croquet stroke
                is_croquet = (rules.turn_info.state == TurnState.CROQUET_REQUIRED)

                if is_croquet and rules.turn_info.just_roqueted:
                    roqueted_color = rules.turn_info.just_roqueted
                    roqueted_ball = balls[roqueted_color]

                    # Place striker
                    placement = ai.select_croquet_placement(ball, roqueted_ball, active_balls, self.court)
                    ball.position = placement

                    # Get croquet shot
                    striker_vel, croqueted_vel, desc, stroke_type = ai.select_croquet_shot(
                        ball, roqueted_ball, active_balls, self.court, rules.deadness
                    )

                    rules.turn_info.state = TurnState.CROQUET_TAKEN

                    # Execute croquet stroke
                    physics.execute_croquet_stroke(ball, roqueted_ball, striker_vel, croqueted_vel)

                    if self.verbose:
                        print(f"  Croquet: {desc}")
                else:
                    # Normal shot
                    is_continuation = rules.turn_info.state in [TurnState.CONTINUATION, TurnState.CROQUET_TAKEN]
                    strokes_left = rules.turn_info.strokes_remaining

                    velocity, desc = ai.select_shot(
                        ball, active_balls, self.court,
                        balls_in_play=balls_in_play,
                        deadness=rules.deadness,
                        strokes_remaining=strokes_left,
                        is_continuation=is_continuation
                    )

                    # Execute shot
                    physics.shoot_ball(ball, velocity)

                    if self.verbose:
                        print(f"  Shot: {desc}")

                # Simulate physics until all balls stop
                max_physics_steps = 500
                steps = 0
                dt = 1.0 / 60.0
                shot_collisions = []

                while steps < max_physics_steps:
                    events = physics.update(balls, dt)
                    # Collect collision events for roquet detection
                    for event in events:
                        if event.get('type') == 'ball_collision':
                            shot_collisions.append(event)
                    if physics.are_all_balls_stopped(balls):
                        break
                    steps += 1

                # Process shot outcome with collisions for roquet detection
                turn_continues, events = rules.process_stroke_result(
                    ball, balls, self.court, shot_collisions
                )

                # MARK IN: Place balls in yard line area onto the yard line
                # Exception: striker's ball if they still have strokes remaining
                physics.mark_in_all_balls(
                    balls,
                    striker_color=current_color,
                    striker_has_strokes=turn_continues
                )

                # Track hoops run
                for event in events:
                    if "runs hoop" in event.lower() or "peels" in event.lower():
                        turn_hoops += 1
                        stats.hoops_run[current_color] = stats.hoops_run.get(current_color, 0) + 1
                        if self.verbose:
                            print(f"    -> {event}")

                if not turn_continues:
                    break

            # End of turn - RECORD LEARNING DATA
            current_break_length = turn_hoops
            if current_break_length > stats.longest_break:
                stats.longest_break = current_break_length

            # Record break result for learning (end_break is called by start_new_break)
            # But we need to make sure experiences are tracked for the break
            # The break stats will be updated when the next turn calls start_new_break

            # Record leave quality (positions at end of turn)
            active_balls = {c: b for c, b in balls.items() if balls_in_play[c]}
            if active_balls:
                # Evaluate position quality
                position_value = self.position_evaluator.evaluate(
                    ball, active_balls, self.court, rules.deadness
                )
                self.learner.record_leave_quality(position_value, active_balls, self.court)

            # Record which ball played
            ball_side = config.BALL_TEAMS[current_color]
            last_ball_played[ball_side] = current_color

            # Next turn
            stats.total_turns += 1

            # Track if opponent scored to evaluate our previous leave
            prev_side = current_side  # Side that just played

            if not opening_complete:
                current_ball_index += 1
                if current_ball_index >= 4:
                    opening_complete = True
            else:
                current_side = "red_yellow" if current_side == "blue_black" else "blue_black"

            # Record opponent outcome for leave learning
            # If WE scored hoops this turn, tell the learner the opponent's leave didn't work
            if turn_hoops > 0 and opening_complete:
                self.learner.record_opponent_outcome(opponent_ran_hoop=True)

        # Calculate final scores (hoops run + 1 for each peg-out)
        # In AC, each ball can score 13 points max: 12 hoops + peg-out
        bb_hoops = balls["blue"].hoops_run + balls["black"].hoops_run
        bb_pegs = sum(1 for c in ["blue", "black"] if balls[c].has_pegged_out)
        stats.blue_black_score = bb_hoops + bb_pegs

        ry_hoops = balls["red"].hoops_run + balls["yellow"].hoops_run
        ry_pegs = sum(1 for c in ["red", "yellow"] if balls[c].has_pegged_out)
        stats.red_yellow_score = ry_hoops + ry_pegs

        if not stats.winner:
            # Determine winner by score if no peg-outs
            if stats.blue_black_score > stats.red_yellow_score:
                stats.winner = "blue_black"
            elif stats.red_yellow_score > stats.blue_black_score:
                stats.winner = "red_yellow"
            else:
                stats.winner = "draw"

        stats.duration_seconds = time.time() - start_time

        # Restore stdout if we suppressed it
        if self.quiet:
            sys.stdout = old_stdout

        return stats

    def _create_balls(self) -> Dict[str, Ball]:
        """Create balls with randomized starting positions."""
        import random
        balls = {}

        # Randomize baulk assignments
        if random.random() < 0.5:
            a_baulk_team = ["blue", "black"]
            b_baulk_team = ["red", "yellow"]
        else:
            a_baulk_team = ["red", "yellow"]
            b_baulk_team = ["blue", "black"]

        for color in ["blue", "black", "red", "yellow"]:
            if color in a_baulk_team:
                x = random.uniform(2, 12)
                y = 1
            else:
                x = random.uniform(16, 26)
                y = 34
            balls[color] = Ball(color, (x, y))

        return balls

    def _create_ai_controllers(self, use_neural: bool = False) -> Dict[str, AIController]:
        """Create AI controllers for each ball with shared learner."""
        import random
        controllers = {}

        for color in ["blue", "black", "red", "yellow"]:
            aggression = random.uniform(0.3, 0.8)
            strategy = LearningStrategy(skill_level=random.uniform(0.7, 0.9))
            # Pass learner to AI controller for tactical learning
            controllers[color] = AIController(
                strategy=strategy,
                aggression=aggression,
                learner=self.learner,
                use_neural=use_neural
            )

        return controllers

    def save_learning(self):
        """Save learning state."""
        self.learner._save_state()


def main():
    """Run headless training."""
    import argparse

    parser = argparse.ArgumentParser(description="Headless croquet training")
    parser.add_argument("num_games", type=int, nargs="?", default=5,
                        help="Number of games to run (default: 5)")
    parser.add_argument("--neural", action="store_true",
                        help="Use trained neural network for shot selection")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")

    args = parser.parse_args()
    num_games = args.num_games
    use_neural = args.neural

    print("=" * 60)
    print("HEADLESS CROQUET TRAINING")
    print("=" * 60)
    print(f"Running {num_games} games without graphics...")
    if use_neural:
        print("Using NEURAL NETWORK for shot selection")
    else:
        print("Using hand-crafted Q-values for shot selection")
    print()

    simulator = HeadlessSimulator(verbose=args.verbose, quiet=not args.verbose, use_neural=use_neural)

    # Track overall stats
    results = []
    total_start = time.time()

    for game_num in range(1, num_games + 1):
        print(f"Game {game_num}/{num_games}...", end=" ", flush=True)

        stats = simulator.run_game()
        results.append(stats)

        print(f"Winner: {stats.winner} | Score: {stats.blue_black_score}-{stats.red_yellow_score} | "
              f"Turns: {stats.total_turns} | Longest break: {stats.longest_break} | "
              f"Time: {stats.duration_seconds:.1f}s")

    # Summary
    total_time = time.time() - total_start

    print()
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    bb_wins = sum(1 for r in results if r.winner == "blue_black")
    ry_wins = sum(1 for r in results if r.winner == "red_yellow")
    draws = sum(1 for r in results if r.winner == "draw")

    print(f"Games played: {num_games}")
    print(f"Blue/Black wins: {bb_wins} ({100*bb_wins/num_games:.1f}%)")
    print(f"Red/Yellow wins: {ry_wins} ({100*ry_wins/num_games:.1f}%)")
    print(f"Draws: {draws}")
    print()

    avg_turns = sum(r.total_turns for r in results) / num_games
    avg_shots = sum(r.total_shots for r in results) / num_games
    avg_break = sum(r.longest_break for r in results) / num_games
    max_break = max(r.longest_break for r in results)

    print(f"Average turns per game: {avg_turns:.1f}")
    print(f"Average shots per game: {avg_shots:.1f}")
    print(f"Average longest break: {avg_break:.1f} hoops")
    print(f"Best break across all games: {max_break} hoops")
    print()
    print(f"Total training time: {total_time:.1f}s ({total_time/num_games:.1f}s per game)")

    # Save learning
    print()
    print("Saving learning data...")
    simulator.save_learning()

    # Show learner stats
    learner_stats = simulator.learner.get_stats()
    print(f"Total experiences recorded: {learner_stats['experiences_recorded']}")
    print(f"Total hoops run in learning: {learner_stats['total_hoops_run']}")
    print(f"Power adjustment learned: {learner_stats.get('power_adjustment', 1.0):.3f}")
    print()
    print("Training complete!")


if __name__ == "__main__":
    main()
