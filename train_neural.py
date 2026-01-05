#!/usr/bin/env python3
"""
Neural Network Training - Train DQN for croquet AI.

This script trains a Deep Q-Network to play croquet by:
1. Running simulated games
2. Collecting experience (state, action, reward, next_state)
3. Training the network using experience replay
4. Evaluating performance periodically

Requirements:
    pip install torch

Usage:
    python train_neural.py [num_episodes] [--eval-freq N] [--checkpoint PATH]

Example:
    python train_neural.py 1000              # Train for 1000 episodes
    python train_neural.py 500 --eval-freq 50  # Evaluate every 50 episodes
    python train_neural.py 100 --checkpoint ai_data/neural/checkpoint.pt
"""
import sys
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Check PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is required for neural network training.")
    print("Install with: pip install torch")
    sys.exit(1)

import config
from models.ball import Ball, Vector2
from models.court import Court
from physics.physics_engine import PhysicsEngine
from rules.rule_engine import RuleEngine, TurnState
from ai.ai_controller import AIController
from ai.learning_strategy import LearningStrategy
from ai.tactical_decision_maker import TacticalDecisionMaker, ShotType

from ai.neural.croquet_net import CroquetNet, StateEncoder, get_device
from ai.neural.replay_buffer import ReplayBuffer
from ai.neural.dqn_trainer import DQNTrainer, TrainingConfig, create_reward_function


@dataclass
class EpisodeResult:
    """Result from a training episode."""
    total_reward: float
    hoops_run: int  # Total hoops (both sides combined)
    bb_hoops: int   # Blue/Black side hoops only (max 24)
    turns: int
    winner: str
    duration: float


class NeuralTrainer:
    """
    Trains neural network through simulated croquet games.

    The training loop:
    1. Reset game state
    2. For each step:
       a. Encode state
       b. Select action (epsilon-greedy)
       c. Execute action in simulator
       d. Observe reward and next state
       e. Store transition in replay buffer
       f. Train network on mini-batch
    3. Track performance metrics
    """

    def __init__(
        self,
        config: TrainingConfig = None,
        verbose: bool = False,
        use_expert_shaping: bool = False,
        use_planner: bool = False
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            verbose: Print detailed output
            use_expert_shaping: Use comprehensive expert tactical reward shaping
                               (includes Aiton, Wylie, Oxford Croquet Rules of Thumb)
            use_planner: Use LLM tactical planner as advisor during evaluation
        """
        self.config = config or TrainingConfig()
        self.verbose = verbose
        self.use_expert_shaping = use_expert_shaping
        self.use_planner = use_planner

        # Initialize DQN trainer
        self.dqn = DQNTrainer(self.config)

        # Initialize tactical planner if enabled
        self.planner = None
        if use_planner:
            try:
                from ai.tactical_planner import TacticalPlanner, PlannerConfig
                self.planner = TacticalPlanner(PlannerConfig())
                if self.planner.is_available():
                    print("LLM Tactical Planner enabled (advisor mode)")
                else:
                    print("LLM Tactical Planner: API key not found, running without")
                    self.planner = None
            except ImportError as e:
                print(f"LLM Tactical Planner unavailable: {e}")
                self.planner = None

        # Planner logging (tracks NN vs LLM agreement)
        self.planner_log = []

        # Game components (reset each episode)
        self.court = Court()
        self.reward_fn = create_reward_function(use_expert_shaping=use_expert_shaping)

        # Track hoops run this turn (for break building rewards)
        self.hoops_this_turn = 0  # Reset at start of each turn
        self.consecutive_hoops = 0  # Legacy: consecutive shots with hoops

        # Track steps since last hoop for time penalty
        self.steps_since_last_hoop = 0

        # Track rover state for peg out incentive
        self.steps_as_rover = 0

        # DELTA-BASED REWARD TRACKING (per Peter's feedback)
        # Track previous tactical context to compute deltas
        # Only reward CREATING a position, not maintaining it
        self.prev_tactical_context = {
            'break_balls': 0,
            'has_pioneer_at_next': False,
            'has_pioneer_at_next_but_one': False,
            'has_pilot': False,
            'has_rush_to_hoop': False,
            'cluster_quality': 0.0,
            'opponent_separation': 0.0
        }

        # ONCE-PER-TURN CAPS (per Peter's calibration feedback)
        # Prevent farming by only awarding each binary feature creation once per turn
        self.turn_tactical_awarded = {
            'pilot_created': False,
            'rush_created': False,
            'pioneer_next_created': False,
            'pioneer_next_but_one_created': False
        }
        # LOSS PENALTY CAPS (per Peter's feedback)
        # Cap loss penalties to at most 2 per turn per feature to avoid punishing necessary repositioning
        self.turn_loss_counts = {
            'pilot_lost': 0,
            'rush_lost': 0,
            'pioneer_next_lost': 0,
            'pioneer_next_but_one_lost': 0
        }

        # PER-TURN SHAPING BUDGET (per Peter's feedback)
        # Cap total tactical shaping rewards/penalties per turn to ensure hoop rewards remain dominant
        self.turn_tactical_reward_total = 0.0
        self.TACTICAL_REWARD_CAP = 3.0   # Max tactical bonus per turn
        self.TACTICAL_PENALTY_CAP = -2.0  # Max tactical penalty per turn

        # Training tracking
        self.episode_results: List[EpisodeResult] = []

        # Greedy eval metrics for plateau detection (updated during _evaluate)
        self.last_eval_metrics = {
            'greedy_avg_hoops': 0.0,
            'greedy_avg_turns': 0.0,
            'greedy_win_rate': 0.0
        }

    def train(
        self,
        num_episodes: int,
        eval_freq: int = 100,
        save_freq: int = 100
    ) -> Dict:
        """
        Train for specified number of episodes.

        Args:
            num_episodes: Number of episodes to train
            eval_freq: Episodes between evaluations
            save_freq: Episodes between checkpoints

        Returns:
            Training statistics
        """
        print(f"Training for {num_episodes} episodes on {self.dqn.device}")
        print(f"Config: batch_size={self.config.batch_size}, "
              f"lr={self.config.learning_rate}, gamma={self.config.gamma}")
        print()

        start_time = time.time()

        for episode in range(1, num_episodes + 1):
            result = self._run_episode()
            self.episode_results.append(result)

            # Print progress
            if episode % 10 == 0 or episode == 1:
                epsilon = self.dqn.get_epsilon()
                stats = self.dqn.stats.to_dict()
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {result.total_reward:.1f} | "
                      f"Hoops: {result.hoops_run} | "
                      f"Epsilon: {epsilon:.3f} | "
                      f"Loss: {stats['avg_loss']:.4f}")

            # Log intent distribution and masking stats every 100 episodes (per Peter's recommendation)
            if episode % 100 == 0:
                intent_dist = self.dqn.stats.get_intent_distribution()
                masking = self.dqn.stats.get_masking_stats()
                if intent_dist:
                    print(f"  Intent distribution: " +
                          " | ".join(f"{k}:{v:.1f}%" for k, v in sorted(intent_dist.items())))
                if masking['avg_valid_actions'] > 0:
                    print(f"  Avg valid actions: {masking['avg_valid_actions']:.1f}")

                # Log target mask effectiveness (should decrease as network learns legality)
                mask_stats = self.dqn.get_mask_stats()
                if mask_stats['mask_total'] > 0:
                    print(f"  Target mask: {mask_stats['mask_change_pct']:.1f}% actions changed by mask"
                          f" (empty fallback: {mask_stats['empty_fallback_pct']:.2f}%)")

            # Evaluation
            if eval_freq > 0 and episode % eval_freq == 0:
                self._evaluate(10)

            # Save checkpoint (include eval metrics for plateau detection)
            if save_freq > 0 and episode % save_freq == 0:
                self.dqn.save_checkpoint(f"_ep{episode}", self.last_eval_metrics)

        # Final save
        self.dqn.save_checkpoint("_final", self.last_eval_metrics)

        total_time = time.time() - start_time
        print()
        print(f"Training complete in {total_time:.1f}s")
        print(self.dqn.get_stats_summary())

        return self.dqn.stats.to_dict()

    def _run_episode(self, training: bool = True, tactical_kpis: dict = None) -> EpisodeResult:
        """Run one episode (training or evaluation).

        Args:
            training: If True, use epsilon-greedy and update network.
                     If False, use pure greedy (epsilon=0) and don't train.
            tactical_kpis: If provided (during eval), accumulate tactical KPI stats into this dict.
        """
        start_time = time.time()

        # Initialize game
        balls = self._create_balls()
        physics = PhysicsEngine(self.court)
        rules = RuleEngine()

        # Game state
        balls_in_play = {c: False for c in ["blue", "black", "red", "yellow"]}
        turn_order = config.TURN_ORDER
        current_ball_index = 0
        current_side = "blue_black"
        opening_complete = False
        last_ball_played = {"blue_black": None, "red_yellow": None}

        # Episode tracking
        total_reward = 0.0
        hoops_run = 0
        turn_count = 0
        max_turns = 200  # Shorter for faster training

        # Reset per-episode counters
        self.hoops_this_turn = 0
        self.consecutive_hoops = 0
        self.steps_since_last_hoop = 0
        self.steps_as_rover = 0
        # Reset delta-tracking for new episode
        self.prev_tactical_context = {
            'break_balls': 0,
            'has_pioneer_at_next': False,
            'has_pioneer_at_next_but_one': False,
            'has_pilot': False,
            'has_rush_to_hoop': False,
            'cluster_quality': 0.0,
            'opponent_separation': 0.0
        }

        while turn_count < max_turns:
            # Check game over
            bb_out = sum(1 for c in ["blue", "black"] if balls[c].has_pegged_out)
            ry_out = sum(1 for c in ["red", "yellow"] if balls[c].has_pegged_out)

            if bb_out == 2:
                break
            if ry_out == 2:
                break

            # Determine current ball
            if not opening_complete:
                if current_ball_index < 4:
                    current_color = turn_order[current_ball_index]
                else:
                    opening_complete = True

            if opening_complete:
                if current_side == "blue_black":
                    options = [c for c in ["blue", "black"]
                              if not balls[c].has_pegged_out]
                else:
                    options = [c for c in ["red", "yellow"]
                              if not balls[c].has_pegged_out]

                if not options:
                    current_side = "red_yellow" if current_side == "blue_black" else "blue_black"
                    continue

                last = last_ball_played.get(current_side)
                if last in options and len(options) > 1:
                    options = [c for c in options if c != last]

                current_color = options[0]

            ball = balls[current_color]
            rules.start_turn(current_color)

            # Reset hoops-this-turn counter at start of each turn
            self.hoops_this_turn = 0
            self.consecutive_hoops = 0

            # Reset delta-tracking at turn start (per Peter's feedback)
            # Tactical deltas must be computed relative to the SAME player's perspective
            # Carrying over prev_tactical_context across turn boundaries would compare
            # apples to oranges (your pioneers vs opponent's pioneers)
            self.prev_tactical_context = {
                'break_balls': 0,
                'has_pioneer_at_next': False,
                'has_pioneer_at_next_but_one': False,
                'has_pilot': False,
                'has_rush_to_hoop': False,
                'cluster_quality': 0.0,
                'opponent_separation': 0.0
            }

            # Reset once-per-turn caps at turn start
            self.turn_tactical_awarded = {
                'pilot_created': False,
                'rush_created': False,
                'pioneer_next_created': False,
                'pioneer_next_but_one_created': False
            }
            # Reset loss penalty counts at turn start
            self.turn_loss_counts = {
                'pilot_lost': 0,
                'rush_lost': 0,
                'pioneer_next_lost': 0,
                'pioneer_next_but_one_lost': 0
            }
            # Reset per-turn shaping budget
            self.turn_tactical_reward_total = 0.0

            # Process strokes in turn
            while rules.turn_info and rules.turn_info.strokes_remaining > 0:
                balls_in_play[current_color] = True
                active_balls = {c: b for c, b in balls.items() if balls_in_play[c]}

                # Skip opening phase for neural training
                if not opening_complete:
                    # Simple opening shot - position ball south of hoop 1
                    # Hoop 1 is at (7, 7), approach from south (y < 7)
                    # Just teleport the ball to a good starting position near hoop 1
                    ball.position = Vector2(
                        random.uniform(5, 9),  # Near hoop 1 x-position
                        random.uniform(2, 5)   # South of hoop 1 (approach side)
                    )
                    ball.velocity = Vector2(0, 0)  # Ball at rest
                    ball.shot_start_position = ball.position.copy()
                    if self.verbose:
                        print(f"  OPENING: {ball.color} placed at {ball.position}")
                else:
                    # Neural network decision
                    episode_reward, step_hoop_run = self._neural_step(
                        ball, active_balls, rules, physics, training=training,
                        tactical_kpis=tactical_kpis
                    )
                    total_reward += episode_reward
                    if step_hoop_run:
                        hoops_run += 1

                # Simulate physics
                self._simulate_physics(balls, physics)

                # Debug: show where ball ended up after opening
                if self.verbose and not opening_complete:
                    print(f"  AFTER OPENING: {ball.color} ended at {ball.position}")

                # Process result
                shot_collisions = []  # Would need to track from physics
                turn_continues, events = rules.process_stroke_result(
                    ball, balls, self.court, shot_collisions
                )

                # Mark in balls
                physics.mark_in_all_balls(
                    balls,
                    striker_color=current_color,
                    striker_has_strokes=turn_continues
                )

                # Note: hoops are already counted via step_hoop_run from _neural_step
                # Don't double-count from rule engine events
                # (The reward function in _neural_step already handles hoop rewards)

                if not turn_continues:
                    break

            # BUDGET TRACKING at turn end (per Peter's feedback)
            # Record how much of the tactical budget was used this turn
            if tactical_kpis is not None:
                tactical_kpis['turn_count'] += 1
                tactical_kpis['total_budget_used'] += abs(self.turn_tactical_reward_total)
                # Check if caps were hit
                if self.turn_tactical_reward_total >= self.TACTICAL_REWARD_CAP * 0.95:
                    tactical_kpis['turns_hit_bonus_cap'] += 1
                if self.turn_tactical_reward_total <= self.TACTICAL_PENALTY_CAP * 0.95:
                    tactical_kpis['turns_hit_penalty_cap'] += 1

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

        return EpisodeResult(
            total_reward=total_reward,
            hoops_run=hoops_run,  # Total (both sides)
            bb_hoops=bb_score,     # Blue/Black only (max 24)
            turns=turn_count,
            winner=winner,
            duration=time.time() - start_time
        )

    def _neural_step(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        rules: RuleEngine,
        physics: PhysicsEngine,
        training: bool = True,
        tactical_kpis: dict = None
    ) -> float:
        """
        Execute one neural network step.

        Args:
            training: If True, use epsilon-greedy and update network.
                     If False, use pure greedy and skip learning.
            tactical_kpis: If provided (during eval), accumulate tactical KPI stats.

        Returns reward for this step.
        """
        deadness = rules.deadness
        strokes_remaining = rules.turn_info.strokes_remaining if rules.turn_info else 1
        is_continuation = rules.turn_info.state in [TurnState.CONTINUATION, TurnState.CROQUET_TAKEN] if rules.turn_info else False

        # Encode current state
        state = self.dqn.encoder.encode(
            ball, balls, self.court, deadness, strokes_remaining, is_continuation
        )

        # Get valid actions
        dm = TacticalDecisionMaker()
        valid_actions = dm._get_valid_neural_actions(ball, balls, self.court, deadness)

        # Select action (epsilon-greedy if training, pure greedy if evaluating)
        action, q_value = self.dqn.select_action(state, valid_actions, training=training)

        # LLM Planner advisor (only during evaluation, not training)
        planner_output = None
        if self.planner and not training:
            planner_output = self.planner.plan(
                striker=ball,
                balls=balls,
                court=self.court,
                deadness=deadness,
                strokes_remaining=strokes_remaining,
                valid_intents=valid_actions
            )

            if planner_output and planner_output.ranked_intents:
                llm_top = planner_output.ranked_intents[0].intent
                llm_top_name = planner_output.ranked_intents[0].intent_name

                # Log NN vs LLM comparison
                action_names = ['HOOP_RUN', 'ROQUET_NEAR', 'ROQUET_PARTNER',
                               'ROQUET_OPP1', 'ROQUET_OPP2', 'APPROACH', 'DEFENSIVE', 'PEG_OUT']
                nn_action_name = action_names[action] if action < len(action_names) else 'UNKNOWN'

                agreement = (action == llm_top)
                self.planner_log.append({
                    'nn_action': action,
                    'nn_action_name': nn_action_name,
                    'llm_top': llm_top,
                    'llm_top_name': llm_top_name,
                    'agreement': agreement,
                    'llm_reason': planner_output.ranked_intents[0].reason,
                    'cached': planner_output.cached,
                })

                if self.verbose:
                    agree_str = "AGREE" if agreement else "DISAGREE"
                    print(f"  [PLANNER] NN={nn_action_name} vs LLM={llm_top_name} ({agree_str})")
                    if not agreement:
                        print(f"            LLM reason: {planner_output.ranked_intents[0].reason}")

        # Convert action to shot
        shot_option = dm._neural_action_to_shot(
            action, q_value, ball, balls, self.court, deadness, {}
        )

        if self.verbose:
            print(f"  NEURAL: {ball.color} at {ball.position} -> {shot_option.shot_type.name} to {shot_option.target}")

        # Execute shot
        to_target = shot_option.target - ball.position
        distance = to_target.magnitude()
        angle = to_target.normalize() if distance > 0.1 else Vector2(1, 0)

        # Calculate power based on shot type
        if shot_option.shot_type == ShotType.HOOP_RUN:
            power = min(distance * 1.5 + 2.0, 8.0)
        elif shot_option.shot_type == ShotType.ROQUET:
            power = distance * 0.8 + 3.0
        else:
            power = distance * 0.6 + 2.0

        # Add executor noise during training (per Peter's recommendation)
        # This forces the network to learn robust preferences, not brittle ones
        import random
        # Aim noise: proportional to distance (harder to aim accurately at long range)
        aim_noise_scale = 0.03 * distance  # ~1.7 degrees per yard at max
        aim_noise = random.gauss(0, aim_noise_scale)
        # Rotate angle by small random amount
        import math
        cos_n, sin_n = math.cos(aim_noise), math.sin(aim_noise)
        noisy_angle = Vector2(
            angle.x * cos_n - angle.y * sin_n,
            angle.x * sin_n + angle.y * cos_n
        )
        # Power noise: small percentage of power
        power_noise = random.gauss(0, power * 0.05)  # 5% std dev
        noisy_power = max(1.0, power + power_noise)

        velocity = noisy_angle * noisy_power
        physics.shoot_ball(ball, velocity)

        # Store positions before simulation to detect collisions, hoop runs, and peg outs
        ball_start_pos = ball.position.copy() if hasattr(ball.position, 'copy') else Vector2(ball.position.x, ball.position.y)
        hoops_before = ball.hoops_run
        was_pegged_out_before = ball.has_pegged_out
        target_ball_start = None
        if shot_option.target_ball:
            target_ball_start = shot_option.target_ball.position.copy() if hasattr(shot_option.target_ball.position, 'copy') else Vector2(shot_option.target_ball.position.x, shot_option.target_ball.position.y)

        # Simulate ALL balls to detect collisions properly
        collisions = self._simulate_physics_with_detection(balls, physics, ball.color)

        # Determine reward with full tactical context
        hoop_run = ball.hoops_run > hoops_before  # Actually check if hoop was scored
        if hoop_run and self.verbose:
            print(f"  [NEURAL HOOP] {ball.color} hoop detected: {hoops_before} -> {ball.hoops_run}")

        # Check for peg out - ball must be rover (12 hoops) and hit the peg
        pegged_out = ball.has_pegged_out and not was_pegged_out_before
        if not pegged_out and ball.hoops_run >= 12:
            # Also check if ball hit the peg during this shot
            pegged_out = self._check_peg_out(ball)
        if pegged_out and self.verbose:
            print(f"  [NEURAL PEG OUT] {ball.color} pegged out!")

        roqueted = len(collisions) > 0  # Check actual collisions

        # Also check by distance if we got close to target ball (physics might have missed collision)
        if not roqueted and shot_option.target_ball and target_ball_start:
            # If target ball moved, we probably hit it
            target_moved = (shot_option.target_ball.position - target_ball_start).magnitude() > 0.5
            # Or if we're very close to where it was
            dist_to_target = (ball.position - shot_option.target_ball.position).magnitude()
            if target_moved or dist_to_target < 1.5:
                roqueted = True

        target_hoop = self.court.get_hoop_for_ball(ball.hoops_run)

        # Calculate distance to hoop for approach quality
        dist_to_hoop = None
        if target_hoop:
            dist_to_hoop = (target_hoop.position - ball.position).magnitude()

        # Track hoops this turn and steps since last hoop
        if hoop_run:
            self.hoops_this_turn += 1  # Track hoops in current turn/break
            self.consecutive_hoops = self.hoops_this_turn  # Update for reward function
            self.steps_since_last_hoop = 0  # Reset on hoop
        else:
            # Don't reset hoops_this_turn on non-hoop shots - that's for turn end
            self.steps_since_last_hoop += 1  # Increment when no hoop

        # Track rover status (ball has run all 12 hoops)
        is_rover = ball.hoops_run >= 12
        if is_rover:
            self.steps_as_rover += 1
        else:
            self.steps_as_rover = 0

        # Check if defensive action was chosen (action index 6)
        chose_defensive = (action == 6)

        # ===========================================
        # COMPUTE TACTICAL CONTEXT FOR EXPERT SHAPING
        # ===========================================
        tactical_context = self._compute_tactical_context(
            ball, balls, target_hoop, dist_to_hoop, shot_option, distance
        )

        # Compute reward with once-per-turn caps for both creation and loss penalties
        # Returns (base_reward, tactical_reward, tactical_awards, loss_increments)
        base_reward, tactical_reward, tactical_awards, loss_increments = self.reward_fn(
            hoop_run=hoop_run,
            roqueted=roqueted,
            good_approach=dist_to_hoop < 3 if dist_to_hoop else False,
            pegged_out=pegged_out,  # Now using actual peg out detection!
            turn_ended=strokes_remaining <= 1 and not roqueted,
            shot_hit=roqueted,
            is_rover=is_rover,
            steps_as_rover=self.steps_as_rover,
            chose_defensive=chose_defensive,
            # DELTA-BASED: Pass previous context for computing deltas
            prev_break_balls=self.prev_tactical_context['break_balls'],
            prev_has_pioneer_at_next=self.prev_tactical_context['has_pioneer_at_next'],
            prev_has_pioneer_at_next_but_one=self.prev_tactical_context['has_pioneer_at_next_but_one'],
            prev_has_pilot=self.prev_tactical_context['has_pilot'],
            prev_has_rush_to_hoop=self.prev_tactical_context['has_rush_to_hoop'],
            prev_cluster_quality=self.prev_tactical_context['cluster_quality'],
            prev_opponent_separation=self.prev_tactical_context['opponent_separation'],
            # ONCE-PER-TURN CAPS: Pass current awarded/loss state
            turn_tactical_awarded=self.turn_tactical_awarded,
            turn_loss_counts=self.turn_loss_counts,
            # Pass full tactical context
            **tactical_context
        )

        # ===========================================
        # PER-TURN SHAPING BUDGET (per Peter's feedback)
        # Cap total tactical shaping per turn so hoop rewards dominate
        # Bonuses capped at +3.0, penalties at -2.0 per turn
        # ===========================================

        # Check budget headroom BEFORE applying this step's tactical reward
        if tactical_reward > 0:
            # Positive tactical reward - check bonus budget
            budget_remaining = self.TACTICAL_REWARD_CAP - self.turn_tactical_reward_total
            if budget_remaining <= 0:
                # Already at cap, no more bonus this turn
                tactical_reward = 0.0
            elif tactical_reward > budget_remaining:
                # Partial - only give what's left in budget
                tactical_reward = budget_remaining
        elif tactical_reward < 0:
            # Negative tactical reward - check penalty budget
            # turn_tactical_reward_total goes negative for penalties
            budget_remaining = self.TACTICAL_PENALTY_CAP - self.turn_tactical_reward_total
            if budget_remaining >= 0:
                # Already at penalty cap, no more penalty this turn
                tactical_reward = 0.0
            elif tactical_reward < budget_remaining:
                # Partial - only apply what's left in budget
                tactical_reward = budget_remaining

        # Update running total for this turn
        self.turn_tactical_reward_total += tactical_reward

        # Final reward = base (never capped) + tactical (capped by budget)
        reward = base_reward + tactical_reward

        # Update once-per-turn caps based on what was awarded this step
        for key, was_awarded in tactical_awards.items():
            if was_awarded:
                self.turn_tactical_awarded[key] = True

        # Update loss penalty counts for capping
        for key, increment in loss_increments.items():
            if increment > 0:
                self.turn_loss_counts[key] = self.turn_loss_counts.get(key, 0) + increment

        # Update previous context for next step's delta calculation
        self.prev_tactical_context = {
            'break_balls': tactical_context.get('break_balls', 0),
            'has_pioneer_at_next': tactical_context.get('has_pioneer_at_next', False),
            'has_pioneer_at_next_but_one': tactical_context.get('has_pioneer_at_next_but_one', False),
            'has_pilot': tactical_context.get('has_pilot', False),
            'has_rush_to_hoop': tactical_context.get('has_rush_to_hoop', False),
            'cluster_quality': tactical_context.get('cluster_quality', 0.0),
            'opponent_separation': tactical_context.get('opponent_separation', 0.0)
        }

        # TACTICAL KPI TRACKING (only during eval when tactical_kpis is provided)
        if tactical_kpis is not None:
            tactical_kpis['sample_count'] += 1
            if tactical_context.get('has_pilot', False):
                tactical_kpis['pilot_rate'] += 1
            if tactical_context.get('has_pioneer_at_next', False):
                tactical_kpis['pioneer_next_rate'] += 1
            if tactical_context.get('has_pioneer_at_next_but_one', False):
                tactical_kpis['pioneer_next2_rate'] += 1
            if tactical_context.get('has_rush_to_hoop', False):
                tactical_kpis['rush_ready_rate'] += 1
            tactical_kpis['avg_cluster_quality'] += tactical_context.get('cluster_quality', 0.0)
            tactical_kpis['avg_break_balls'] += tactical_context.get('break_balls', 0)
            tactical_kpis['avg_opp_separation'] += tactical_context.get('opponent_separation', 0.0)

        # Encode next state
        next_state = self.dqn.encoder.encode(
            ball, balls, self.court, deadness, strokes_remaining - 1, True
        )

        # Only store transitions and train during training mode
        # (Skip during greedy evaluation to avoid polluting replay buffer
        # and triggering LR decay at wrong epsilon values)
        if training:
            # Get valid actions for next state (for proper target masking)
            # This prevents bootstrapping from invalid actions in Double DQN
            next_valid_actions = dm._get_valid_neural_actions(ball, balls, self.court, deadness)

            # Store transition
            # TERMINAL HANDLING (per Peter's feedback):
            # done=True ONLY when game actually ends, not just turn end.
            # - Turn ending is NOT terminal (opponent plays, then you play again)
            # - Single ball peg-out is NOT terminal (partner ball continues)
            # - Game ends when BOTH balls of a side peg out
            #
            # Check if this peg-out ends the game
            bb_out = sum(1 for c in ["blue", "black"] if balls[c].has_pegged_out)
            ry_out = sum(1 for c in ["red", "yellow"] if balls[c].has_pegged_out)
            game_over = (bb_out == 2) or (ry_out == 2)

            # Only mark done=True at actual game end
            done = game_over
            # Store next_valid_actions in info dict for target masking
            self.dqn.store_transition(
                state, action, reward, next_state, done,
                info={'next_valid_actions': next_valid_actions}
            )

            # Train
            self.dqn.train_step()

        return reward, hoop_run

    def _compute_tactical_context(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        target_hoop,
        dist_to_hoop: float,
        shot_option,
        shot_distance: float
    ) -> Dict:
        """
        Compute tactical context for reward shaping.

        Break-building context is ALWAYS computed (needed for base rewards).
        Expert tactical context (approach quality, rush quality, etc.) is
        only computed when use_expert_shaping=True.
        """
        context = {}

        # ===========================================
        # BREAK BUILDING CONTEXT (always computed)
        # These are needed for base break-building rewards
        # ===========================================

        # Count balls in useful break positions
        break_balls = self._count_break_balls(ball, balls, target_hoop)
        context['break_balls'] = break_balls

        # Check for pioneers at upcoming hoops
        # Use hysteresis from prev_tactical_context to prevent flip-flopping
        if target_hoop:
            # Pilot: ball near CURRENT hoop (for approach/croquet)
            # Hysteresis: use previous pilot state
            was_pilot = self.prev_tactical_context.get('has_pilot', False)
            context['has_pilot'] = self._has_pilot(
                balls, target_hoop.position, ball.color, radius=3.0,
                was_pilot=was_pilot
            )

            # Pioneer at next hoop (with hysteresis)
            next_hoop = self.court.get_hoop_for_ball(ball.hoops_run + 1)
            if next_hoop:
                was_pioneer_next = self.prev_tactical_context.get('has_pioneer_at_next', False)
                context['has_pioneer_at_next'] = self._has_pioneer_near(
                    balls, next_hoop.position, ball.color, radius=5.0,
                    was_pioneer=was_pioneer_next
                )
                # Rush availability: can we rush a ball to the next hoop?
                context['has_rush_to_hoop'] = self._has_rush_to_position(
                    ball, balls, next_hoop.position, max_cut_angle=30.0
                )
            else:
                context['has_pioneer_at_next'] = False
                context['has_rush_to_hoop'] = False

            # Pioneer at next-but-one (with hysteresis)
            next_but_one = self.court.get_hoop_for_ball(ball.hoops_run + 2)
            if next_but_one:
                was_pioneer_next2 = self.prev_tactical_context.get('has_pioneer_at_next_but_one', False)
                context['has_pioneer_at_next_but_one'] = self._has_pioneer_near(
                    balls, next_but_one.position, ball.color, radius=7.0,
                    was_pioneer=was_pioneer_next2
                )
            else:
                context['has_pioneer_at_next_but_one'] = False
        else:
            context['has_pilot'] = False
            context['has_pioneer_at_next'] = False
            context['has_pioneer_at_next_but_one'] = False
            context['has_rush_to_hoop'] = False

        # Cluster quality: how tightly grouped are all balls?
        context['cluster_quality'] = self._get_cluster_quality(ball, balls)

        # Opponent separation: how spread out are opponent balls?
        context['opponent_separation'] = self._get_opponent_separation(ball, balls)

        # Track consecutive hoops for break building reward
        context['consecutive_hoops'] = self.consecutive_hoops
        context['turn_continues'] = True  # We're mid-turn

        # Track steps since last hoop for time penalty
        context['steps_since_last_hoop'] = self.steps_since_last_hoop

        # ===========================================
        # EXPERT TACTICAL CONTEXT (only when enabled)
        # ===========================================
        if not self.use_expert_shaping:
            return context

        # 1. APPROACH QUALITY (Aiton: 1 yard = ~0.9m is ideal)
        # CRITICAL: Also check approach angle - being close but on wrong side is BAD
        if dist_to_hoop is not None and target_hoop:
            # Check if ball is on approach side (can run through in correct direction)
            ball_to_hoop = target_hoop.position - ball.position
            required_dir = target_hoop.direction  # Direction ball must travel through hoop

            # Dot product tells us if we're on the approach side (positive = good)
            # Ball should be "behind" the hoop relative to required direction
            approach_dot = ball_to_hoop.normalize().dot(required_dir) if ball_to_hoop.magnitude() > 0.1 else 0

            if approach_dot > 0.3:  # Ball is on correct side to approach
                if dist_to_hoop <= 1.2:
                    context['approach_quality'] = 'excellent'
                elif dist_to_hoop <= 2.0:
                    context['approach_quality'] = 'good'
                elif dist_to_hoop <= 3.5:
                    context['approach_quality'] = 'fair'
                else:
                    context['approach_quality'] = 'poor'
            else:
                # Ball is on wrong side of hoop - this is a bad position!
                context['approach_quality'] = 'poor'
                # Don't give distance bonus for wrong-side approaches

            context['approach_distance'] = dist_to_hoop if approach_dot > 0.3 else None
            context['approach_angle'] = max(0, approach_dot)  # 0-1 quality score

        # 2. RUSH QUALITY (Wylie: straight > slight cut >> fine cut)
        if shot_option.target_ball:
            target_pos = shot_option.target_ball.position
            # Calculate cut angle
            if target_hoop:
                to_hoop = target_hoop.position - ball.position
                to_target = target_pos - ball.position
                if to_hoop.magnitude() > 0.1 and to_target.magnitude() > 0.1:
                    # Dot product for angle between directions
                    dot = (to_hoop.normalize().x * to_target.normalize().x +
                           to_hoop.normalize().y * to_target.normalize().y)
                    if dot > 0.95:  # < ~18 degrees
                        context['rush_quality'] = 'straight'
                        context['has_rush'] = True
                    elif dot > 0.7:  # < ~45 degrees
                        context['rush_quality'] = 'slight_cut'
                        context['has_rush'] = True
                    else:
                        context['rush_quality'] = 'fine_cut'
                        context['has_rush'] = False  # Risky
            context['rush_distance'] = (target_pos - ball.position).magnitude()

        # 3. BREAK BUILDING - already computed above in base context

        # 4. SHOT QUALITY (Rules of Thumb: short strokes are better)
        if shot_distance < 3.0:
            context['stroke_length'] = 'short'
        elif shot_distance < 7.0:
            context['stroke_length'] = 'medium'
        else:
            context['stroke_length'] = 'long'

        # 5. POSITIONAL QUALITY
        # Check if ball is in middle of court (bad) or near boundary (good for defense)
        court_center = Vector2(14, 17.5)  # Approximate center
        dist_to_center = (ball.position - court_center).magnitude()

        # Near boundary = defensive (good for leaves)
        dist_to_boundary = min(
            ball.position.x,  # West
            28 - ball.position.x,  # East
            ball.position.y,  # South
            35 - ball.position.y  # North
        )

        context['at_boundary'] = dist_to_boundary < 2.0
        context['in_middle_of_court'] = dist_to_center < 6.0 and dist_to_boundary > 5.0

        # Check if near opponent's hoop (bad for leaves)
        opponent_colors = ['red', 'yellow'] if ball.color in ['blue', 'black'] else ['blue', 'black']
        for opp_color in opponent_colors:
            if opp_color in balls:
                opp_ball = balls[opp_color]
                opp_hoop = self.court.get_hoop_for_ball(opp_ball.hoops_run)
                if opp_hoop:
                    dist_to_opp_hoop = (ball.position - opp_hoop.position).magnitude()
                    if dist_to_opp_hoop < 4.0:
                        context['near_opponent_hoop'] = True
                        break

        # consecutive_hoops and steps_since_last_hoop already set in base context above

        return context

    def _count_break_balls(self, striker: Ball, balls: Dict[str, Ball], target_hoop) -> int:
        """
        Count balls in useful break positions (controllable, not on boundary).

        A ball is "in the break" if:
        1. It's within roqueting distance of striker (< 15 yards)
        2. It's not on/near the boundary (> 2 yards from edge)
        3. It's not wired from striker (simplified: no hoop between)
        """
        count = 1  # Striker always counts
        if not target_hoop:
            return count

        for color, ball in balls.items():
            if color == striker.color:
                continue

            # Check boundary distance - must be off the boundary
            dist_to_boundary = min(
                ball.position.x, 28 - ball.position.x,
                ball.position.y, 35 - ball.position.y
            )
            if dist_to_boundary < 2.0:
                continue  # Ball on boundary - not useful

            # Check distance to striker - must be within roqueting range
            dist_to_striker = (ball.position - striker.position).magnitude()
            if dist_to_striker > 15.0:
                continue  # Too far to be part of active break

            count += 1

        return min(count, 4)  # Cap at 4-ball break

    def _has_pilot(self, balls: Dict[str, Ball], hoop_position: Vector2,
                   striker_color: str, radius: float = 3.0,
                   was_pilot: bool = False) -> bool:
        """
        Check if there's a pilot ball near the current hoop.

        A pilot is a ball positioned near the hoop that can be used for:
        - A croquet stroke to position another ball
        - An approach shot after roqueting

        Pilots should be within ~3 yards of hoop (closer than pioneers).

        HYSTERESIS (per Peter's feedback):
        - Create at radius (3y)
        - Lose at radius + 1y (4y)
        This prevents flip-flopping on boundary cases.
        """
        # Use hysteresis: tighter threshold to gain, looser to lose
        threshold = radius if not was_pilot else radius + 1.0

        for color, ball in balls.items():
            if color == striker_color:
                continue
            dist = (ball.position - hoop_position).magnitude()
            if dist < threshold:
                return True
        return False

    def _has_rush_to_position(self, striker: Ball, balls: Dict[str, Ball],
                              target_position: Vector2, max_cut_angle: float = 30.0) -> bool:
        """
        Check if striker has a rush (two balls together) toward a target position.

        A good rush requires:
        1. A ball within 3 yards of striker
        2. Alignment such that hitting it sends it toward target
        3. Cut angle < max_cut_angle degrees (straight rush is best)

        Returns True if a usable rush exists.
        """
        import math

        for color, ball in balls.items():
            if color == striker.color:
                continue

            # Check distance - must be close for a controllable rush
            dist = (ball.position - striker.position).magnitude()
            if dist > 3.0 or dist < 0.1:
                continue

            # Calculate rush direction (striker -> rush ball)
            rush_dir = (ball.position - striker.position).normalize()

            # Calculate target direction (rush ball -> target)
            target_dir = (target_position - ball.position)
            if target_dir.magnitude() < 0.1:
                continue  # Already at target
            target_dir = target_dir.normalize()

            # Calculate cut angle (angle between rush direction and target direction)
            dot = rush_dir.dot(target_dir)
            dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability
            angle_rad = math.acos(dot)
            angle_deg = math.degrees(angle_rad)

            # Good rush if cut angle is small
            if angle_deg < max_cut_angle:
                return True

        return False

    def _get_opponent_separation(self, striker: Ball, balls: Dict[str, Ball]) -> float:
        """
        Calculate how separated the opponent balls are.

        Returns distance between opponent balls in yards.
        Higher = better for us (opponents can't easily get a 4-ball break).
        """
        # Determine which side striker is on
        if striker.color in ['blue', 'black']:
            opp_colors = ['red', 'yellow']
        else:
            opp_colors = ['blue', 'black']

        opp_balls = [balls[c] for c in opp_colors if c in balls]
        if len(opp_balls) < 2:
            return 0.0

        return (opp_balls[0].position - opp_balls[1].position).magnitude()

    def _get_cluster_quality(self, striker: Ball, balls: Dict[str, Ball]) -> float:
        """
        Measure how tightly clustered OUR SIDE'S balls are (not opponent's).

        Per Peter's feedback: cluster quality should measure striker + partner + at most
        one useful opponent ball. Using all 4 might reward positions that help opponents.

        Returns a quality score 0-1 where:
        - 1.0 = our balls + one useful ball within 5 yards (tight, controllable break)
        - 0.5 = average ~10 yards apart
        - 0.0 = balls scattered (> 15 yards average)
        """
        # Determine our side
        if striker.color in ['blue', 'black']:
            our_colors = ['blue', 'black']
            opp_colors = ['red', 'yellow']
        else:
            our_colors = ['red', 'yellow']
            opp_colors = ['blue', 'black']

        # Get our balls
        our_positions = [balls[c].position for c in our_colors if c in balls]

        # Find closest opponent ball (potential break ball)
        closest_opp_dist = float('inf')
        closest_opp_pos = None
        for c in opp_colors:
            if c in balls:
                dist = (balls[c].position - striker.position).magnitude()
                if dist < closest_opp_dist and dist < 10.0:  # Within usable range
                    closest_opp_dist = dist
                    closest_opp_pos = balls[c].position

        # Positions to consider: our balls + closest opponent (if useful)
        positions = our_positions.copy()
        if closest_opp_pos is not None:
            positions.append(closest_opp_pos)

        if len(positions) < 2:
            return 0.0

        # Calculate average pairwise distance
        total_dist = 0.0
        pairs = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                total_dist += (positions[i] - positions[j]).magnitude()
                pairs += 1

        if pairs == 0:
            return 0.0

        avg_dist = total_dist / pairs

        # Convert to 0-1 score (5 yards = 1.0, 15 yards = 0.0)
        quality = max(0.0, min(1.0, (15.0 - avg_dist) / 10.0))
        return quality

    def _has_pioneer_near(self, balls: Dict[str, Ball], position: Vector2,
                          striker_color: str, radius: float = 5.0,
                          was_pioneer: bool = False) -> bool:
        """
        Check if there's a ball near a position that could be a pioneer.

        HYSTERESIS (per Peter's feedback):
        - Create at radius (5y for next, 7y for next-but-one)
        - Lose at radius + 1.5y
        This prevents flip-flopping on boundary cases.
        """
        # Use hysteresis: tighter threshold to gain, looser to lose
        threshold = radius if not was_pioneer else radius + 1.5

        for color, ball in balls.items():
            if color == striker_color:
                continue
            dist = (ball.position - position).magnitude()
            if dist < threshold:
                return True
        return False

    def _simulate_physics(self, balls: Dict[str, Ball], physics: PhysicsEngine):
        """Simulate physics until balls stop."""
        max_steps = 300
        dt = 1.0 / 60.0

        for _ in range(max_steps):
            physics.update(balls, dt)
            if physics.are_all_balls_stopped(balls):
                break

    def _simulate_physics_with_detection(
        self,
        balls: Dict[str, Ball],
        physics: PhysicsEngine,
        striker_color: str
    ) -> List[Dict]:
        """
        Simulate physics until balls stop and track collisions.

        Returns list of collision events involving the striker.
        Also checks for hoop runs during simulation.
        """
        max_steps = 300
        dt = 1.0 / 60.0
        collisions = []
        striker = balls[striker_color]

        for _ in range(max_steps):
            events = physics.update(balls, dt)

            # Track ball-ball collisions
            for event in events:
                if event.get('type') == 'ball_collision':
                    ball1 = event.get('ball1')
                    ball2 = event.get('ball2')
                    # Check if striker was involved
                    if ball1 == striker_color or ball2 == striker_color:
                        collisions.append(event)

            # Check for hoop run during each physics step
            self._check_hoop_run(striker)

            # Check for peg out during physics step (rover hitting peg)
            if striker.hoops_run >= 12 and not striker.has_pegged_out:
                self._check_peg_out(striker)

            if physics.are_all_balls_stopped(balls):
                break

        return collisions

    def _check_hoop_run(self, ball: Ball) -> bool:
        """
        Check if ball has run its next hoop.

        Uses ball's shot_start_position and current position to detect
        if it crossed through the hoop in the correct direction.
        """
        target_hoop = self.court.get_hoop_for_ball(ball.hoops_run)
        if target_hoop is None:
            return False

        hoop_pos = target_hoop.position
        start_pos = ball.shot_start_position

        # Safety check - shot_start_position might not be set
        if start_pos is None:
            return False

        # Check if ball path came near the hoop
        movement = ball.position - start_pos
        move_len = movement.magnitude()

        if move_len < 0.1:
            return False

        move_dir = movement.normalize()

        # Project hoop position onto the ball's path
        to_hoop = hoop_pos - start_pos
        projection = to_hoop.dot(move_dir)

        # Clamp projection to the actual path length
        projection = max(0, min(move_len, projection))

        # Find closest point on path to hoop
        closest_point = start_pos + move_dir * projection
        min_dist_to_hoop = (closest_point - hoop_pos).magnitude()

        # Ball must have passed within 0.75 yards of hoop center
        if min_dist_to_hoop > 0.75:
            return False

        # Get the required direction for this hoop
        required_dir = target_hoop.direction

        # Must be moving in roughly the correct direction (within ~45 degrees)
        dot = move_dir.dot(required_dir)
        if dot < 0.7:
            return False

        # Check if ball crossed the hoop plane
        start_dist = to_hoop.dot(required_dir)
        curr_to_hoop = hoop_pos - ball.position
        curr_dist = curr_to_hoop.dot(required_dir)

        # Ball crossed the plane if started on approach side and ended past
        crossed_plane = start_dist > 0 and curr_dist <= 0

        if crossed_plane:
            # Check if ball passed within hoop width at the crossing point
            perp_dir = Vector2(-required_dir.y, required_dir.x)
            perp_dist = abs((closest_point - hoop_pos).dot(perp_dir))

            if self.verbose:
                print(f"    -> Ball crossed hoop {target_hoop.number} plane! perp_dist={perp_dist:.2f} (need < 0.5)")

            # Ball must pass through the actual hoop gap (0.65 yards tolerance for training)
            if perp_dist < 0.65:
                if self.verbose:
                    print(f"  [SUCCESS] {ball.color} RAN HOOP {target_hoop.number}!")
                ball.run_hoop()
                return True

        return False

    def _check_peg_out(self, ball: Ball) -> bool:
        """
        Check if a rover ball has hit the peg to finish.

        Ball must be a rover (hoops_run >= 12) and be near the peg.
        """
        # Only rovers can peg out
        if ball.hoops_run < 12:
            return False

        # Peg is at center of court (14, 17.5)
        peg_position = Vector2(14, 17.5)

        # Check if ball is very close to peg (hit it)
        dist_to_peg = (ball.position - peg_position).magnitude()

        if dist_to_peg < 1.0:  # Ball hit the peg (within 1 yard)
            ball.has_pegged_out = True
            if self.verbose:
                print(f"  [PEG OUT] {ball.color} hit the peg at distance {dist_to_peg:.2f}!")
            return True

        return False

    def _create_balls(self) -> Dict[str, Ball]:
        """Create balls with starting positions.

        20% of games start with rover scenarios to train peg out behavior.
        """
        import random
        balls = {}

        if random.random() < 0.5:
            a_baulk = ["blue", "black"]
            b_baulk = ["red", "yellow"]
        else:
            a_baulk = ["red", "yellow"]
            b_baulk = ["blue", "black"]

        # 20% chance to start with a rover scenario
        rover_scenario = random.random() < 0.2

        for color in ["blue", "black", "red", "yellow"]:
            if color in a_baulk:
                x = random.uniform(2, 12)
                y = 1
            else:
                x = random.uniform(16, 26)
                y = 34
            balls[color] = Ball(color, (x, y))

            # In rover scenario, start one ball (blue) as a rover near the peg
            if rover_scenario and color == "blue":
                # Place rover near peg (center of court)
                balls[color].position = Vector2(
                    random.uniform(10, 18),  # Near peg x
                    random.uniform(13, 22)   # Near peg y
                )
                balls[color].hoops_run = 12  # Make it a rover
                balls[color].is_rover = True  # IMPORTANT: Set is_rover flag!
                if self.verbose:
                    print(f"  [ROVER SCENARIO] {color} starts as rover at {balls[color].position}")

        return balls

    def _evaluate(self, num_games: int = 10) -> Dict:
        """
        Evaluate current policy without exploration (pure greedy).

        Per Peter's recommendation: snapshot evaluation with epsilon=0
        to get unbiased performance metrics separate from exploration noise.

        NOTE: We pass training=False to _run_episode() instead of hacking
        the config epsilon values. This ensures:
        1. select_action() uses epsilon=0 (greedy)
        2. No transitions are stored to replay buffer
        3. No train_step() calls (which would trigger LR decay at wrong epsilon)

        Returns:
            Dict with greedy eval metrics for plateau detection
        """
        # Capture LR before eval to verify it doesn't change (per Peter's verification suggestion)
        lr_before = self.dqn.optimizer.param_groups[0]['lr']

        print(f"\n  Evaluating over {num_games} games (greedy)...")

        total_bb_hoops = 0  # Only blue_black side hoops (max 24)
        total_turns = 0
        wins = 0

        # TACTICAL KPI TRACKING (per Peter's recommendation)
        # Track tactical features during eval to see if policy is learning the right patterns
        tactical_kpis = {
            'pilot_rate': 0,           # % of positions with pilot at current hoop
            'pioneer_next_rate': 0,    # % of positions with pioneer at next hoop
            'pioneer_next2_rate': 0,   # % of positions with pioneer at next-but-one
            'rush_ready_rate': 0,      # % of positions with rush to next hoop available
            'avg_cluster_quality': 0,  # Average cluster quality score
            'avg_break_balls': 0,      # Average break balls in position
            'avg_opp_separation': 0,   # Average opponent ball separation
            'sample_count': 0,         # Total position samples
            # Budget tracking (per Peter's feedback)
            'total_budget_used': 0.0,  # Sum of tactical budget used across all turns
            'turns_hit_bonus_cap': 0,  # Number of turns that hit +3.0 cap
            'turns_hit_penalty_cap': 0, # Number of turns that hit -2.0 cap
            'turn_count': 0            # Total turns for averaging
        }

        for _ in range(num_games):
            # Run episode in evaluation mode (no training, pure greedy)
            # Pass tactical_kpis dict to accumulate stats
            result = self._run_episode(training=False, tactical_kpis=tactical_kpis)
            total_bb_hoops += result.bb_hoops  # Only our side's hoops
            total_turns += result.turns
            if result.winner == "blue_black":
                wins += 1

        avg_hoops = total_bb_hoops / num_games
        avg_turns = total_turns / num_games
        win_rate = wins / num_games * 100

        # Verify LR didn't change during eval (confirms fix is working)
        lr_after = self.dqn.optimizer.param_groups[0]['lr']
        if lr_before != lr_after:
            print(f"  WARNING: LR changed during eval! {lr_before:.2e} -> {lr_after:.2e}")
            print(f"  This indicates train_step() was called during eval - BUG!")

        print(f"  Greedy eval: Hoops={avg_hoops:.1f}, "
              f"Turns={avg_turns:.1f}, Win={win_rate:.0f}%")

        # Print tactical KPIs (per Peter's recommendation)
        if tactical_kpis['sample_count'] > 0:
            n = tactical_kpis['sample_count']
            print(f"  Tactical KPIs: pilot={100*tactical_kpis['pilot_rate']/n:.0f}%, "
                  f"pioneer={100*tactical_kpis['pioneer_next_rate']/n:.0f}%, "
                  f"rush={100*tactical_kpis['rush_ready_rate']/n:.0f}%, "
                  f"cluster={tactical_kpis['avg_cluster_quality']/n:.2f}, "
                  f"break_balls={tactical_kpis['avg_break_balls']/n:.1f}")

        # Print budget tracking stats (per Peter's feedback)
        if tactical_kpis['turn_count'] > 0:
            t = tactical_kpis['turn_count']
            avg_budget = tactical_kpis['total_budget_used'] / t
            pct_bonus_cap = 100 * tactical_kpis['turns_hit_bonus_cap'] / t
            pct_penalty_cap = 100 * tactical_kpis['turns_hit_penalty_cap'] / t
            print(f"  Budget: avg_used={avg_budget:.2f}/turn, "
                  f"hit_bonus_cap={pct_bonus_cap:.0f}%, hit_penalty_cap={pct_penalty_cap:.0f}%")

        # Report planner agreement if enabled
        if self.planner and self.planner_log:
            agreements = sum(1 for x in self.planner_log if x['agreement'])
            total = len(self.planner_log)
            agree_pct = 100 * agreements / total if total > 0 else 0
            cached_pct = 100 * sum(1 for x in self.planner_log if x['cached']) / total if total > 0 else 0
            print(f"  Planner: NN-LLM agreement={agree_pct:.1f}% ({agreements}/{total}), cached={cached_pct:.0f}%")

            # Clear log for next eval period
            self.planner_log = []

        print()

        # Store last eval metrics for checkpoint saving (include tactical KPIs)
        self.last_eval_metrics = {
            'greedy_avg_hoops': avg_hoops,
            'greedy_avg_turns': avg_turns,
            'greedy_win_rate': win_rate
        }
        # Add tactical KPIs to metrics if we have samples
        if tactical_kpis['sample_count'] > 0:
            n = tactical_kpis['sample_count']
            self.last_eval_metrics['pilot_rate'] = tactical_kpis['pilot_rate'] / n
            self.last_eval_metrics['pioneer_rate'] = tactical_kpis['pioneer_next_rate'] / n
            self.last_eval_metrics['rush_rate'] = tactical_kpis['rush_ready_rate'] / n
            self.last_eval_metrics['cluster_quality'] = tactical_kpis['avg_cluster_quality'] / n
            self.last_eval_metrics['break_balls'] = tactical_kpis['avg_break_balls'] / n
        # Add budget tracking stats
        if tactical_kpis['turn_count'] > 0:
            t = tactical_kpis['turn_count']
            self.last_eval_metrics['avg_budget_used'] = tactical_kpis['total_budget_used'] / t
            self.last_eval_metrics['pct_bonus_cap_hit'] = tactical_kpis['turns_hit_bonus_cap'] / t
            self.last_eval_metrics['pct_penalty_cap_hit'] = tactical_kpis['turns_hit_penalty_cap'] / t

        return self.last_eval_metrics

    def save_model(self, path: str = None):
        """Save trained model."""
        if path is None:
            path = str(self.dqn.save_dir / "model.pt")
        self.dqn.online_net.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load pre-trained model."""
        self.dqn.online_net = CroquetNet.load(path)
        self.dqn.target_net.load_state_dict(self.dqn.online_net.state_dict())
        print(f"Model loaded from {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train neural network for croquet AI"
    )
    parser.add_argument(
        "num_episodes",
        type=int,
        nargs="?",
        default=100,
        help="Number of episodes to train (default: 100)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50,
        help="Episodes between evaluations (default: 50)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100,
        help="Episodes between checkpoints (default: 100)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--expert",
        action="store_true",
        help="Use comprehensive expert tactical reward shaping (Aiton + Wylie + Oxford Croquet)"
    )
    parser.add_argument(
        "--aiton",
        action="store_true",
        help="(Legacy) Alias for --expert"
    )
    parser.add_argument(
        "--dueling",
        action="store_true",
        help="Use dueling network architecture (separates V and A)"
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=1,
        help="N-step returns for better credit assignment (default: 1, recommended: 3-5)"
    )
    parser.add_argument(
        "--prioritized",
        action="store_true",
        help="Use prioritized experience replay"
    )
    parser.add_argument(
        "--epsilon-decay",
        type=int,
        default=200000,
        help="Steps over which epsilon decays from 1.0 to 0.05 (default: 200000)"
    )
    parser.add_argument(
        "--min-buffer",
        type=int,
        default=10000,
        help="Min replay buffer size before training starts (default: 10000)"
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default=None,
        help="Path to demonstration file for behavior cloning pretrain (e.g., ai_data/neural/demos.pt)"
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=20,
        help="Number of epochs for behavior cloning pretrain (default: 20)"
    )
    parser.add_argument(
        "--pretrain-lr",
        type=float,
        default=None,
        help="Learning rate AFTER pretrain for RL phase (lower = preserve pretrain better). "
             "Recommended: 1e-5 to 5e-5. If not set, uses --learning-rate."
    )
    parser.add_argument(
        "--pretrain-epsilon",
        type=float,
        default=None,
        help="Starting epsilon AFTER pretrain (lower = less exploration, preserve policy). "
             "Recommended: 0.3 to 0.5. If not set, uses 1.0."
    )
    parser.add_argument(
        "--no-demo-buffer",
        action="store_true",
        help="Don't add demo transitions to replay buffer (disable DQfD-style mixing)"
    )
    parser.add_argument(
        "--demo-frac",
        type=float,
        default=0.25,
        help="Guaranteed fraction of demos per training batch (default: 0.25 = 25%%). "
             "Uses DemoMixingReplayBuffer which protects demos from being overwritten."
    )
    parser.add_argument(
        "--planner",
        action="store_true",
        help="Enable LLM tactical planner as advisor during evaluation (requires ANTHROPIC_API_KEY)"
    )

    args = parser.parse_args()

    # --aiton is now an alias for --expert
    use_expert = args.expert or args.aiton

    print("=" * 60)
    print("NEURAL NETWORK CROQUET TRAINING")
    print("=" * 60)
    print(f"Device: {get_device()}")

    # Show architecture options
    if args.dueling:
        print("Using DUELING network architecture (V + A streams)")
    if args.n_step > 1:
        print(f"Using {args.n_step}-STEP returns for credit assignment")
    if args.prioritized:
        print("Using PRIORITIZED experience replay")

    # Show exploration settings
    print(f"Epsilon decay: {args.epsilon_decay} steps (floor ~ep{args.epsilon_decay // 200})")
    print(f"Replay warmup: {args.min_buffer} transitions before learning")

    if use_expert:
        print("Using EXPERT tactical reward shaping:")
        print("  - Aiton approach quality (1 yard ideal)")
        print("  - Wylie rush tactics (avoid fine cuts)")
        print("  - Oxford Croquet Rules of Thumb")
        print("  - Defensive play principles")
        print("  - 4-ball break building rewards")
    print()

    # Enable demo mixing if pretrain with demos is specified
    use_demo_mixing = args.pretrain is not None and not args.no_demo_buffer

    # Configure training
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_dueling=args.dueling,
        n_step=args.n_step,
        use_prioritized=args.prioritized,
        epsilon_decay=args.epsilon_decay,
        min_buffer_size=args.min_buffer,
        use_demo_mixing=use_demo_mixing,
        demo_fraction=args.demo_frac,
    )

    # Create trainer
    trainer = NeuralTrainer(
        config=train_config,
        verbose=args.verbose,
        use_expert_shaping=use_expert,
        use_planner=args.planner
    )

    # Load checkpoint if specified
    if args.checkpoint:
        trainer.dqn.load_checkpoint(args.checkpoint)
        print(f"Resumed from {args.checkpoint}")

    # Behavior cloning pretrain if specified
    if args.pretrain:
        print()
        print("=" * 60)
        print("BEHAVIOR CLONING PRETRAIN")
        print("=" * 60)

        # Show post-pretrain settings
        if args.pretrain_lr:
            print(f"Post-pretrain LR: {args.pretrain_lr:.2e} (prevents catastrophic forgetting)")
        if args.pretrain_epsilon:
            print(f"Post-pretrain epsilon: {args.pretrain_epsilon:.2f} (preserves good policy)")
        if not args.no_demo_buffer:
            print(f"DQfD-style: Demo transitions mixed into buffer ({args.demo_frac:.0%} per batch)")
        print()

        pretrain_stats = trainer.dqn.pretrain_from_demos(
            demo_path=args.pretrain,
            epochs=args.pretrain_epochs,
            verbose=True,
            post_pretrain_lr=args.pretrain_lr,
            post_pretrain_epsilon=args.pretrain_epsilon,
            keep_demos_in_buffer=not args.no_demo_buffer
        )
        print(f"Pretrain complete: avg loss = {pretrain_stats['average_loss']:.4f}, "
              f"accuracy = {pretrain_stats['final_accuracy']:.1f}%")
        print()

    # Train
    stats = trainer.train(
        num_episodes=args.num_episodes,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq
    )

    # Save final model
    trainer.save_model()

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Average reward: {stats['avg_episode_reward']:.2f}")
    print(f"Model saved to: ai_data/neural/model.pt")


if __name__ == "__main__":
    main()
