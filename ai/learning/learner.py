"""
Croquet Learner - Self-play learning system.

Combines position evaluation, shot simulation, and tactical rules
with experience replay to improve play over time.
"""
import json
import os
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from models.ball import Ball, Vector2
from models.court import Court
from .position_evaluator import PositionEvaluator
from .shot_simulator import ShotSimulator
from .tactical_rules import TacticalRules, TacticalAdvice
from .leave_patterns import get_pattern_library, LeaveType


@dataclass
class Experience:
    """Record of a single decision and its outcome."""
    # Situation
    ball_positions: Dict[str, Tuple[float, float]]
    ball_hoops: Dict[str, int]
    striker_color: str
    deadness: Dict[str, List[str]]

    # Decision made
    shot_type: str
    target_angle: float
    target_power: float
    tactical_rules_used: List[str]

    # Outcome
    hoop_run: bool
    roqueted_ball: Optional[str]
    position_value_change: float
    turn_continued: bool

    # Learning signal
    reward: float

    # Additional tracking (with defaults at end)
    target_hoop_num: int = 0  # Which hoop was being targeted (1-12)

    # Enhanced tracking for better learning
    stroke_type: str = "standard"  # "standard", "croquet", "continuation"
    approach_distance: float = 0.0  # Distance to hoop when attempting hoop run
    approach_angle: float = 0.0  # Angle quality (0-1) when attempting hoop run
    break_sequence_id: int = 0  # Links shots within a break for credit assignment
    balls_in_break: int = 0  # How many balls controlled in this break
    pioneer_placed: bool = False  # Did we place a pioneer?
    rush_achieved: bool = False  # Did we get a useful rush?


class CroquetLearner:
    """
    Self-play learning system for croquet AI.

    Learns by:
    1. Recording experiences during play
    2. Evaluating outcomes vs predictions
    3. Adjusting weights based on what worked
    """

    def __init__(self, save_dir: str = "ai_data"):
        """
        Initialize the learner.

        Args:
            save_dir: Directory to save learning data
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.position_evaluator = PositionEvaluator()
        self.shot_simulator = ShotSimulator(num_simulations=30)
        self.tactical_rules = TacticalRules()

        self.experiences: List[Experience] = []
        self.game_outcomes: List[Dict] = []  # Win/loss records

        # Learning parameters
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        self.momentum = 0.9  # Momentum for weight updates to prevent oscillation

        # Learned adjustments
        self.power_adjustment = 1.0  # Multiplier for shot power
        self.hoop_success_rates = {i: 0.0 for i in range(1, 13)}  # Per-hoop success tracking
        self.hoop_attempts = {i: 0 for i in range(1, 13)}  # Only counts actual approach attempts

        # Per-hoop strategy modifiers (base weights + hoop-specific adjustments)
        self.hoop_modifiers = {i: {} for i in range(1, 13)}

        # Weight update velocities for momentum
        self.weight_velocities = {k: 0.0 for k in PositionEvaluator().weights.keys()}

        # Statistics
        self.games_played = 0
        self.total_hoops_run = 0
        self.total_roquets = 0

        # Enhanced learning tracking
        self.current_break_id = 0  # Increments each time a turn starts
        self.current_break_experiences = []  # Experiences in current break
        self.break_stats = {  # Track what makes breaks successful
            'hoops_per_break': [],  # How many hoops run per break
            'avg_break_length': 0.0,
            'best_break': 0,
        }

        # Approach pattern learning (key = hoop_num, value = list of (distance, angle, success))
        self.approach_patterns = {i: [] for i in range(1, 13)}

        # Croquet shot effectiveness tracking
        self.croquet_shot_stats = {
            'pioneer_placements': {'attempts': 0, 'useful': 0},  # Useful = led to using that pioneer
            'rush_setups': {'attempts': 0, 'achieved': 0},  # Got the rush we wanted
            'stop_shots': {'attempts': 0, 'success': 0},
            'roll_shots': {'attempts': 0, 'success': 0},
        }

        # Leave quality tracking (end of turn positions)
        self.leave_outcomes = []  # List of (leave_quality_score, opponent_scored_next)

        # Leave pattern library for bootstrapping
        self.leave_patterns = get_pattern_library()
        self.leave_pattern_matches = {  # Track how often we match standard patterns
            'NSL': {'matches': 0, 'opponent_scored': 0},
            'OSL': {'matches': 0, 'opponent_scored': 0},
            'DIAGONAL': {'matches': 0, 'opponent_scored': 0},
            'CUSTOM': {'matches': 0, 'opponent_scored': 0},
        }

        # Checkpointing configuration
        self.checkpoint_interval = 100  # Save checkpoint every N games
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Best metrics for milestone checkpointing
        self.best_metrics = {
            'best_break': 0,
            'best_avg_break': 0.0,
            'best_win_rate': 0.0,  # Over last 20 games
            'best_hoop_success': 0.0,  # Average across all hoops
        }

        # Load previous learning if exists
        self._load_state()

    def select_shot(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        skill_level: float = 0.8
    ) -> Tuple[float, float, str]:
        """
        Select a shot using learned knowledge.

        Args:
            striker: The ball to shoot
            all_balls: All balls on the court
            court: The court
            deadness: Deadness tracking
            skill_level: Player skill level

        Returns:
            Tuple of (angle, power, description)
        """
        # Get tactical advice
        advice_list = self.tactical_rules.get_advice(
            striker, all_balls, court, deadness
        )

        # Exploration: sometimes try random shots
        if random.random() < self.exploration_rate:
            return self._random_shot(striker, all_balls, court)

        # Use shot simulation to evaluate top tactical options
        best_shot = None
        best_value = float('-inf')

        for advice in advice_list[:3]:  # Evaluate top 3 suggestions
            angle, power = self._advice_to_shot(striker, advice, court)

            # Simulate outcomes
            outcomes = self.shot_simulator.simulate_shot(
                striker, angle, power, skill_level, all_balls, court
            )

            # Calculate expected value
            value = self._evaluate_outcomes(outcomes, striker, court)

            # Adjust by tactical priority
            value += advice.priority * 5

            if value > best_value:
                best_value = value
                best_shot = (angle, power, advice.reason)

        if best_shot:
            return best_shot

        # Fallback to first advice
        if advice_list:
            advice = advice_list[0]
            angle, power = self._advice_to_shot(striker, advice, court)
            return (angle, power, advice.reason)

        # Ultimate fallback
        return self._random_shot(striker, all_balls, court)

    def record_experience(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        deadness: Dict[str, set],
        shot_angle: float,
        shot_power: float,
        hoop_run: bool,
        roqueted: Optional[str],
        turn_continued: bool,
        old_position_value: float,
        new_position_value: float,
        shot_type: str = None,
        rules_applied: List[str] = None,
        # Enhanced tracking parameters
        stroke_type: str = "standard",  # "standard", "croquet", "continuation"
        approach_distance: float = 0.0,
        approach_angle: float = 0.0,
        pioneer_placed: bool = False,
        rush_achieved: bool = False,
        court: Court = None
    ):
        """
        Record an experience for learning.

        Args:
            striker: Ball that was shot
            all_balls: All balls (before shot)
            deadness: Deadness state (before shot)
            shot_angle: Angle of shot taken
            shot_power: Power of shot taken
            hoop_run: Whether a hoop was run
            roqueted: Ball that was roqueted (if any)
            turn_continued: Whether the turn continued
            old_position_value: Position value before shot
            new_position_value: Position value after shot
            shot_type: Type of shot taken (from tactical rules)
            rules_applied: List of tactical rules that were applied
            stroke_type: Type of stroke (standard, croquet, continuation)
            approach_distance: Distance to hoop when attempting hoop run
            approach_angle: Angle quality (0-1) for hoop approach
            pioneer_placed: Whether a pioneer was placed (in croquet shot)
            rush_achieved: Whether a useful rush was achieved
            court: Court object for additional calculations
        """
        if rules_applied is None:
            rules_applied = []
        # Determine target hoop (1-12)
        target_hoop_num = striker.hoops_run + 1 if striker.hoops_run < 12 else 0

        # Track per-hoop statistics - ONLY count actual approach attempts
        # Fix for hoop 1 tracking bug: only count when ball is in reasonable approach position
        is_actual_hoop_attempt = False
        if target_hoop_num > 0 and court is not None:
            target_hoop = court.get_hoop_for_ball(striker.hoops_run)
            if target_hoop:
                dist_to_hoop = (striker.position - target_hoop.position).magnitude()
                # Only count as attempt if:
                # 1. Within 5 yards of hoop (reasonable approach distance), OR
                # 2. Shot type indicates hoop attempt, OR
                # 3. Approach distance was tracked (meaning it was an approach shot)
                if dist_to_hoop <= 5.0 or shot_type == "hoop_run_opportunity" or approach_distance > 0:
                    is_actual_hoop_attempt = True

        if target_hoop_num > 0 and is_actual_hoop_attempt:
            self.hoop_attempts[target_hoop_num] = self.hoop_attempts.get(target_hoop_num, 0) + 1
            if hoop_run:
                # Update success rate with exponential moving average
                old_rate = self.hoop_success_rates.get(target_hoop_num, 0.0)
                self.hoop_success_rates[target_hoop_num] = old_rate * 0.95 + 0.05  # Success contributes 0.05
            else:
                old_rate = self.hoop_success_rates.get(target_hoop_num, 0.0)
                self.hoop_success_rates[target_hoop_num] = old_rate * 0.95  # Decay toward 0

        # Calculate reward
        reward = 0.0
        if hoop_run:
            reward += 10.0
            self.total_hoops_run += 1
        if roqueted:
            reward += 5.0
            self.total_roquets += 1
        if turn_continued:
            reward += 2.0

        # Position improvement
        reward += (new_position_value - old_position_value) * 0.5

        # Use passed shot_type if provided, otherwise try learner's own rules (fallback)
        actual_shot_type = shot_type if shot_type else (
            self.tactical_rules.rules_applied[0] if self.tactical_rules.rules_applied else "unknown"
        )
        actual_rules = rules_applied if rules_applied else self.tactical_rules.rules_applied.copy()

        # Track approach patterns for hoop attempts
        if shot_type == "hoop_run_opportunity" or approach_distance > 0:
            if target_hoop_num > 0 and target_hoop_num <= 12:
                # Record this approach attempt
                self.approach_patterns[target_hoop_num].append({
                    'distance': approach_distance,
                    'angle': approach_angle,
                    'success': hoop_run,
                    'power': shot_power,
                })
                # Keep last 50 attempts per hoop
                if len(self.approach_patterns[target_hoop_num]) > 50:
                    self.approach_patterns[target_hoop_num] = self.approach_patterns[target_hoop_num][-50:]

        # Track croquet shot effectiveness
        if stroke_type == "croquet":
            if pioneer_placed:
                self.croquet_shot_stats['pioneer_placements']['attempts'] += 1
            if rush_achieved:
                self.croquet_shot_stats['rush_setups']['achieved'] += 1

        # Count balls controlled in this break
        dead_on = deadness.get(striker.color, set())
        balls_in_break = 3 - len(dead_on)  # Max 3 other balls, minus those we're dead on

        experience = Experience(
            ball_positions={c: (b.position.x, b.position.y) for c, b in all_balls.items()},
            ball_hoops={c: b.hoops_run for c, b in all_balls.items()},
            striker_color=striker.color,
            deadness={c: list(d) for c, d in deadness.items()},
            target_hoop_num=target_hoop_num,
            shot_type=actual_shot_type,
            target_angle=shot_angle,
            target_power=shot_power,
            tactical_rules_used=actual_rules,
            hoop_run=hoop_run,
            roqueted_ball=roqueted,
            position_value_change=new_position_value - old_position_value,
            turn_continued=turn_continued,
            reward=reward,
            # Enhanced fields
            stroke_type=stroke_type,
            approach_distance=approach_distance,
            approach_angle=approach_angle,
            break_sequence_id=self.current_break_id,
            balls_in_break=balls_in_break,
            pioneer_placed=pioneer_placed,
            rush_achieved=rush_achieved,
        )

        self.experiences.append(experience)

        # Also track in current break for temporal credit assignment
        self.current_break_experiences.append(experience)

        # Limit experience buffer
        if len(self.experiences) > 10000:
            self.experiences = self.experiences[-5000:]

        # Learn and save continuously - don't wait for game completion!
        # Learn every 50 experiences
        if len(self.experiences) % 50 == 0:
            self._learn_from_experiences()

        # Save every 100 experiences (about every 25 turns)
        if len(self.experiences) % 100 == 0:
            self._save_state()
            print(f"  [Auto-save: {len(self.experiences)} exp, {self.total_hoops_run} hoops, power={self.power_adjustment:.2f}]")

        # Also learn immediately when something good happens
        if hoop_run or roqueted:
            self._learn_from_experiences()

    def start_new_break(self):
        """Call when a new turn/break starts."""
        # End the previous break if there was one
        if self.current_break_experiences:
            self._end_break()

        self.current_break_id += 1
        self.current_break_experiences = []

    def _end_break(self):
        """Process the completed break for learning."""
        if not self.current_break_experiences:
            return

        # Count hoops run in this break
        hoops_in_break = sum(1 for e in self.current_break_experiences if e.hoop_run)
        self.break_stats['hoops_per_break'].append(hoops_in_break)
        self.break_stats['best_break'] = max(self.break_stats['best_break'], hoops_in_break)

        # Keep only last 100 break stats
        if len(self.break_stats['hoops_per_break']) > 100:
            self.break_stats['hoops_per_break'] = self.break_stats['hoops_per_break'][-100:]

        # Update average
        if self.break_stats['hoops_per_break']:
            self.break_stats['avg_break_length'] = (
                sum(self.break_stats['hoops_per_break']) / len(self.break_stats['hoops_per_break'])
            )

        # TEMPORAL CREDIT ASSIGNMENT
        # If this break ran hoops, give credit to earlier shots that set it up
        if hoops_in_break > 0:
            self._assign_break_credit(hoops_in_break)

    def _assign_break_credit(self, hoops_run: int):
        """
        Give credit to earlier shots in the break that enabled the hoops.

        Key insight: A good croquet shot that places a pioneer or sets up a rush
        should get credit when that setup is later used to run a hoop.
        """
        credit_bonus = hoops_run * 2.0  # Bonus per hoop run

        for i, exp in enumerate(self.current_break_experiences):
            # Pioneer placements get credit if hoops were later run
            if exp.pioneer_placed:
                exp.reward += credit_bonus * 0.5
                self.croquet_shot_stats['pioneer_placements']['useful'] += 1

            # Rush setups get credit
            if exp.rush_achieved:
                exp.reward += credit_bonus * 0.3

            # Earlier roquets in a successful break get credit (they kept it going)
            if exp.roqueted_ball and i < len(self.current_break_experiences) - 1:
                exp.reward += credit_bonus * 0.2

    def record_leave_quality(self, position_value: float, all_balls: Dict[str, Ball], court: Court):
        """
        Record the quality of a leave (position at end of turn).

        Call this when a turn ends to track defensive positioning.
        Also compares against standard leave patterns for learning.
        """
        # Convert ball positions to Vector2 dict for pattern matching
        actual_positions = {c: b.position for c, b in all_balls.items()}

        # Find best matching leave pattern
        best_pattern, similarity = self.leave_patterns.find_best_matching_pattern(actual_positions)

        pattern_type = 'CUSTOM'
        pattern_name = None
        if best_pattern and similarity > 0.5:  # Reasonable match threshold
            pattern_type = best_pattern.leave_type.name
            pattern_name = best_pattern.name

            # Track pattern matches
            if pattern_type in self.leave_pattern_matches:
                self.leave_pattern_matches[pattern_type]['matches'] += 1

        # Combine position value with pattern quality bonus
        pattern_bonus = 0.0
        if best_pattern and similarity > 0.5:
            # Bonus for matching a known good pattern
            pattern_bonus = similarity * best_pattern.quality_score * 5.0

        adjusted_value = position_value + pattern_bonus

        # Record the leave outcome
        self.leave_outcomes.append({
            'position_value': position_value,
            'adjusted_value': adjusted_value,
            'opponent_scored_next': None,  # Will be filled in by record_opponent_outcome
            'ball_positions': {c: (b.position.x, b.position.y) for c, b in all_balls.items()},
            'pattern_type': pattern_type,
            'pattern_name': pattern_name,
            'pattern_similarity': similarity,
        })

        # Keep last 100 leaves
        if len(self.leave_outcomes) > 100:
            self.leave_outcomes = self.leave_outcomes[-100:]

    def record_opponent_outcome(self, opponent_ran_hoop: bool):
        """Record whether the opponent scored after our leave."""
        if self.leave_outcomes and self.leave_outcomes[-1]['opponent_scored_next'] is None:
            leave = self.leave_outcomes[-1]
            leave['opponent_scored_next'] = opponent_ran_hoop

            # Track pattern effectiveness
            pattern_type = leave.get('pattern_type', 'CUSTOM')
            if pattern_type in self.leave_pattern_matches:
                if opponent_ran_hoop:
                    self.leave_pattern_matches[pattern_type]['opponent_scored'] += 1

            # Learn from this
            if not opponent_ran_hoop:
                # Good leave! Reinforce defensive positioning
                self.position_evaluator.weights['opponent_threatens_hoop'] -= self.learning_rate * 0.5

                # Extra reinforcement if we matched a standard pattern
                if pattern_type != 'CUSTOM' and leave.get('pattern_similarity', 0) > 0.6:
                    # Pattern worked well - slightly boost pattern matching
                    pass  # Future: could adjust pattern quality scores
            else:
                # Opponent scored - our leave wasn't good enough
                if pattern_type != 'CUSTOM':
                    # Pattern didn't work in this case - note for analysis
                    pass  # Future: could track what went wrong

    def record_game_outcome(self, winner_side: str, final_scores: Dict[str, int]):
        """
        Record the outcome of a game.

        Args:
            winner_side: "blue_black" or "red_yellow"
            final_scores: Dictionary of final scores per side
        """
        self.games_played += 1
        self.game_outcomes.append({
            "game_number": self.games_played,
            "winner": winner_side,
            "scores": final_scores
        })

        # Learn from recent experiences
        self._learn_from_experiences()

        # Check for milestone achievements and save checkpoints
        self._check_milestones_and_checkpoint()

        # Save periodically
        if self.games_played % 10 == 0:
            self._save_state()

    def _learn_from_experiences(self):
        """
        Apply learning from recorded experiences.

        Learns:
        1. All position evaluation weights
        2. Shot power calibration
        3. Approach angle preferences
        4. Per-hoop success patterns
        """
        if len(self.experiences) < 50:
            return  # Need some data

        recent = self.experiences[-500:]

        # === 1. Learn position weights from successful vs failed shots ===
        successful = [e for e in recent if e.reward > 5]
        failed = [e for e in recent if e.reward < 1]

        if len(successful) > 10 and len(failed) > 10:
            self._learn_position_weights(successful, failed)

        # === 2. Learn from shot type outcomes ===
        shot_type_data = {}
        for exp in recent:
            st = exp.shot_type
            if st not in shot_type_data:
                shot_type_data[st] = {'rewards': [], 'hoop_runs': 0, 'roquets': 0, 'count': 0}
            shot_type_data[st]['rewards'].append(exp.reward)
            shot_type_data[st]['count'] += 1
            if exp.hoop_run:
                shot_type_data[st]['hoop_runs'] += 1
            if exp.roqueted_ball:
                shot_type_data[st]['roquets'] += 1

        self._learn_from_shot_types(shot_type_data)

        # === 3. Learn power calibration ===
        self._learn_power_calibration(recent)

        # === 4. Learn approach patterns ===
        self._learn_approach_patterns(recent)

        # === 5. Learn from break sequences ===
        self._learn_from_breaks(recent)

        # === 6. Learn optimal approach distances/angles per hoop ===
        self._learn_optimal_approaches()

    def _learn_position_weights(self, successful: list, failed: list):
        """Learn position weights by comparing successful vs failed shots.

        Uses momentum-based updates to prevent oscillation:
        velocity = momentum * velocity + learning_rate * gradient
        weight = weight + velocity
        """
        # Collect weight adjustments (gradients)
        weight_adjustments = {k: 0.0 for k in self.position_evaluator.weights.keys()}

        # Hoop runs are the ultimate success - boost hoop-related weights
        hoop_run_exps = [e for e in successful if e.hoop_run]
        if len(hoop_run_exps) > 3:
            # Being close to hoop with good angle worked!
            weight_adjustments['threatens_hoop'] += 2.0
            weight_adjustments['approach_angle_quality'] += 1.0
            weight_adjustments['distance_to_next_hoop'] -= 0.5  # More negative = closer is better
            weight_adjustments['is_in_good_position'] += 1.0

        # Roquets lead to breaks - boost roquet-related weights
        roquet_exps = [e for e in successful if e.roqueted_ball]
        if len(roquet_exps) > 5:
            weight_adjustments['can_roquet_count'] += 1.0
            weight_adjustments['distance_to_partner'] -= 0.3  # Closer partner helps

        # Turn continuation is good
        continued_exps = [e for e in successful if e.turn_continued]
        if len(continued_exps) > 5:
            # Whatever led to continued turns is working
            avg_position_change = sum(e.position_value_change for e in continued_exps) / len(continued_exps)
            if avg_position_change > 0:
                # Position improvement correlated with turn continuation
                weight_adjustments['is_in_good_position'] += 0.5

        # Learn from failures too - what to avoid
        boundary_failures = [e for e in failed if e.position_value_change < -5]
        if len(boundary_failures) > 3:
            weight_adjustments['distance_from_boundary'] += 0.5

        # Apply momentum-based updates to prevent oscillation
        for weight_name, gradient in weight_adjustments.items():
            if gradient != 0.0:
                # Update velocity with momentum
                old_velocity = self.weight_velocities.get(weight_name, 0.0)
                new_velocity = self.momentum * old_velocity + self.learning_rate * gradient
                self.weight_velocities[weight_name] = new_velocity

                # Apply velocity to weight
                self.position_evaluator.weights[weight_name] += new_velocity

    def _learn_from_shot_types(self, shot_type_data: dict):
        """Adjust weights based on which shot types are working."""
        for shot_type, data in shot_type_data.items():
            if data['count'] < 5:
                continue

            avg_reward = sum(data['rewards']) / len(data['rewards'])
            hoop_rate = data['hoop_runs'] / data['count']
            roquet_rate = data['roquets'] / data['count']

            # Hoop run attempts
            if shot_type == "hoop_run_opportunity":
                if hoop_rate > 0.3:  # 30%+ success rate is good
                    self.position_evaluator.weights['threatens_hoop'] += self.learning_rate
                elif hoop_rate < 0.1:  # Less than 10% - maybe too aggressive
                    self.position_evaluator.weights['threatens_hoop'] -= self.learning_rate * 0.3

            # Approach shots
            elif shot_type == "hoop_approach":
                if avg_reward > 3:
                    self.position_evaluator.weights['approach_angle_quality'] += self.learning_rate * 0.5

            # Roquet attempts
            elif shot_type in ["rush_opportunity", "any_roquet"]:
                if roquet_rate > 0.4:
                    self.position_evaluator.weights['can_roquet_count'] += self.learning_rate
                elif roquet_rate < 0.15:
                    self.position_evaluator.weights['can_roquet_count'] -= self.learning_rate * 0.3

            # Clearance shots (defensive)
            elif shot_type == "clearance_needed":
                if avg_reward > 2:
                    self.position_evaluator.weights['opponent_threatens_hoop'] -= self.learning_rate  # More negative = more defensive

    def _learn_power_calibration(self, experiences: list):
        """Learn to calibrate shot power based on overshoot/undershoot."""
        # Track: did shots go too far or not far enough?
        # We can infer this from position_value_change and whether goals were achieved

        # Initialize power adjustment tracking if not exists
        if not hasattr(self, 'power_adjustment'):
            self.power_adjustment = 1.0  # Multiplier for shot power

        hoop_attempts = [e for e in experiences if e.shot_type == "hoop_run_opportunity"]
        if len(hoop_attempts) > 10:
            successes = [e for e in hoop_attempts if e.hoop_run]
            success_rate = len(successes) / len(hoop_attempts)

            # If we're missing hoops, analyze why
            if success_rate < 0.2:
                # Likely undershooting or overshooting
                # Check position changes to infer
                avg_change = sum(e.position_value_change for e in hoop_attempts) / len(hoop_attempts)
                if avg_change < -2:
                    # Position got worse - probably overshooting
                    self.power_adjustment *= 0.98
                else:
                    # Position okay but no hoop - probably undershooting
                    self.power_adjustment *= 1.02

                # Clamp to reasonable range
                self.power_adjustment = max(0.7, min(1.3, self.power_adjustment))

    def _learn_approach_patterns(self, experiences: list):
        """Learn which approach angles lead to successful hoop runs."""
        # Track success by situation
        hoop_runs = [e for e in experiences if e.hoop_run]

        if len(hoop_runs) > 5:
            # Hoops are being run - reinforce current approach
            self.position_evaluator.weights['approach_angle_quality'] += self.learning_rate * 0.5

            # Check if runs came from close or far
            # (We'd need to store more data to do this properly, but we can infer)
            avg_reward = sum(e.reward for e in hoop_runs) / len(hoop_runs)
            if avg_reward > 12:
                # High reward = probably good setup
                self.position_evaluator.weights['is_in_good_position'] += self.learning_rate

        # If no hoops being run, adjust down
        total = len(experiences)
        hoop_rate = len(hoop_runs) / total if total > 0 else 0

        if total > 100 and hoop_rate < 0.02:  # Less than 2% hoop rate
            # Need to be more aggressive about getting to hoops
            self.position_evaluator.weights['distance_to_next_hoop'] -= self.learning_rate * 0.5  # More negative = prefer closer

    def _learn_from_breaks(self, experiences: list):
        """Learn what makes breaks successful."""
        # Group experiences by break_sequence_id
        breaks = {}
        for exp in experiences:
            bid = exp.break_sequence_id
            if bid not in breaks:
                breaks[bid] = []
            breaks[bid].append(exp)

        # Analyze successful vs unsuccessful breaks
        successful_breaks = []
        failed_breaks = []

        for bid, exps in breaks.items():
            hoops = sum(1 for e in exps if e.hoop_run)
            if hoops >= 2:  # 2+ hoops is a successful break
                successful_breaks.append(exps)
            elif len(exps) >= 3 and hoops == 0:  # Long turn with no hoops
                failed_breaks.append(exps)

        # Learn from successful breaks
        if len(successful_breaks) >= 3:
            # What did successful breaks have in common?
            avg_roquets = sum(
                sum(1 for e in b if e.roqueted_ball) for b in successful_breaks
            ) / len(successful_breaks)

            avg_pioneers = sum(
                sum(1 for e in b if e.pioneer_placed) for b in successful_breaks
            ) / len(successful_breaks)

            # If pioneers correlate with success, boost pioneer-related positioning
            if avg_pioneers > 0.5:
                self.position_evaluator.weights['distance_to_partner'] -= self.learning_rate * 0.3

            # If maintaining roquets is key, boost roquet weights
            if avg_roquets >= 2:
                self.position_evaluator.weights['can_roquet_count'] += self.learning_rate * 0.5

    def _learn_optimal_approaches(self):
        """Learn optimal approach distances and angles for each hoop."""
        for hoop_num, patterns in self.approach_patterns.items():
            if len(patterns) < 10:
                continue

            # Separate successful and failed attempts
            successes = [p for p in patterns if p['success']]
            failures = [p for p in patterns if not p['success']]

            if len(successes) < 3:
                continue

            # Find optimal distance range for this hoop
            success_distances = [p['distance'] for p in successes]
            avg_success_dist = sum(success_distances) / len(success_distances)

            # Find optimal angle range
            success_angles = [p['angle'] for p in successes]
            avg_success_angle = sum(success_angles) / len(success_angles)

            # Store learned optimal approach for this hoop
            if not hasattr(self, 'optimal_approaches'):
                self.optimal_approaches = {}

            self.optimal_approaches[hoop_num] = {
                'optimal_distance': avg_success_dist,
                'optimal_angle': avg_success_angle,
                'success_rate': len(successes) / len(patterns),
                'sample_size': len(patterns),
            }

            # Learn per-hoop modifiers based on what worked
            self._learn_hoop_modifiers(hoop_num, successes, failures)

    def _learn_hoop_modifiers(self, hoop_num: int, successes: list, failures: list):
        """
        Learn per-hoop weight modifiers based on success patterns.

        Per-hoop modifiers adjust base weights for specific hoops:
        effective_weight = base_weight * (1 + modifier)

        Args:
            hoop_num: The hoop number (1-12)
            successes: List of successful approach patterns for this hoop
            failures: List of failed approach patterns for this hoop
        """
        if len(successes) < 5 or len(failures) < 5:
            return  # Need enough data

        # Initialize hoop modifiers if needed
        if hoop_num not in self.hoop_modifiers:
            self.hoop_modifiers[hoop_num] = {}

        # Calculate success rate for this hoop
        total = len(successes) + len(failures)
        success_rate = len(successes) / total

        # Compare to average success rate across all hoops
        all_rates = []
        for h, patterns in self.approach_patterns.items():
            if len(patterns) >= 10:
                s = sum(1 for p in patterns if p['success'])
                all_rates.append(s / len(patterns))

        if not all_rates:
            return

        avg_rate = sum(all_rates) / len(all_rates)

        # If this hoop is harder than average, adjust approach-related weights
        if success_rate < avg_rate * 0.8:  # 20% harder than average
            # This hoop is difficult - increase importance of good positioning
            self.hoop_modifiers[hoop_num]['approach_angle_quality'] = \
                self.hoop_modifiers[hoop_num].get('approach_angle_quality', 0) + self.learning_rate * 0.5
            self.hoop_modifiers[hoop_num]['distance_to_next_hoop'] = \
                self.hoop_modifiers[hoop_num].get('distance_to_next_hoop', 0) - self.learning_rate * 0.3

        elif success_rate > avg_rate * 1.2:  # 20% easier than average
            # This hoop is easier - can be more aggressive
            self.hoop_modifiers[hoop_num]['threatens_hoop'] = \
                self.hoop_modifiers[hoop_num].get('threatens_hoop', 0) + self.learning_rate * 0.3

        # Analyze successful vs failed approach distances
        avg_success_dist = sum(p['distance'] for p in successes) / len(successes)
        avg_failure_dist = sum(p['distance'] for p in failures) / len(failures)

        if avg_success_dist < avg_failure_dist * 0.7:
            # Successes come from closer - weight distance more for this hoop
            self.hoop_modifiers[hoop_num]['distance_to_next_hoop'] = \
                self.hoop_modifiers[hoop_num].get('distance_to_next_hoop', 0) - self.learning_rate * 0.2

        # Clamp modifiers to reasonable range (-0.5 to +0.5)
        for key in self.hoop_modifiers[hoop_num]:
            self.hoop_modifiers[hoop_num][key] = max(-0.5, min(0.5, self.hoop_modifiers[hoop_num][key]))

    def get_optimal_approach(self, hoop_num: int) -> Optional[Dict]:
        """Get the learned optimal approach for a hoop."""
        if hasattr(self, 'optimal_approaches') and hoop_num in self.optimal_approaches:
            return self.optimal_approaches[hoop_num]
        return None

    def _evaluate_outcomes(
        self,
        outcomes: List,
        striker: Ball,
        court: Court
    ) -> float:
        """Evaluate simulated outcomes."""
        total_value = 0.0

        for outcome in outcomes:
            value = 0.0

            # Immediate rewards
            if outcome.hoop_run:
                value += 20.0
            if outcome.roqueted_ball:
                value += 10.0

            # Position quality
            target_hoop = court.get_hoop_for_ball(striker.hoops_run)
            if target_hoop:
                dist = (outcome.striker_position - target_hoop.position).magnitude()
                value -= dist * 0.3

            total_value += value

        return total_value / len(outcomes) if outcomes else 0

    def _advice_to_shot(
        self,
        striker: Ball,
        advice: TacticalAdvice,
        court: Court
    ) -> Tuple[float, float]:
        """Convert tactical advice to shot parameters."""
        import math

        if advice.target_position:
            to_target = advice.target_position - striker.position
            angle = math.atan2(to_target.y, to_target.x)
            distance = to_target.magnitude()
            power = self.shot_simulator._power_for_distance(distance + 2)
        elif advice.target_ball:
            # This shouldn't happen as we should have target_position
            angle = 0
            power = 5.0
        else:
            # Default
            angle = 0
            power = 5.0

        return (angle, power)

    def _random_shot(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        court: Court
    ) -> Tuple[float, float, str]:
        """Generate a random exploratory shot."""
        import math

        # Random angle
        angle = random.uniform(0, 2 * math.pi)

        # Random power (biased toward medium shots)
        power = random.gauss(6.0, 2.0)
        power = max(2.0, min(10.0, power))

        return (angle, power, "Exploration shot")

    def _save_state(self):
        """Save learning state to disk."""
        state = {
            "games_played": self.games_played,
            "total_hoops_run": self.total_hoops_run,
            "total_roquets": self.total_roquets,
            "position_weights": self.position_evaluator.weights,
            "power_adjustment": self.power_adjustment,
            "hoop_success_rates": self.hoop_success_rates,
            "hoop_attempts": self.hoop_attempts,
            "game_outcomes": self.game_outcomes[-100:],  # Keep last 100
            # Enhanced learning data
            "break_stats": self.break_stats,
            "approach_patterns": self.approach_patterns,
            "croquet_shot_stats": self.croquet_shot_stats,
            "optimal_approaches": getattr(self, 'optimal_approaches', {}),
            # Checkpointing data
            "best_metrics": self.best_metrics,
            "momentum": self.momentum,
            "weight_velocities": self.weight_velocities,
            "hoop_modifiers": self.hoop_modifiers,
        }

        state_file = self.save_dir / "learning_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        # Save recent experiences
        if self.experiences:
            exp_data = [asdict(e) for e in self.experiences[-1000:]]
            exp_file = self.save_dir / "experiences.json"
            with open(exp_file, 'w') as f:
                json.dump(exp_data, f)

    def _check_milestones_and_checkpoint(self):
        """
        Check for milestone achievements and save checkpoints.

        Saves checkpoints:
        1. Every checkpoint_interval games
        2. When new best metrics are achieved
        """
        import datetime

        # Regular interval checkpoint
        if self.games_played % self.checkpoint_interval == 0:
            self._save_checkpoint(f"game_{self.games_played}")
            print(f"  [Checkpoint saved at game {self.games_played}]")

        # Check for milestone achievements
        milestone_reason = None

        # Best break length
        current_best_break = self.break_stats.get('best_break', 0)
        if current_best_break > self.best_metrics['best_break']:
            self.best_metrics['best_break'] = current_best_break
            milestone_reason = f"best_break_{current_best_break}"
            print(f"  [NEW BEST BREAK: {current_best_break} hoops!]")

        # Best average break length
        current_avg_break = self.break_stats.get('avg_break_length', 0.0)
        if current_avg_break > self.best_metrics['best_avg_break'] + 0.1:  # Significant improvement
            self.best_metrics['best_avg_break'] = current_avg_break
            if milestone_reason is None:  # Don't overwrite more important milestone
                milestone_reason = f"avg_break_{current_avg_break:.2f}"
            print(f"  [NEW BEST AVG BREAK: {current_avg_break:.2f} hoops!]")

        # Win rate over last 20 games
        if len(self.game_outcomes) >= 20:
            recent_games = self.game_outcomes[-20:]
            # Count wins (we need to know which side we're tracking - use both)
            bb_wins = sum(1 for g in recent_games if g.get('winner') == 'blue_black')
            ry_wins = sum(1 for g in recent_games if g.get('winner') == 'red_yellow')
            max_win_rate = max(bb_wins, ry_wins) / 20.0

            if max_win_rate > self.best_metrics['best_win_rate'] + 0.05:  # 5% improvement
                self.best_metrics['best_win_rate'] = max_win_rate
                if milestone_reason is None:
                    milestone_reason = f"win_rate_{max_win_rate:.0%}"
                print(f"  [NEW BEST WIN RATE: {max_win_rate:.0%}!]")

        # Average hoop success rate
        if self.hoop_success_rates:
            avg_success = sum(self.hoop_success_rates.values()) / len(self.hoop_success_rates)
            if avg_success > self.best_metrics['best_hoop_success'] + 0.02:  # 2% improvement
                self.best_metrics['best_hoop_success'] = avg_success
                if milestone_reason is None:
                    milestone_reason = f"hoop_success_{avg_success:.0%}"
                print(f"  [NEW BEST HOOP SUCCESS: {avg_success:.0%}!]")

        # Save milestone checkpoint
        if milestone_reason:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_checkpoint(f"milestone_{milestone_reason}_{timestamp}")

    def _save_checkpoint(self, checkpoint_name: str):
        """
        Save a checkpoint with all learning state.

        Args:
            checkpoint_name: Name for this checkpoint
        """
        checkpoint_data = {
            "checkpoint_name": checkpoint_name,
            "games_played": self.games_played,
            "total_hoops_run": self.total_hoops_run,
            "total_roquets": self.total_roquets,
            "position_weights": dict(self.position_evaluator.weights),
            "power_adjustment": self.power_adjustment,
            "hoop_success_rates": self.hoop_success_rates,
            "hoop_attempts": self.hoop_attempts,
            "break_stats": self.break_stats,
            "approach_patterns": self.approach_patterns,
            "croquet_shot_stats": self.croquet_shot_stats,
            "optimal_approaches": getattr(self, 'optimal_approaches', {}),
            "best_metrics": self.best_metrics,
            "momentum": self.momentum,
            "weight_velocities": self.weight_velocities,
            "hoop_modifiers": self.hoop_modifiers,
        }

        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Load a specific checkpoint.

        Args:
            checkpoint_name: Name of checkpoint to load

        Returns:
            True if loaded successfully, False otherwise
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"
        if not checkpoint_file.exists():
            print(f"Checkpoint not found: {checkpoint_name}")
            return False

        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)

            self.games_played = data.get("games_played", 0)
            self.total_hoops_run = data.get("total_hoops_run", 0)
            self.total_roquets = data.get("total_roquets", 0)
            self.power_adjustment = data.get("power_adjustment", 1.0)

            if "position_weights" in data:
                self.position_evaluator.weights.update(data["position_weights"])
            if "hoop_success_rates" in data:
                self.hoop_success_rates = {int(k): v for k, v in data["hoop_success_rates"].items()}
            if "hoop_attempts" in data:
                self.hoop_attempts = {int(k): v for k, v in data["hoop_attempts"].items()}
            if "break_stats" in data:
                self.break_stats = data["break_stats"]
            if "approach_patterns" in data:
                self.approach_patterns = {int(k): v for k, v in data["approach_patterns"].items()}
            if "croquet_shot_stats" in data:
                self.croquet_shot_stats = data["croquet_shot_stats"]
            if "optimal_approaches" in data:
                self.optimal_approaches = {int(k): v for k, v in data["optimal_approaches"].items()}
            if "best_metrics" in data:
                self.best_metrics = data["best_metrics"]
            if "weight_velocities" in data:
                self.weight_velocities = data["weight_velocities"]
            if "hoop_modifiers" in data:
                self.hoop_modifiers = {int(k): v for k, v in data["hoop_modifiers"].items()}

            print(f"Loaded checkpoint: {checkpoint_name}")
            print(f"  Games: {self.games_played}, Hoops: {self.total_hoops_run}")
            return True

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        checkpoints = []
        for f in self.checkpoint_dir.glob("*.json"):
            checkpoints.append(f.stem)
        return sorted(checkpoints)

    def _load_state(self):
        """Load learning state from disk."""
        state_file = self.save_dir / "learning_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                self.games_played = state.get("games_played", 0)
                self.total_hoops_run = state.get("total_hoops_run", 0)
                self.total_roquets = state.get("total_roquets", 0)
                self.game_outcomes = state.get("game_outcomes", [])
                self.power_adjustment = state.get("power_adjustment", 1.0)

                # Load hoop stats (convert string keys back to int)
                if "hoop_success_rates" in state:
                    self.hoop_success_rates = {int(k): v for k, v in state["hoop_success_rates"].items()}
                if "hoop_attempts" in state:
                    self.hoop_attempts = {int(k): v for k, v in state["hoop_attempts"].items()}

                if "position_weights" in state:
                    self.position_evaluator.weights.update(state["position_weights"])

                # Load enhanced learning data
                if "break_stats" in state:
                    self.break_stats = state["break_stats"]
                if "approach_patterns" in state:
                    # Convert string keys back to int
                    self.approach_patterns = {int(k): v for k, v in state["approach_patterns"].items()}
                if "croquet_shot_stats" in state:
                    self.croquet_shot_stats = state["croquet_shot_stats"]
                if "optimal_approaches" in state:
                    self.optimal_approaches = {int(k): v for k, v in state["optimal_approaches"].items()}

                # Load checkpointing data
                if "best_metrics" in state:
                    self.best_metrics = state["best_metrics"]
                if "momentum" in state:
                    self.momentum = state["momentum"]
                if "weight_velocities" in state:
                    self.weight_velocities = state["weight_velocities"]
                if "hoop_modifiers" in state:
                    self.hoop_modifiers = {int(k): v for k, v in state["hoop_modifiers"].items()}

                print(f"Loaded learning state: {self.games_played} games, {self.total_hoops_run} hoops run")
                print(f"  Power adjustment: {self.power_adjustment:.2f}")
                if hasattr(self, 'break_stats') and self.break_stats.get('avg_break_length', 0) > 0:
                    print(f"  Avg break length: {self.break_stats['avg_break_length']:.1f} hoops")
                if self.best_metrics.get('best_break', 0) > 0:
                    print(f"  Best break: {self.best_metrics['best_break']} hoops")
            except Exception as e:
                print(f"Could not load learning state: {e}")

    def get_stats(self) -> Dict:
        """Get learning statistics."""
        # Calculate leave pattern effectiveness
        leave_pattern_effectiveness = {}
        for pattern_type, stats in self.leave_pattern_matches.items():
            matches = stats['matches']
            scored = stats['opponent_scored']
            if matches > 0:
                # Lower is better (opponent scored less often)
                effectiveness = 1.0 - (scored / matches)
                leave_pattern_effectiveness[pattern_type] = {
                    'matches': matches,
                    'opponent_scored': scored,
                    'effectiveness': effectiveness
                }

        return {
            "games_played": self.games_played,
            "total_hoops_run": self.total_hoops_run,
            "total_roquets": self.total_roquets,
            "avg_hoops_per_game": self.total_hoops_run / max(1, self.games_played),
            "avg_roquets_per_game": self.total_roquets / max(1, self.games_played),
            "experiences_recorded": len(self.experiences),
            "exploration_rate": self.exploration_rate,
            "power_adjustment": self.power_adjustment,
            "position_weights": dict(self.position_evaluator.weights),
            "hoop_success_rates": self.hoop_success_rates,
            # Enhanced stats
            "break_stats": self.break_stats,
            "croquet_shot_stats": self.croquet_shot_stats,
            "optimal_approaches": getattr(self, 'optimal_approaches', {}),
            # Leave pattern stats
            "leave_pattern_effectiveness": leave_pattern_effectiveness,
        }

    def get_leave_suggestion(
        self,
        striker_color: str,
        striker_next_hoop: int,
        current_positions: Dict[str, Ball]
    ) -> Dict[str, Vector2]:
        """
        Get suggested leave positions based on standard patterns.

        Args:
            striker_color: Ball color making the leave
            striker_next_hoop: Striker's next hoop (1-12)
            current_positions: Current ball positions

        Returns:
            Dict of suggested positions for each ball
        """
        actual_positions = {c: b.position for c, b in current_positions.items()}
        return self.leave_patterns.suggest_leave_positions(
            striker_color, striker_next_hoop, actual_positions
        )

    def learn_from_leave_outcome(self, leave_worked: bool, pattern_used: str):
        """
        Learn from a leave outcome - did it prevent opponent from scoring?

        This is the core of adaptive leave learning. Over many games,
        we learn which leave patterns are most effective.

        Args:
            leave_worked: True if opponent didn't score after our leave
            pattern_used: Which pattern was used (NSL, OSL, DIAGONAL, CUSTOM)
        """
        if pattern_used not in self.leave_pattern_matches:
            self.leave_pattern_matches[pattern_used] = {'matches': 0, 'opponent_scored': 0}

        self.leave_pattern_matches[pattern_used]['matches'] += 1
        if not leave_worked:
            self.leave_pattern_matches[pattern_used]['opponent_scored'] += 1

        # Adjust pattern preference weights based on outcomes
        if leave_worked:
            # Pattern worked - boost it slightly
            if hasattr(self.leave_patterns, 'pattern_weights'):
                current_weight = self.leave_patterns.pattern_weights.get(pattern_used, 1.0)
                self.leave_patterns.pattern_weights[pattern_used] = min(2.0, current_weight + 0.05)
        else:
            # Pattern failed - reduce its weight
            if hasattr(self.leave_patterns, 'pattern_weights'):
                current_weight = self.leave_patterns.pattern_weights.get(pattern_used, 1.0)
                self.leave_patterns.pattern_weights[pattern_used] = max(0.5, current_weight - 0.03)

    def get_best_leave_pattern(self) -> Optional[str]:
        """
        Get the currently best-performing leave pattern based on learned data.

        Returns:
            Name of best pattern, or None if not enough data
        """
        best_pattern = None
        best_effectiveness = 0.0

        for pattern_type, stats in self.leave_pattern_matches.items():
            matches = stats['matches']
            if matches < 10:
                continue  # Not enough data

            effectiveness = 1.0 - (stats['opponent_scored'] / matches)
            if effectiveness > best_effectiveness:
                best_effectiveness = effectiveness
                best_pattern = pattern_type

        return best_pattern

    def record_peel_attempt(self, peel_successful: bool, hoop_num: int, peel_type: str):
        """
        Record a peel attempt for learning.

        Args:
            peel_successful: Whether the peel went through
            hoop_num: Which hoop was peeled (10=4-back, 11=penult, 12=rover)
            peel_type: Type of peel (straight, irish, rush, etc.)
        """
        if not hasattr(self, 'peel_stats'):
            self.peel_stats = {
                'attempts': 0,
                'successes': 0,
                'by_hoop': {10: {'attempts': 0, 'successes': 0},
                            11: {'attempts': 0, 'successes': 0},
                            12: {'attempts': 0, 'successes': 0}},
                'by_type': {}
            }

        self.peel_stats['attempts'] += 1
        if peel_successful:
            self.peel_stats['successes'] += 1

        # Track by hoop
        if hoop_num in self.peel_stats['by_hoop']:
            self.peel_stats['by_hoop'][hoop_num]['attempts'] += 1
            if peel_successful:
                self.peel_stats['by_hoop'][hoop_num]['successes'] += 1

        # Track by type
        if peel_type not in self.peel_stats['by_type']:
            self.peel_stats['by_type'][peel_type] = {'attempts': 0, 'successes': 0}
        self.peel_stats['by_type'][peel_type]['attempts'] += 1
        if peel_successful:
            self.peel_stats['by_type'][peel_type]['successes'] += 1

    def get_peel_success_rate(self, hoop_num: int = None, peel_type: str = None) -> float:
        """
        Get the learned peel success rate.

        Args:
            hoop_num: Optional - specific hoop (10, 11, 12)
            peel_type: Optional - specific peel type

        Returns:
            Success rate (0-1) or 0.5 if no data
        """
        if not hasattr(self, 'peel_stats') or self.peel_stats['attempts'] < 5:
            return 0.5  # Default rate if no data

        if hoop_num is not None and hoop_num in self.peel_stats['by_hoop']:
            data = self.peel_stats['by_hoop'][hoop_num]
            if data['attempts'] >= 3:
                return data['successes'] / data['attempts']

        if peel_type is not None and peel_type in self.peel_stats['by_type']:
            data = self.peel_stats['by_type'][peel_type]
            if data['attempts'] >= 3:
                return data['successes'] / data['attempts']

        # Overall rate
        return self.peel_stats['successes'] / self.peel_stats['attempts']

    def get_adaptive_strategy_weights(self) -> Dict[str, float]:
        """
        Get current adaptive strategy weights based on learning.

        These weights modify the base tactical rules based on what
        has been working in games.

        Returns:
            Dict of strategy aspect -> weight modifier
        """
        weights = {
            'aggression': 1.0,  # How aggressively to attack
            'defense': 1.0,    # How much to prioritize defense
            'break_building': 1.0,  # Focus on 4-ball breaks
            'peeling': 1.0,    # Willingness to attempt peels
        }

        # Adjust based on learned data
        if self.break_stats.get('avg_break_length', 0) > 3:
            # We're good at breaks - emphasize them
            weights['break_building'] = 1.2
        elif self.break_stats.get('avg_break_length', 0) < 1:
            # Struggling with breaks - be more cautious
            weights['break_building'] = 0.8
            weights['defense'] = 1.2

        # Adjust peel willingness based on success rate
        if hasattr(self, 'peel_stats') and self.peel_stats.get('attempts', 0) >= 10:
            peel_rate = self.peel_stats['successes'] / self.peel_stats['attempts']
            if peel_rate > 0.6:
                weights['peeling'] = 1.3
            elif peel_rate < 0.3:
                weights['peeling'] = 0.7

        # Adjust aggression based on leave effectiveness
        best_leave = self.get_best_leave_pattern()
        if best_leave and best_leave in ['NSL', 'OSL']:
            # Defensive leaves working well - can be aggressive
            weights['aggression'] = 1.1
        elif best_leave == 'CUSTOM':
            # Custom leaves not reliable - be more careful
            weights['defense'] = 1.1

        return weights
