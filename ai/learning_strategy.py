"""
Learning Strategy - AI strategy that uses the hybrid learning system.

Combines tactical rules, position evaluation, and shot simulation
to make informed decisions that improve over time.
"""
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple, Optional

import config
from models.ball import Ball, Vector2
from models.court import Court, Hoop
from ai.learning.position_evaluator import PositionEvaluator
from ai.learning.shot_simulator import ShotSimulator
from ai.learning.tactical_rules import TacticalRules, ShotType


class LearningStrategy:
    """
    AI strategy that learns from experience.

    Uses:
    - Tactical rules for high-level decisions
    - Position evaluation for assessing board states
    - Shot simulation for evaluating shot options
    - Learned power adjustment from experience
    """

    def __init__(self, skill_level: float = 0.8):
        """
        Initialize learning strategy.

        Args:
            skill_level: Base skill level (affects shot accuracy)
        """
        self.skill_level = skill_level
        self.name = "Learning"

        # Learning components
        self.position_evaluator = PositionEvaluator()
        self.shot_simulator = ShotSimulator(num_simulations=20)
        self.tactical_rules = TacticalRules()

        # Track decisions for learning
        self.last_advice = None
        self.last_shot_type = None

        # Load learned parameters from saved state
        self.power_adjustment = self._load_power_adjustment()

    def _load_power_adjustment(self) -> float:
        """Load the learned power adjustment from saved learning state."""
        state_file = Path("ai_data/learning_state.json")
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    return state.get("power_adjustment", 1.0)
            except Exception:
                pass
        return 1.0

    def select_shot(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        court: Court,
        target_hoop_num: int = None,
        deadness: Dict[str, set] = None,
        strokes_remaining: int = 1,
        is_continuation: bool = False
    ) -> Tuple[float, float]:
        """
        Select a shot using tactical analysis and simulation.

        Args:
            ball: The ball to shoot
            balls: All balls on the court
            court: The court
            target_hoop_num: Ignored (we use ball's next hoop)
            deadness: Which balls we're dead on
            strokes_remaining: Number of strokes remaining this turn
            is_continuation: Whether this is a continuation stroke

        Returns:
            Tuple of (angle in radians, power in yards/second)
        """
        if deadness is None:
            deadness = {c: set() for c in ["blue", "black", "red", "yellow"]}

        # Store deadness for use in outcome evaluation
        self._current_deadness = deadness
        self._current_striker_color = ball.color

        # Get tactical advice, passing continuation context
        advice_list = self.tactical_rules.get_advice(
            ball, balls, court, deadness,
            strokes_remaining=strokes_remaining,
            is_continuation=is_continuation
        )

        if not advice_list:
            # Fallback to basic shot
            return self._fallback_shot(ball, court)

        # Evaluate top advice options with simulation
        # Use composite scores for tie-breaking
        best_angle = 0
        best_power = 5.0
        best_composite = None  # (primary_value, hoop_score, roquet_score, position_score)

        for advice in advice_list[:3]:
            angle, power = self._advice_to_shot(ball, advice, court)

            # Simulate the shot
            outcomes = self.shot_simulator.simulate_shot(
                ball, angle, power, self.skill_level, balls, court
            )

            # Calculate composite scores for tie-breaking
            composite = self._evaluate_outcomes_composite(outcomes, ball, court)

            # Weight by tactical priority
            primary_value = composite[0] + advice.priority * 10
            composite = (primary_value,) + composite[1:]

            # Compare using tuple ordering for automatic tie-breaking
            # Ties on primary value break by hoop score, then roquet score, then position
            if best_composite is None or composite > best_composite:
                best_composite = composite
                best_angle = angle
                best_power = power
                self.last_advice = advice
                self.last_shot_type = advice.recommended_shot

        # Apply learned power adjustment (calibrated from experience)
        adjusted_power = best_power * self.power_adjustment

        # Add skill-based inaccuracy
        angle_error = random.gauss(0, (1 - self.skill_level) * 0.2)
        power_error = random.gauss(0, (1 - self.skill_level) * 0.1)

        final_angle = best_angle + angle_error
        final_power = adjusted_power * (1 + power_error)
        final_power = max(1.5, min(config.MAX_SHOT_POWER, final_power))

        return (final_angle, final_power)

    def select_shot_for_hoop(
        self,
        ball: Ball,
        target_hoop: Hoop,
        balls: Dict[str, Ball],
        court: Court
    ) -> Tuple[float, float]:
        """Select a shot targeting a specific hoop."""
        # Check approach quality
        if self._is_good_approach(ball, target_hoop):
            return self._shoot_for_hoop(ball, target_hoop)
        else:
            return self._shoot_for_position(ball, target_hoop, court)

    def _advice_to_shot(
        self,
        ball: Ball,
        advice,
        court: Court
    ) -> Tuple[float, float]:
        """Convert tactical advice to shot parameters."""
        if advice.target_position:
            to_target = advice.target_position - ball.position
            angle = math.atan2(to_target.y, to_target.x)
            distance = to_target.magnitude()

            # Adjust power based on shot type
            if advice.recommended_shot == ShotType.HOOP_RUN:
                # Need to go through the hoop
                power = self._calculate_power_for_distance(distance + 3)
            elif advice.recommended_shot == ShotType.HOOP_APPROACH:
                # Want to stop at position
                power = self._calculate_power_for_distance(distance)
            elif advice.recommended_shot in [ShotType.ROQUET, ShotType.RUSH]:
                # Hit the ball with some follow-through
                power = self._calculate_power_for_distance(distance + 1)
            else:
                power = self._calculate_power_for_distance(distance + 2)

            return (angle, power)

        return (0, 5.0)

    def _evaluate_outcomes_composite(
        self,
        outcomes,
        ball: Ball,
        court: Court
    ) -> Tuple[float, float, float, float, float]:
        """
        Evaluate simulated shot outcomes with composite scoring for tie-breaking.

        Returns a tuple of scores for tie-breaking:
        (primary_value, hoop_score, roquet_score, position_score, boundary_penalty)

        Using a tuple allows Python's natural comparison to break ties:
        - First compare primary value
        - If tied, prefer higher hoop score (running hoops)
        - If still tied, prefer higher roquet score (gaining strokes)
        - If still tied, prefer better position
        - If still tied, prefer avoiding boundaries
        """
        total_primary = 0.0
        total_hoop = 0.0
        total_roquet = 0.0
        total_position = 0.0
        total_boundary = 0.0

        # Get which balls we're dead on
        dead_on = set()
        if hasattr(self, '_current_deadness') and hasattr(self, '_current_striker_color'):
            dead_on = self._current_deadness.get(self._current_striker_color, set())

        for outcome in outcomes:
            primary = 0.0
            hoop = 0.0
            roquet = 0.0
            position = 0.0
            boundary = 0.0

            # Immediate game rewards - hoops are the goal!
            if outcome.hoop_run:
                primary += 50.0  # Running hoops is VERY valuable - this is the objective!
                hoop += 100.0   # Maximum hoop score for tie-breaking

            # Roquets only count if the ball is LIVE (not dead on)
            if outcome.roqueted_ball:
                if outcome.roqueted_ball not in dead_on:
                    primary += 15.0  # Roquets earn extra strokes - but only if live!
                    roquet += 50.0   # Roquet score for tie-breaking
                else:
                    # PENALTY for hitting a dead ball - this wastes the shot!
                    primary -= 20.0  # Strong penalty - hitting dead ball ends turn with no benefit
                    roquet -= 100.0  # Heavily penalize in tie-breaking too

            # Position quality after the shot
            target_hoop = court.get_hoop_for_ball(ball.hoops_run)
            if target_hoop:
                dist = (outcome.striker_position - target_hoop.position).magnitude()

                # Prefer being in front of hoop with good approach angle
                to_hoop = target_hoop.position - outcome.striker_position
                if to_hoop.magnitude() > 0.5:
                    approach_dot = to_hoop.normalize().dot(target_hoop.direction)
                    if approach_dot > 0.5:
                        # Good approach - closer is better
                        pos_value = max(0, (8 - dist)) * 1.5  # Up to 12 points for being close
                        primary += pos_value
                        # Position score: weighted by approach angle quality
                        position += (10 - dist) * approach_dot * 5  # Refined position score
                    elif approach_dot > 0:
                        # OK approach
                        pos_value = max(0, (6 - dist)) * 0.5
                        primary += pos_value
                        position += (8 - dist) * approach_dot * 2
                    else:
                        # Wrong side of hoop - penalize
                        primary -= 5.0
                        position -= dist * 2  # Distance penalty when on wrong side

            # Avoid boundaries
            if outcome.hit_boundary:
                primary -= 2.0
                boundary -= 10.0  # Boundary penalty for tie-breaking

            total_primary += primary
            total_hoop += hoop
            total_roquet += roquet
            total_position += position
            total_boundary += boundary

        n = len(outcomes) if outcomes else 1
        return (
            total_primary / n,
            total_hoop / n,
            total_roquet / n,
            total_position / n,
            total_boundary / n
        )

    def _evaluate_outcomes(
        self,
        outcomes,
        ball: Ball,
        court: Court
    ) -> float:
        """Evaluate simulated shot outcomes (legacy single-value version)."""
        composite = self._evaluate_outcomes_composite(outcomes, ball, court)
        return composite[0]  # Return just the primary value

    def _fallback_shot(self, ball: Ball, court: Court) -> Tuple[float, float]:
        """Fallback shot when no good options."""
        target_hoop = court.get_hoop_for_ball(ball.hoops_run)

        if target_hoop:
            to_hoop = target_hoop.position - ball.position
            angle = math.atan2(to_hoop.y, to_hoop.x)
            distance = to_hoop.magnitude()
            power = self._calculate_power_for_distance(distance + 2)
        elif ball.is_rover:
            # Rover: aim for the peg to finish the game!
            to_peg = court.peg_position - ball.position
            angle = math.atan2(to_peg.y, to_peg.x)
            distance = to_peg.magnitude()
            # Need to hit the peg with enough power to reach it
            power = self._calculate_power_for_distance(distance + 1)
        else:
            # Aim for center
            center = Vector2(court.width / 2, court.height / 2)
            to_center = center - ball.position
            angle = math.atan2(to_center.y, to_center.x)
            power = 5.0

        return (angle, power)

    def _is_good_approach(self, ball: Ball, hoop: Hoop) -> bool:
        """Check if ball has good approach to hoop."""
        to_hoop = hoop.position - ball.position
        distance = to_hoop.magnitude()

        if distance < 0.5 or distance > 6:
            return False

        approach_dir = to_hoop.normalize()
        dot = approach_dir.dot(hoop.direction)
        return dot > 0.5

    def _shoot_for_hoop(self, ball: Ball, target_hoop: Hoop) -> Tuple[float, float]:
        """Shoot to run the hoop."""
        # Aim past the hoop
        aim_point = target_hoop.position + target_hoop.direction * 2

        delta = aim_point - ball.position
        angle = math.atan2(delta.y, delta.x)
        distance = delta.magnitude()
        power = self._calculate_power_for_distance(distance + 3)

        return (angle, power)

    def _shoot_for_position(
        self,
        ball: Ball,
        target_hoop: Hoop,
        court: Court
    ) -> Tuple[float, float]:
        """Shoot to set up for hoop."""
        # Position 4 yards in front of hoop
        setup_pos = target_hoop.position - target_hoop.direction * 4

        # Add slight randomness
        offset = Vector2(
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        )
        setup_pos = setup_pos + offset

        # Clamp to court
        setup_pos.x = max(1, min(court.width - 1, setup_pos.x))
        setup_pos.y = max(1, min(court.height - 1, setup_pos.y))

        delta = setup_pos - ball.position
        angle = math.atan2(delta.y, delta.x)
        distance = delta.magnitude()
        power = self._calculate_power_for_distance(distance)

        return (angle, power)

    def _calculate_power_for_distance(self, distance: float) -> float:
        """Calculate power needed for distance."""
        friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY
        velocity = math.sqrt(2 * friction_decel * distance)
        return min(velocity * 1.1, config.MAX_SHOT_POWER)

    def get_description(self) -> str:
        """Return description of this strategy."""
        return f"Learning (skill: {self.skill_level:.0%})"
