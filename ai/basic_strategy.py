"""
Basic AI Strategy for Golf Croquet.

Aims to run the current hoop that all balls are contesting.
Includes strategic positioning when a direct run isn't viable.
"""
import math
import random
from typing import Dict, Tuple, Optional

import config
from models.ball import Ball, Vector2
from models.court import Court, Hoop


class BasicStrategy:
    """
    AI strategy for Golf Croquet.

    For Golf Croquet:
    - Aims to go THROUGH the hoop (not just to it)
    - Considers the hoop direction
    - Sets up in front of hoop if approach angle is poor
    - Adds realistic inaccuracy based on skill level
    """

    def __init__(self, skill_level: float = 0.7):
        """
        Initialize basic strategy.

        Args:
            skill_level: 0.0 to 1.0, affects accuracy
        """
        self.skill_level = skill_level
        self.name = "Basic"

    def _is_good_approach(self, ball: Ball, hoop: Hoop) -> bool:
        """
        Check if ball has a good approach angle to run the hoop.

        A good approach means:
        1. Ball is on the correct side of the hoop (entry side)
        2. Approach angle is within ~60 degrees of hoop direction
        """
        ball_to_hoop = hoop.position - ball.position
        distance = ball_to_hoop.magnitude()

        if distance < 0.5:
            return False  # Too close to hoop

        # Check if on correct side (positive dot product means entry side)
        approach_dot = ball_to_hoop.dot(hoop.direction)
        if approach_dot < 0:
            return False  # Wrong side of hoop

        # Check angle alignment
        ball_to_hoop_normalized = ball_to_hoop.normalize()
        # We want to be moving roughly opposite to hoop direction
        # (if hoop faces north, we should approach from south)
        alignment = ball_to_hoop_normalized.dot(hoop.direction)
        return alignment > 0.5  # Within ~60 degrees

    def _find_setup_position(self, ball: Ball, hoop: Hoop, court: Court) -> Vector2:
        """
        Find a good position to set up for running the hoop.

        The ideal setup position is:
        - 3-5 yards in front of the hoop (on the entry side)
        - Slightly off to one side to avoid being blocked
        """
        # Position 4 yards in front of hoop (opposite to hoop direction)
        setup_distance = 4.0

        # Add some randomness to side positioning (-1.5 to 1.5 yards)
        side_offset = random.uniform(-1.5, 1.5)

        # Calculate perpendicular direction
        perp_dir = Vector2(-hoop.direction.y, hoop.direction.x)

        # Setup position is behind the hoop entry point
        setup_pos = (hoop.position
                     - hoop.direction * setup_distance
                     + perp_dir * side_offset)

        # Clamp to court boundaries
        margin = 1.0
        setup_pos.x = max(margin, min(court.width - margin, setup_pos.x))
        setup_pos.y = max(margin, min(court.height - margin, setup_pos.y))

        return setup_pos

    def select_shot_for_hoop(
        self,
        ball: Ball,
        target_hoop: Hoop,
        balls: Dict[str, Ball],
        court: Court
    ) -> Tuple[float, float]:
        """
        Select a shot to run a specific hoop.

        Strategy:
        1. If good approach angle: shoot to run the hoop
        2. If poor approach: set up in front of the hoop

        Args:
            ball: The ball to shoot
            target_hoop: The hoop to run
            balls: All balls on the court
            court: The court

        Returns:
            Tuple of (angle in radians, power in yards/second)
        """
        hoop_pos = target_hoop.position
        hoop_dir = target_hoop.direction

        # Check if we have a good approach
        if self._is_good_approach(ball, target_hoop):
            # Go for the hoop run
            return self._shoot_for_hoop(ball, target_hoop)
        else:
            # Set up in front of the hoop
            return self._shoot_for_position(ball, target_hoop, court)

    def _shoot_for_hoop(self, ball: Ball, target_hoop: Hoop) -> Tuple[float, float]:
        """Shoot to run through the hoop."""
        hoop_pos = target_hoop.position
        hoop_dir = target_hoop.direction

        # Aim at a point 2 yards past the hoop in the running direction
        aim_point = hoop_pos + hoop_dir * 2.0

        # Calculate angle from ball to aim point
        delta = aim_point - ball.position
        base_angle = math.atan2(delta.y, delta.x)

        # Add inaccuracy based on skill level
        accuracy_error = (1.0 - self.skill_level) * 0.25  # Max ~14 degrees error
        angle_error = random.gauss(0, accuracy_error)
        angle = base_angle + angle_error

        # Calculate distance and power
        distance = delta.magnitude()

        # We need enough power to go THROUGH the hoop (add extra distance)
        target_distance = distance + 3.0  # Go 3 yards past the hoop

        base_power = self._calculate_power_for_distance(target_distance)

        # Add power variation
        power_error = (1.0 - self.skill_level) * 0.15
        power_multiplier = 1.0 + random.gauss(0, power_error)
        power = base_power * power_multiplier

        # Clamp power
        power = max(2.0, min(config.MAX_SHOT_POWER, power))

        return (angle, power)

    def _shoot_for_position(
        self,
        ball: Ball,
        target_hoop: Hoop,
        court: Court
    ) -> Tuple[float, float]:
        """Shoot to set up a good position in front of the hoop."""
        setup_pos = self._find_setup_position(ball, target_hoop, court)

        # Calculate angle to setup position
        delta = setup_pos - ball.position
        base_angle = math.atan2(delta.y, delta.x)

        # Add inaccuracy
        accuracy_error = (1.0 - self.skill_level) * 0.2
        angle_error = random.gauss(0, accuracy_error)
        angle = base_angle + angle_error

        # Calculate distance and power - we want to stop AT the position
        distance = delta.magnitude()
        base_power = self._calculate_power_for_distance(distance)

        # Add power variation (less than hoop shots - positioning is gentler)
        power_error = (1.0 - self.skill_level) * 0.1
        power_multiplier = 1.0 + random.gauss(0, power_error)
        power = base_power * power_multiplier

        # Clamp power (positioning shots are usually gentler)
        power = max(1.5, min(config.MAX_SHOT_POWER * 0.7, power))

        return (angle, power)

    def select_shot(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        court: Court,
        target_hoop_num: int = None
    ) -> Tuple[float, float]:
        """
        Select a shot for the given ball.

        Args:
            ball: The ball to shoot
            balls: All balls on the court
            court: The court
            target_hoop_num: Hoop number to aim for (for Golf Croquet)

        Returns:
            Tuple of (angle in radians, power in yards/second)
        """
        # Get target hoop
        if target_hoop_num is not None:
            target_hoop = court.get_hoop(target_hoop_num)
        else:
            # Fallback: use ball's next hoop (Association Croquet style)
            target_hoop = court.get_hoop_for_ball(ball.hoops_run)

        if target_hoop is None:
            # Aim for center of court
            target = Vector2(court.width / 2, court.height / 2)
            delta = target - ball.position
            angle = math.atan2(delta.y, delta.x)
            power = 5.0
            return (angle, power)

        return self.select_shot_for_hoop(ball, target_hoop, balls, court)

    def _calculate_power_for_distance(self, distance: float) -> float:
        """
        Calculate the power needed to travel a given distance.

        Uses physics: v² = 2 * a * d, where a is friction deceleration.
        """
        friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY

        # v = sqrt(2 * a * d)
        required_velocity = math.sqrt(2 * friction_decel * distance)

        # Add a bit extra to ensure we get there
        return required_velocity * 1.1

    def get_description(self) -> str:
        """Return a description of this strategy."""
        return f"Basic (skill: {self.skill_level:.0%})"
