"""
Croquet Stroke Types - Different ways to play a croquet stroke.

Each stroke type has a different ratio of striker ball to croqueted ball movement.
The physics vary based on where you hit your ball and follow-through.
"""
import math
from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple

from models.ball import Ball, Vector2
import config


class StrokeType(Enum):
    """Types of croquet strokes."""
    STOP_SHOT = auto()    # Striker barely moves, croqueted goes far (1:5 to 1:10)
    DRIVE = auto()        # Standard stroke (1:3)
    HALF_ROLL = auto()    # Both move, striker less (1:2)
    FULL_ROLL = auto()    # Both move equal distance (1:1)
    PASS_ROLL = auto()    # Striker goes further than croqueted (1.5:1)
    TAKE_OFF = auto()     # Striker goes far, croqueted barely moves
    SPLIT = auto()        # Balls go in different directions


@dataclass
class StrokeResult:
    """Result of a croquet stroke calculation."""
    striker_velocity: Vector2
    croqueted_velocity: Vector2
    description: str


class CroquetStrokeCalculator:
    """
    Calculates the physics of different croquet strokes.

    In a croquet stroke, the striker ball is placed in contact with
    the croqueted ball. When struck, both balls move according to
    the stroke type chosen.
    """

    # Ratio of striker distance to croqueted distance for each stroke type
    # Refined based on Aiton's teachings (Section 2.3, 2.7):
    # - Stop-shot ratio ~1:6 for standard approaches
    # - Tighter control needed for close work (12 inches)
    STROKE_RATIOS = {
        StrokeType.STOP_SHOT: 0.167,   # Striker moves ~17% (1:6 ratio per Aiton)
        StrokeType.DRIVE: 0.33,        # Striker moves 33% (1:3 ratio)
        StrokeType.HALF_ROLL: 0.5,     # Striker moves 50% (1:2 ratio)
        StrokeType.FULL_ROLL: 1.0,     # Equal distance (1:1 ratio)
        StrokeType.PASS_ROLL: 1.5,     # Striker moves 150% of croqueted
        StrokeType.TAKE_OFF: 5.0,      # Striker moves 5x croqueted distance
    }

    # Aiton's approach-specific ratios
    APPROACH_RATIOS = {
        'close': 0.125,    # 1:8 for 12-inch approaches - needs excellent control
        'standard': 0.167, # 1:6 for standard 1-yard approaches
        'long': 0.20,      # 1:5 for longer approaches (3+ yards)
    }

    def calculate_stroke(
        self,
        stroke_type: StrokeType,
        striker: Ball,
        croqueted: Ball,
        aim_direction: Vector2,
        power: float,
        split_angle: float = 0.0
    ) -> StrokeResult:
        """
        Calculate the resulting velocities for a croquet stroke.

        Args:
            stroke_type: Type of stroke to play
            striker: The striker's ball (in contact with croqueted)
            croqueted: The ball being croqueted
            aim_direction: Direction to aim (normalized vector)
            power: Power of the stroke
            split_angle: For split shots, angle between ball paths (radians)

        Returns:
            StrokeResult with velocities for both balls
        """
        if stroke_type == StrokeType.SPLIT:
            return self._calculate_split(
                striker, croqueted, aim_direction, power, split_angle
            )

        ratio = self.STROKE_RATIOS[stroke_type]

        # Croqueted ball moves in aim direction
        # The total energy is distributed based on the ratio
        total_energy = power * power

        if stroke_type == StrokeType.TAKE_OFF:
            # Take-off: striker gets most energy, croqueted barely moves
            croqueted_power = power * 0.15  # Just enough to be legal
            striker_power = power * 0.9
            # Striker can go at an angle to the croqueted ball
            croqueted_vel = aim_direction * croqueted_power
            striker_vel = aim_direction * striker_power
        else:
            # For other strokes, energy is shared based on ratio
            # If ratio is 0.33 (drive), croqueted gets 3x the speed
            # v_striker / v_croqueted = ratio
            # Total momentum: m*v_striker + m*v_croqueted = m*power (roughly)

            # Calculate velocities based on ratio
            # striker_speed = ratio * croqueted_speed
            # For stop shots and drives, croqueted ball needs to go far (10-20 yards)
            # BUT we need to be careful not to send balls out of bounds
            # Reduced multiplier from 1.8 to 1.3 to prevent balls going out
            croqueted_speed = power / (1 + ratio) * 1.3
            striker_speed = croqueted_speed * ratio

            croqueted_vel = aim_direction * croqueted_speed
            striker_vel = aim_direction * striker_speed

        # Generate description
        descriptions = {
            StrokeType.STOP_SHOT: "stop shot",
            StrokeType.DRIVE: "drive",
            StrokeType.HALF_ROLL: "half roll",
            StrokeType.FULL_ROLL: "full roll",
            StrokeType.PASS_ROLL: "pass roll",
            StrokeType.TAKE_OFF: "take-off",
        }

        return StrokeResult(
            striker_velocity=striker_vel,
            croqueted_velocity=croqueted_vel,
            description=descriptions.get(stroke_type, "croquet stroke")
        )

    def _calculate_split(
        self,
        striker: Ball,
        croqueted: Ball,
        aim_direction: Vector2,
        power: float,
        split_angle: float
    ) -> StrokeResult:
        """
        Calculate a split shot where balls go in different directions.

        In a split shot, you aim between where you want the two balls to go.
        The balls separate at the split_angle.
        """
        # Half the split angle for each ball from the aim direction
        half_angle = split_angle / 2

        # Rotate aim direction for each ball
        cos_a = math.cos(half_angle)
        sin_a = math.sin(half_angle)

        # Croqueted goes one way
        croqueted_dir = Vector2(
            aim_direction.x * cos_a - aim_direction.y * sin_a,
            aim_direction.x * sin_a + aim_direction.y * cos_a
        )

        # Striker goes the other way
        striker_dir = Vector2(
            aim_direction.x * cos_a + aim_direction.y * sin_a,
            -aim_direction.x * sin_a + aim_direction.y * cos_a
        )

        # Power is shared - wider splits mean less power to each ball
        # Reduced power to prevent balls going out of bounds during split shots
        split_factor = math.cos(half_angle)  # Reduces power as angle increases
        ball_power = power * split_factor * 0.75  # More reduction for safer splits

        return StrokeResult(
            striker_velocity=striker_dir * ball_power,
            croqueted_velocity=croqueted_dir * ball_power,
            description=f"split shot ({math.degrees(split_angle):.0f}°)"
        )

    def select_best_stroke(
        self,
        striker: Ball,
        croqueted: Ball,
        striker_target: Vector2,
        croqueted_target: Vector2,
        court
    ) -> Tuple[StrokeType, Vector2, float, float]:
        """
        Select the best stroke type to send both balls to their targets.

        Args:
            striker: Striker ball
            croqueted: Croqueted ball
            striker_target: Where striker should end up
            croqueted_target: Where croqueted ball should end up
            court: The court

        Returns:
            Tuple of (stroke_type, aim_direction, power, split_angle)
        """
        # Calculate required distances
        striker_dist = (striker_target - striker.position).magnitude()
        croqueted_dist = (croqueted_target - croqueted.position).magnitude()

        # Calculate directions
        to_striker_target = (striker_target - striker.position)
        to_croqueted_target = (croqueted_target - croqueted.position)

        if croqueted_dist < 0.5:
            croqueted_dist = 0.5

        # Calculate angle between targets
        if to_striker_target.magnitude() > 0.1 and to_croqueted_target.magnitude() > 0.1:
            striker_dir = to_striker_target.normalize()
            croqueted_dir = to_croqueted_target.normalize()
            dot = striker_dir.dot(croqueted_dir)
            angle_between = math.acos(max(-1, min(1, dot)))
        else:
            angle_between = 0

        # If balls need to go in very different directions, use split shot
        if angle_between > math.radians(30):
            # Split shot
            aim_dir = (to_striker_target.normalize() + to_croqueted_target.normalize())
            if aim_dir.magnitude() > 0.1:
                aim_dir = aim_dir.normalize()
            else:
                aim_dir = to_croqueted_target.normalize()

            power = self._power_for_distance(max(striker_dist, croqueted_dist))
            return (StrokeType.SPLIT, aim_dir, power, angle_between)

        # Calculate desired ratio
        if croqueted_dist > 0.1:
            desired_ratio = striker_dist / croqueted_dist
        else:
            desired_ratio = 5.0  # Take-off

        # Find best matching stroke type
        best_stroke = StrokeType.DRIVE
        best_diff = float('inf')

        for stroke_type, ratio in self.STROKE_RATIOS.items():
            diff = abs(ratio - desired_ratio)
            if diff < best_diff:
                best_diff = diff
                best_stroke = stroke_type

        # Aim direction is toward croqueted target
        aim_dir = to_croqueted_target.normalize() if to_croqueted_target.magnitude() > 0.1 else Vector2(1, 0)

        # Power based on croqueted ball distance (it determines the shot)
        power = self._power_for_distance(croqueted_dist)

        return (best_stroke, aim_dir, power, 0.0)

    def _power_for_distance(self, distance: float) -> float:
        """Calculate power needed for a ball to travel a distance."""
        friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY
        velocity = math.sqrt(2 * friction_decel * distance)
        # Reduced from 1.5x to 1.2x max power to prevent balls going out of bounds
        # Most croquet strokes are 5-15 yards, not 20+ yards
        max_croquet_power = config.MAX_SHOT_POWER * 1.2
        return min(velocity * 1.2, max_croquet_power)
