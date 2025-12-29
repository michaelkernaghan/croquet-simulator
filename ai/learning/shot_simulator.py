"""
Shot Simulator - Monte Carlo simulation of shot outcomes.

Simulates shots to predict where balls will end up,
accounting for physics and skill-based variance.
"""
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from models.ball import Ball, Vector2
from models.court import Court
import config


@dataclass
class ShotOutcome:
    """Result of a simulated shot."""
    # Final positions
    striker_position: Vector2
    other_ball_positions: Dict[str, Vector2]

    # What happened
    hoop_run: bool = False
    roqueted_ball: Optional[str] = None
    hit_boundary: bool = False

    # Quality metrics
    success: bool = False  # Did the shot achieve its goal?
    expected_value: float = 0.0  # Estimated value of resulting position


class ShotSimulator:
    """
    Simulates shot outcomes using simplified physics.

    Uses Monte Carlo sampling to account for skill-based variance.
    """

    def __init__(self, num_simulations: int = 50):
        """
        Initialize simulator.

        Args:
            num_simulations: Number of Monte Carlo samples per shot
        """
        self.num_simulations = num_simulations

    def simulate_shot(
        self,
        striker: Ball,
        target_angle: float,
        target_power: float,
        skill_level: float,
        all_balls: Dict[str, Ball],
        court: Court
    ) -> List[ShotOutcome]:
        """
        Simulate a shot multiple times with skill-based variance.

        Args:
            striker: The ball being shot
            target_angle: Intended angle in radians
            target_power: Intended power
            skill_level: Player skill (0-1), affects variance
            all_balls: All balls on the court
            court: The court

        Returns:
            List of ShotOutcome results from simulations
        """
        outcomes = []

        for _ in range(self.num_simulations):
            # Add skill-based variance
            angle_error = random.gauss(0, (1 - skill_level) * 0.25)
            power_error = random.gauss(0, (1 - skill_level) * 0.15)

            actual_angle = target_angle + angle_error
            actual_power = target_power * (1 + power_error)
            actual_power = max(0.5, min(config.MAX_SHOT_POWER, actual_power))

            # Create velocity vector
            velocity = Vector2(
                math.cos(actual_angle) * actual_power,
                math.sin(actual_angle) * actual_power
            )

            # Simulate the shot
            outcome = self._simulate_single_shot(
                striker, velocity, all_balls, court
            )
            outcomes.append(outcome)

        return outcomes

    def _simulate_single_shot(
        self,
        striker: Ball,
        velocity: Vector2,
        all_balls: Dict[str, Ball],
        court: Court
    ) -> ShotOutcome:
        """
        Simulate a single shot to completion.

        Uses simplified physics (no need for full engine accuracy).
        """
        # Copy positions
        positions = {
            color: ball.position.copy()
            for color, ball in all_balls.items()
        }

        striker_pos = positions[striker.color]
        start_pos = striker_pos.copy()
        striker_vel = velocity.copy()

        # Track what happened
        roqueted_ball = None
        hit_boundary = False
        hoop_run = False

        # Simplified simulation
        dt = 0.05  # Larger time step for speed
        max_steps = 200
        friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY

        for _ in range(max_steps):
            if striker_vel.magnitude() < config.MIN_VELOCITY:
                break

            # Update striker position
            striker_pos = striker_pos + striker_vel * dt

            # Apply friction
            speed = striker_vel.magnitude()
            new_speed = max(0, speed - friction_decel * dt)
            if new_speed > 0:
                striker_vel = striker_vel.normalize() * new_speed
            else:
                striker_vel = Vector2(0, 0)

            # Check boundary collision
            if striker_pos.x < 0 or striker_pos.x > court.width:
                striker_vel.x *= -0.8
                striker_pos.x = max(0.1, min(court.width - 0.1, striker_pos.x))
                hit_boundary = True
            if striker_pos.y < 0 or striker_pos.y > court.height:
                striker_vel.y *= -0.8
                striker_pos.y = max(0.1, min(court.height - 0.1, striker_pos.y))
                hit_boundary = True

            # Check ball collisions
            if roqueted_ball is None:  # Only count first roquet
                for color, pos in positions.items():
                    if color == striker.color:
                        continue
                    dist = (striker_pos - pos).magnitude()
                    if dist < config.BALL_RADIUS_YARDS * 4:
                        roqueted_ball = color
                        # Simple collision response
                        normal = (pos - striker_pos).normalize()
                        striker_vel = striker_vel - normal * striker_vel.dot(normal) * 1.6
                        # Move the other ball
                        positions[color] = pos + normal * 2.0
                        break

        # Update final positions
        positions[striker.color] = striker_pos

        # Check for hoop run (simplified)
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)
        if target_hoop:
            movement = striker_pos - start_pos
            if movement.magnitude() > 0.5:
                to_hoop = target_hoop.position - start_pos
                if to_hoop.magnitude() > 0:
                    # Check if we crossed the hoop
                    start_dist = to_hoop.dot(target_hoop.direction)
                    end_to_hoop = target_hoop.position - striker_pos
                    end_dist = end_to_hoop.dot(target_hoop.direction)

                    if start_dist > 0 and end_dist <= 0.5:
                        perp_dist = abs((striker_pos - target_hoop.position).dot(
                            Vector2(-target_hoop.direction.y, target_hoop.direction.x)
                        ))
                        if perp_dist < 1.5:
                            hoop_run = True

        return ShotOutcome(
            striker_position=striker_pos,
            other_ball_positions={c: p for c, p in positions.items() if c != striker.color},
            hoop_run=hoop_run,
            roqueted_ball=roqueted_ball,
            hit_boundary=hit_boundary
        )

    def evaluate_shot_options(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        court: Court,
        skill_level: float,
        position_evaluator,
        deadness: Dict[str, set] = None
    ) -> List[Tuple[float, float, float, ShotOutcome]]:
        """
        Evaluate multiple shot options and return sorted by expected value.

        Args:
            striker: The ball being shot
            all_balls: All balls on the court
            court: The court
            skill_level: Player skill level
            position_evaluator: PositionEvaluator for scoring positions
            deadness: Which balls striker is dead on (won't consider roquet shots at these)

        Returns:
            List of (angle, power, expected_value, sample_outcome) sorted by value
        """
        if deadness is None:
            deadness = {c: set() for c in ["blue", "black", "red", "yellow"]}

        # Get balls we're dead on - don't consider shooting at these
        dead_on = deadness.get(striker.color, set())

        candidates = []

        # Generate candidate shots
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)

        # Shot at hoop
        if target_hoop:
            to_hoop = target_hoop.position - striker.position
            hoop_angle = math.atan2(to_hoop.y, to_hoop.x)
            hoop_distance = to_hoop.magnitude()
            hoop_power = self._power_for_distance(hoop_distance + 3)
            candidates.append((hoop_angle, hoop_power, "hoop"))

        # Shots at other balls (roquet attempts) - ONLY at live balls!
        for color, ball in all_balls.items():
            if color == striker.color:
                continue
            if color in dead_on:
                continue  # Skip dead balls - no point shooting at them!
            to_ball = ball.position - striker.position
            ball_angle = math.atan2(to_ball.y, to_ball.x)
            ball_distance = to_ball.magnitude()
            ball_power = self._power_for_distance(ball_distance + 0.5)
            candidates.append((ball_angle, ball_power, f"roquet_{color}"))

        # Evaluate each candidate
        results = []
        for angle, power, shot_type in candidates:
            outcomes = self.simulate_shot(
                striker, angle, power, skill_level, all_balls, court
            )

            # Calculate expected value from outcomes
            total_value = 0.0
            for outcome in outcomes:
                # Immediate rewards
                value = 0.0
                if outcome.hoop_run:
                    value += 20.0  # Big bonus for running hoop
                if outcome.roqueted_ball:
                    value += 10.0  # Bonus for roquet (earn strokes)

                # Position value
                # Would need to create temporary balls to evaluate properly
                # For now, use distance-based heuristics
                if target_hoop:
                    dist_to_hoop = (outcome.striker_position - target_hoop.position).magnitude()
                    value -= dist_to_hoop * 0.5  # Prefer closer to hoop

                total_value += value

            expected_value = total_value / len(outcomes)
            results.append((angle, power, expected_value, outcomes[0]))

        # Sort by expected value (best first)
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def _power_for_distance(self, distance: float) -> float:
        """Calculate power needed for a distance."""
        friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY
        velocity = math.sqrt(2 * friction_decel * distance)
        return min(velocity * 1.1, config.MAX_SHOT_POWER)
