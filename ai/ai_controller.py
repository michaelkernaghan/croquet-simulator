"""
AI Controller - Manages AI decision making for shots.

Coordinates between:
- Opening strategy: First 4 turns with tice/supershot placements
- Break planning: 2/3/4-ball break building with pilot/pioneer/pivot
- Croquet strokes: Drive, roll, take-off, split shots
"""
import math
from typing import Dict, Tuple, Optional

import config
from models.ball import Ball, Vector2
from models.court import Court
from ai.basic_strategy import BasicStrategy
from ai.break_strategy import BreakPlanner, BreakPlan
from ai.opening_strategy import OpeningPlanner
from physics.croquet_strokes import CroquetStrokeCalculator, StrokeType, StrokeResult


class AIController:
    """
    Controls AI decision-making for a ball/player.

    Uses opening strategy for early game, break planning for mid-game,
    and croquet stroke selection for strategic play.
    """

    def __init__(self, strategy=None, aggression: float = 0.5):
        """
        Initialize AI controller.

        Args:
            strategy: Strategy to use (defaults to BasicStrategy)
            aggression: How aggressive to play openings (0-1)
        """
        self.strategy = strategy or BasicStrategy(skill_level=0.75)
        self.break_planner = BreakPlanner()
        self.opening_planner = OpeningPlanner(aggression=aggression)
        self.stroke_calculator = CroquetStrokeCalculator()
        self.current_break_plan: Optional[BreakPlan] = None

    def select_shot(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        court: Court,
        target_hoop_num: int = None,
        balls_in_play: Dict[str, bool] = None,
        deadness: Dict[str, set] = None,
        strokes_remaining: int = 1,
        is_continuation: bool = False
    ) -> Tuple[Vector2, str]:
        """
        Select a shot for the given ball.

        Args:
            ball: The ball to shoot
            balls: All balls on the court
            court: The court
            target_hoop_num: Which hoop to aim for (Golf Croquet mode, None for Association)
            balls_in_play: Which balls have entered play (for opening detection)
            deadness: Which balls the striker is dead on
            strokes_remaining: Strokes remaining in this turn
            is_continuation: Whether this is a continuation stroke

        Returns:
            Tuple of (velocity Vector2, description string)
        """
        # Check if we're in opening phase
        if balls_in_play is not None and self.opening_planner.is_opening_phase(balls_in_play):
            return self._select_opening_shot(ball, balls, balls_in_play, court)

        # Get the target hoop
        if target_hoop_num is not None:
            target_hoop = court.get_hoop(target_hoop_num)
            hoop_num = target_hoop_num
        else:
            # Association Croquet: ball aims for its own next hoop
            target_hoop = court.get_hoop_for_ball(ball.hoops_run)
            hoop_num = ball.get_next_hoop_number()

        # Determine if this is a run attempt or positioning shot
        is_running = False
        if target_hoop and hasattr(self.strategy, '_is_good_approach'):
            is_running = self.strategy._is_good_approach(ball, target_hoop)

        # Use strategy to get angle and power, passing continuation context
        if hasattr(self.strategy, 'select_shot'):
            # Learning strategy supports extra parameters
            try:
                angle, power = self.strategy.select_shot(
                    ball, balls, court, target_hoop_num,
                    deadness=deadness,
                    strokes_remaining=strokes_remaining,
                    is_continuation=is_continuation
                )
            except TypeError:
                # Fallback for strategies that don't support extra params
                angle, power = self.strategy.select_shot(ball, balls, court, target_hoop_num)
        else:
            angle, power = self.strategy.select_shot(ball, balls, court, target_hoop_num)

        # Convert to velocity vector
        velocity = Vector2.from_angle(angle, power)

        # Generate description based on shot type
        if target_hoop and hoop_num > 0:
            if is_running:
                description = f"{ball.color.capitalize()} runs at hoop {hoop_num}"
            else:
                description = f"{ball.color.capitalize()} sets up for hoop {hoop_num}"
        elif ball.is_rover:
            description = f"{ball.color.capitalize()} aims for peg"
        else:
            description = f"{ball.color.capitalize()} shoots"

        return (velocity, description)

    def _select_opening_shot(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        balls_in_play: Dict[str, bool],
        court: Court
    ) -> Tuple[Vector2, str]:
        """
        Select an opening shot using opening strategy.

        Args:
            ball: The ball to shoot
            balls: All balls on court
            balls_in_play: Which balls have entered play
            court: The court

        Returns:
            Tuple of (velocity, description)
        """
        # Count turn number based on balls in play
        turn_number = sum(1 for v in balls_in_play.values() if v) + 1

        # Get opening shot recommendation
        target, power, description = self.opening_planner.get_opening_shot(
            ball, balls, balls_in_play, court, turn_number
        )

        # Convert target position to velocity
        to_target = target - ball.position
        angle = math.atan2(to_target.y, to_target.x)
        velocity = Vector2.from_angle(angle, power)

        return (velocity, f"{ball.color.capitalize()}: {description}")

    def select_croquet_placement(
        self,
        striker: Ball,
        roqueted: Ball,
        balls: Dict[str, Ball],
        court: Court
    ) -> Vector2:
        """
        Select where to place the striker ball for a croquet stroke.

        In croquet, after roqueting a ball, you place your ball in contact
        with the roqueted ball, then hit your ball causing both to move.

        Strategic placement depends on where you want to send both balls.

        Args:
            striker: The striker ball
            roqueted: The ball that was roqueted
            balls: All balls on the court
            court: The court

        Returns:
            Position to place the striker ball
        """
        # Get the next hoop for the striker
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)

        if target_hoop:
            # Place ball so we can rush toward the hoop
            # Direction from roqueted ball to hoop
            to_hoop = target_hoop.position - roqueted.position
            to_hoop_dir = to_hoop.normalize()

            # Place striker ball behind roqueted ball (opposite direction to hoop)
            # This way we can send roqueted ball toward hoop and follow
            contact_distance = (striker.radius + roqueted.radius) * 2 + 0.05
            placement = roqueted.position - to_hoop_dir * contact_distance
        else:
            # Default: place to one side
            placement = roqueted.position + Vector2(0.15, 0)

        # Ensure placement is within court bounds
        margin = 0.5
        placement.x = max(margin, min(court.width - margin, placement.x))
        placement.y = max(margin, min(court.height - margin, placement.y))

        return placement

    def select_croquet_shot(
        self,
        striker: Ball,
        roqueted: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set] = None
    ) -> Tuple[Vector2, Vector2, str, StrokeType]:
        """
        Select the direction and power for a croquet stroke.

        KEY PRINCIPLE (from Wylie): The croquet stroke is the heart of break-building.
        It should achieve TWO objectives simultaneously:
        1. Send the croqueted ball to a STRATEGIC position (pioneer at next hoop, or useful location)
        2. Position the striker to get a RUSH on another LIVE ball toward the current hoop

        The pattern is:
        - Roquet ball A -> Croquet: send A as pioneer, get position for rush on B
        - Rush B to hoop -> Run hoop -> Rush to next ball -> Repeat

        Args:
            striker: The striker ball (placed in contact with roqueted)
            roqueted: The ball that was roqueted
            balls: All balls on the court
            court: The court
            deadness: Which balls striker is dead on

        Returns:
            Tuple of (striker_velocity, croqueted_velocity, description, stroke_type)
        """
        if deadness is None:
            deadness = {c: set() for c in ["blue", "black", "red", "yellow"]}

        # Get which balls we're dead on (including the one we just roqueted)
        dead_on = deadness.get(striker.color, set())
        # The roqueted ball is now dead - we can't roquet it again until we run a hoop
        dead_on = dead_on | {roqueted.color}

        # Find LIVE balls we can rush after this croquet stroke
        live_balls = []
        for color, ball in balls.items():
            if color != striker.color and color not in dead_on and not ball.has_pegged_out:
                live_balls.append((color, ball))

        # Update break plan
        self.current_break_plan = self.break_planner.analyze_position(
            striker, balls, court, deadness
        )

        # Get the target hoops
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)
        next_hoop = court.get_hoop_for_ball(striker.hoops_run + 1)
        hoop_after_next = court.get_hoop_for_ball(striker.hoops_run + 2)

        # Print croquet planning info
        print(f"  [CROQUET] {striker.color} taking croquet on {roqueted.color}")
        print(f"    Live balls for rush: {[c for c, _ in live_balls]}")
        if next_hoop:
            print(f"    Pioneer target: hoop {striker.hoops_run + 2} at {next_hoop.position}")

        # STRATEGIC CROQUET SHOT SELECTION (4-ball break pattern)
        # Priority 1: If we have live balls, position to RUSH one toward hoop
        # Priority 2: Send croqueted ball as PIONEER to next hoop
        # Priority 3: If no live balls, position for direct hoop approach

        if target_hoop and live_balls:
            # Find the best ball to rush and calculate ideal positions
            best_rush_ball, striker_target, rush_description = self._find_best_rush_setup(
                striker, live_balls, target_hoop, court
            )

            if best_rush_ball:
                # We have a live ball to rush - this is ideal break play!
                # Send croqueted ball as pioneer, position for rush
                pioneer_dest = "center (pivot)"
                if next_hoop:
                    # Pioneer position: 3-4 yards in front of next hoop
                    croqueted_target = next_hoop.position - next_hoop.direction * 4
                    pioneer_dest = f"hoop {striker.hoops_run + 2}"
                elif hoop_after_next:
                    croqueted_target = hoop_after_next.position - hoop_after_next.direction * 4
                    pioneer_dest = f"hoop {striker.hoops_run + 3}"
                else:
                    # Send to center as pivot
                    croqueted_target = Vector2(court.width / 2, court.height / 2)

                stroke_type, aim_dir, power, split_angle = self.stroke_calculator.select_best_stroke(
                    striker, roqueted, striker_target, croqueted_target, court
                )

                result = self.stroke_calculator.calculate_stroke(
                    stroke_type, striker, roqueted, aim_dir, power, split_angle
                )

                # Verbose break description
                print(f"    SPLIT: {roqueted.color} -> {pioneer_dest}, striker -> {rush_description}")
                description = f"{striker.color.capitalize()} {result.description}: {roqueted.color} as pioneer to {pioneer_dest}, {rush_description}"
                return (result.striker_velocity, result.croqueted_velocity, description, stroke_type)

        # No live balls to rush - simpler approach
        if target_hoop:
            to_hoop = target_hoop.position - striker.position
            distance_to_hoop = to_hoop.magnitude()

            if distance_to_hoop < 5:
                # Already close to hoop - take-off to run it, send croqueted away
                striker_target = target_hoop.position - target_hoop.direction * 2

                if next_hoop:
                    croqueted_target = next_hoop.position - next_hoop.direction * 4
                else:
                    croqueted_target = Vector2(court.width / 2, court.height / 2)

                stroke_type, aim_dir, power, split_angle = self.stroke_calculator.select_best_stroke(
                    striker, roqueted, striker_target, croqueted_target, court
                )

                result = self.stroke_calculator.calculate_stroke(
                    stroke_type, striker, roqueted, aim_dir, power, split_angle
                )

                description = f"{striker.color.capitalize()} {result.description} to approach hoop"
            else:
                # Far from hoop, no rush available - drive toward hoop
                striker_target = target_hoop.position - target_hoop.direction * 4

                if next_hoop:
                    croqueted_target = next_hoop.position - next_hoop.direction * 4
                else:
                    croqueted_target = Vector2(court.width / 2, court.height / 2)

                stroke_type, aim_dir, power, split_angle = self.stroke_calculator.select_best_stroke(
                    striker, roqueted, striker_target, croqueted_target, court
                )

                result = self.stroke_calculator.calculate_stroke(
                    stroke_type, striker, roqueted, aim_dir, power, split_angle
                )

                description = f"{striker.color.capitalize()} {result.description}"
        else:
            # Rover - aim toward peg
            peg_pos = court.peg_position
            aim_dir = (peg_pos - roqueted.position).normalize()
            result = self.stroke_calculator.calculate_stroke(
                StrokeType.DRIVE, striker, roqueted, aim_dir, 6.0
            )
            stroke_type = StrokeType.DRIVE
            description = f"{striker.color.capitalize()} drives toward peg"

        return (result.striker_velocity, result.croqueted_velocity, description, stroke_type)

    def _find_best_rush_setup(
        self,
        striker: Ball,
        live_balls: list,
        target_hoop,
        court: Court
    ) -> Tuple[Optional[Ball], Optional[Vector2], str]:
        """
        Find the best live ball to set up a rush toward the hoop.

        The ideal croquet shot positions the striker BEHIND a live ball,
        so the striker can then roquet (rush) that ball toward the hoop.

        Returns:
            Tuple of (ball to rush, striker target position, description)
            Returns (None, None, "") if no good rush available
        """
        best_ball = None
        best_position = None
        best_score = 0
        best_description = ""

        for color, ball in live_balls:
            # Calculate where striker should be to rush this ball toward hoop
            # Ideal: striker behind ball, with ball between striker and hoop
            ball_to_hoop = target_hoop.position - ball.position
            ball_to_hoop_dist = ball_to_hoop.magnitude()

            if ball_to_hoop_dist < 1:
                continue  # Ball too close to hoop

            rush_direction = ball_to_hoop.normalize()

            # Striker should be 2-4 yards behind the ball (opposite to rush direction)
            ideal_striker_pos = ball.position - rush_direction * 3

            # Check if this is a reasonable position (1 yard from boundary)
            if not court.is_in_bounds(ideal_striker_pos, radius=1):
                continue

            # Score this rush setup
            # Factors: distance to ball, alignment quality, ball distance to hoop
            striker_to_ball = (ball.position - ideal_striker_pos).magnitude()

            # Good rush: straight or slight cut (not too angled)
            # The alignment is already good by construction

            # Prefer balls that are closer to the hoop (shorter rush needed)
            hoop_proximity_score = max(0, 1 - ball_to_hoop_dist / 15)

            # Prefer balls that are easier to reach
            reach_score = max(0, 1 - striker_to_ball / 10)

            # Combined score
            score = hoop_proximity_score * 0.6 + reach_score * 0.4

            if score > best_score:
                best_score = score
                best_ball = ball
                best_position = ideal_striker_pos
                best_description = f"rush {color} to hoop"

        # Only return if we found a decent option
        if best_score > 0.3:
            return (best_ball, best_position, best_description)

        return (None, None, "")

    def set_strategy(self, strategy):
        """Change the AI strategy."""
        self.strategy = strategy

    def get_strategy_name(self) -> str:
        """Get the name of the current strategy."""
        return self.strategy.name
