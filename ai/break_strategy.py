"""
Break Strategy - Strategic planning for multi-hoop turns.

Implements authentic croquet break-building concepts:

2-Ball Break:
- Only striker and one other ball (pilot/reception)
- Very difficult - requires flawless rush and hoop control
- Even strong players struggle beyond a few hoops

3-Ball Break:
- Adds a pioneer at the next hoop
- Pattern: roquet reception -> croquet to next hoop as pioneer ->
           roquet pilot -> score hoop
- Good players can complete all hoops in one turn

4-Ball Break:
- Adds a pivot ball near midcourt
- By far the easiest break to play
- Pivot eliminates difficult strokes and enables recovery

Key concepts:
- Hoop control: Running hoops to a specific spot for advantageous rush
- Pioneer placement: Keep pioneers within inner rectangle of corner hoops
- Rush direction: Position for rush toward next ball/hoop
"""
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto

from models.ball import Ball, Vector2
from models.court import Court
from physics.croquet_strokes import StrokeType, CroquetStrokeCalculator


class BallRole(Enum):
    """Role a ball plays in a break."""
    PILOT = auto()      # Ball near current hoop to use for approach (reception ball)
    PIONEER = auto()    # Ball pre-positioned at next hoop
    PIVOT = auto()      # Ball in center for flexibility (4-ball break)
    BACKUP = auto()     # Backup pioneer or spare ball
    UNUSED = auto()     # Not currently part of the break


class BreakType(Enum):
    """Type of break being played."""
    TWO_BALL = auto()   # Striker + 1 ball - hardest
    THREE_BALL = auto() # Striker + 2 balls - manageable
    FOUR_BALL = auto()  # Striker + 3 balls - easiest


@dataclass
class BreakPlan:
    """Plan for executing a break."""
    # Break type
    break_type: BreakType = BreakType.TWO_BALL

    # Ball assignments
    pilot_ball: Optional[str] = None      # Which ball is pilot/reception
    pioneer_ball: Optional[str] = None    # Which ball is pioneer
    pivot_ball: Optional[str] = None      # Which ball is pivot

    # Targets - where balls should ideally be
    pilot_target: Optional[Vector2] = None      # Where pilot should be
    pioneer_target: Optional[Vector2] = None    # Where pioneer should be
    pivot_target: Optional[Vector2] = None      # Where pivot should be

    # Rush targets - where to rush balls after roquet
    rush_direction: Optional[Vector2] = None    # Direction to rush pilot

    # Quality assessment
    break_quality: float = 0.0    # 0-1, how good is this break setup
    hoop_control_quality: float = 0.0  # How well positioned for controlled hoop run


class BreakPlanner:
    """
    Plans and evaluates break opportunities.

    A break is a sequence of hoops scored in one turn by using
    roquets and croquet strokes to position balls strategically.

    The inner rectangle (bounded by the four corner hoops) is the
    ideal zone for keeping balls during a break.
    """

    # Inner rectangle bounds (corner hoops form the boundary)
    INNER_RECT_MIN_X = 7   # Hoop 1/2 x-position
    INNER_RECT_MAX_X = 21  # Hoop 3/4 x-position
    INNER_RECT_MIN_Y = 7   # Hoop 1/4 y-position
    INNER_RECT_MAX_Y = 28  # Hoop 2/3 y-position

    def __init__(self):
        """Initialize break planner."""
        self.stroke_calc = CroquetStrokeCalculator()

    def analyze_position(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set]
    ) -> BreakPlan:
        """
        Analyze current position and create a break plan.

        Determines break type (2/3/4-ball) based on available balls
        and assigns optimal roles.

        Args:
            striker: The ball currently playing
            all_balls: All balls on court
            court: The court
            deadness: Which balls striker is dead on

        Returns:
            BreakPlan with ball assignments and targets
        """
        plan = BreakPlan()

        # Get target hoops
        current_hoop = court.get_hoop_for_ball(striker.hoops_run)
        next_hoop = court.get_hoop_for_ball(striker.hoops_run + 1)
        hoop_after_next = court.get_hoop_for_ball(striker.hoops_run + 2)

        if not current_hoop:
            return plan

        # Get available balls (not dead on)
        dead_on = deadness.get(striker.color, set())
        available = [c for c in all_balls.keys()
                     if c != striker.color and c not in dead_on]

        if not available:
            return plan

        # Determine break type based on available balls
        if len(available) >= 3:
            plan.break_type = BreakType.FOUR_BALL
        elif len(available) >= 2:
            plan.break_type = BreakType.THREE_BALL
        else:
            plan.break_type = BreakType.TWO_BALL

        # Score each ball for each role
        ball_scores = {}
        for color in available:
            ball = all_balls[color]
            ball_scores[color] = {
                'pilot': self._score_as_pilot(ball, striker, current_hoop),
                'pioneer': self._score_as_pioneer(ball, next_hoop, court) if next_hoop else 0,
                'pivot': self._score_as_pivot(ball, court),
            }

        # Assign roles based on scores - priority: pilot > pioneer > pivot
        assigned = set()

        # Best pilot (reception ball)
        best_pilot = max(available, key=lambda c: ball_scores[c]['pilot'])
        if ball_scores[best_pilot]['pilot'] > 0.2:
            plan.pilot_ball = best_pilot
            assigned.add(best_pilot)

        # Best pioneer (from remaining) - critical for 3+ ball breaks
        remaining = [c for c in available if c not in assigned]
        if remaining and next_hoop:
            best_pioneer = max(remaining, key=lambda c: ball_scores[c]['pioneer'])
            if ball_scores[best_pioneer]['pioneer'] > 0.15:
                plan.pioneer_ball = best_pioneer
                assigned.add(best_pioneer)

        # Best pivot (from remaining) - for 4-ball breaks
        remaining = [c for c in available if c not in assigned]
        if remaining:
            best_pivot = max(remaining, key=lambda c: ball_scores[c]['pivot'])
            plan.pivot_ball = best_pivot

        # Calculate ideal target positions
        self._calculate_targets(plan, striker, current_hoop, next_hoop, hoop_after_next, court)

        # Calculate rush direction for hoop control
        if plan.pilot_ball and next_hoop:
            # After running hoop, want to rush toward pioneer at next hoop
            plan.rush_direction = (next_hoop.position - current_hoop.position).normalize()

        # Calculate break quality
        plan.break_quality = self._calculate_break_quality(plan, all_balls, court)
        plan.hoop_control_quality = self._calculate_hoop_control(striker, current_hoop, plan)

        return plan

    def _calculate_targets(
        self,
        plan: BreakPlan,
        striker: Ball,
        current_hoop,
        next_hoop,
        hoop_after_next,
        court: Court
    ):
        """Calculate ideal target positions for each ball role."""
        if current_hoop:
            # Pilot should be 2-4 yards in front of hoop, positioned for rush
            # After roqueting pilot, we want to be able to take off/rush toward hoop
            approach_pos = current_hoop.position - current_hoop.direction * 3
            plan.pilot_target = approach_pos

        if next_hoop:
            # Pioneer placement depends on which hoop
            # Special case: 1-back pioneer should be south of hoop for easier
            # continuation to 2-back
            hoop_num = (striker.hoops_run + 1) % 6 + 1

            if hoop_num == 1 and striker.hoops_run >= 6:
                # 1-back: position south of hoop
                plan.pioneer_target = next_hoop.position - Vector2(0, 4)
            else:
                # Standard: 3-4 yards in front of next hoop
                plan.pioneer_target = next_hoop.position - next_hoop.direction * 4

            # Keep pioneer within inner rectangle
            plan.pioneer_target = self._clamp_to_inner_rect(plan.pioneer_target)

        # Pivot near center, but biased toward the hoops we're playing
        if current_hoop and next_hoop:
            # Pivot between current and next hoop areas
            mid_point = (current_hoop.position + next_hoop.position) * 0.5
            center = Vector2(court.width / 2, court.height / 2)
            # Weighted toward center but influenced by hoop positions
            plan.pivot_target = mid_point * 0.3 + center * 0.7
        else:
            plan.pivot_target = Vector2(court.width / 2, court.height / 2)

    def _clamp_to_inner_rect(self, pos: Vector2) -> Vector2:
        """Clamp position to inner rectangle (bounded by corner hoops)."""
        return Vector2(
            max(self.INNER_RECT_MIN_X, min(self.INNER_RECT_MAX_X, pos.x)),
            max(self.INNER_RECT_MIN_Y, min(self.INNER_RECT_MAX_Y, pos.y))
        )

    def _is_in_inner_rect(self, pos: Vector2) -> bool:
        """Check if position is within inner rectangle."""
        return (self.INNER_RECT_MIN_X <= pos.x <= self.INNER_RECT_MAX_X and
                self.INNER_RECT_MIN_Y <= pos.y <= self.INNER_RECT_MAX_Y)

    def _score_as_pilot(self, ball: Ball, striker: Ball, hoop) -> float:
        """
        Score how good a ball is as pilot for current hoop.

        Good pilot characteristics:
        - 2-5 yards from hoop, in front (approach side)
        - Accessible from striker's position (can be roqueted)
        - Position allows rush toward hoop after roquet
        """
        if not hoop:
            return 0.0

        to_hoop = hoop.position - ball.position
        dist_to_hoop = to_hoop.magnitude()

        # Too close or too far from hoop
        if dist_to_hoop < 1 or dist_to_hoop > 12:
            return 0.1

        # Check if in front of hoop (approach side)
        if dist_to_hoop > 0.5:
            approach_dot = to_hoop.normalize().dot(hoop.direction)
            if approach_dot < 0:
                return 0.15  # Wrong side of hoop

        # Score based on distance from hoop (3-4 yards is ideal)
        ideal_dist = 3.5
        dist_score = 1.0 - abs(dist_to_hoop - ideal_dist) / 10

        # Bonus if striker can easily roquet this ball
        striker_to_ball = (ball.position - striker.position).magnitude()
        if striker_to_ball < 8:
            dist_score += 0.2
        elif striker_to_ball > 15:
            dist_score -= 0.2

        # Bonus if ball is in good rush position (between striker and hoop)
        striker_to_hoop = hoop.position - striker.position
        if striker_to_hoop.magnitude() > 0.5:
            # Check if pilot is roughly between striker and hoop
            to_pilot = ball.position - striker.position
            if to_pilot.magnitude() > 0.5:
                alignment = to_pilot.normalize().dot(striker_to_hoop.normalize())
                if alignment > 0.5:
                    dist_score += 0.15  # Good rush alignment

        return max(0.1, min(1.0, dist_score))

    def _score_as_pioneer(self, ball: Ball, hoop, court: Court) -> float:
        """
        Score how good a ball is as pioneer for next hoop.

        Good pioneer characteristics:
        - 3-5 yards from next hoop, in front
        - Within inner rectangle (bounded by corner hoops)
        - Not too far from current action
        """
        if not hoop:
            return 0.0

        to_hoop = hoop.position - ball.position
        dist = to_hoop.magnitude()

        # Ideal pioneer is 3-5 yards from hoop
        if dist < 2 or dist > 15:
            return 0.1

        # Check if in front of hoop
        if dist > 0.5:
            approach_dot = to_hoop.normalize().dot(hoop.direction)
            if approach_dot < 0:
                return 0.15  # Wrong side

        # Score based on distance (4 yards is ideal)
        ideal_dist = 4
        dist_score = 1.0 - abs(dist - ideal_dist) / 12

        # Bonus for being in inner rectangle
        if self._is_in_inner_rect(ball.position):
            dist_score += 0.15

        return max(0.1, min(1.0, dist_score))

    def _calculate_hoop_control(self, striker: Ball, hoop, plan: BreakPlan) -> float:
        """
        Calculate hoop control quality.

        Hoop control means running the hoop to a specific point
        that allows a rush in an advantageous direction.
        """
        if not hoop:
            return 0.0

        to_hoop = hoop.position - striker.position
        dist = to_hoop.magnitude()

        if dist > 6:
            return 0.0  # Too far for controlled run

        # Check approach angle
        if dist > 0.5:
            approach_dot = to_hoop.normalize().dot(hoop.direction)
            if approach_dot < 0.3:
                return 0.1  # Bad angle

            # Better angle = better control
            angle_quality = approach_dot

            # Closer = better control
            dist_quality = max(0, 1 - dist / 6)

            # Can we rush toward next objective after running?
            rush_quality = 0.5
            if plan.rush_direction:
                # Check if we can position for rush after hoop
                exit_pos = hoop.position + hoop.direction * 2
                # Ideally exit toward rush direction
                rush_quality = 0.5 + 0.5 * max(0, hoop.direction.dot(plan.rush_direction))

            return angle_quality * 0.4 + dist_quality * 0.3 + rush_quality * 0.3

        return 0.0

    def _score_as_pivot(self, ball: Ball, court: Court) -> float:
        """Score how good a ball is as pivot."""
        center = Vector2(court.width / 2, court.height / 2)
        dist_to_center = (ball.position - center).magnitude()

        # Pivot should be near center
        max_useful_dist = 12
        if dist_to_center > max_useful_dist:
            return 0.2

        return 1.0 - (dist_to_center / max_useful_dist) * 0.8

    def _calculate_break_quality(
        self,
        plan: BreakPlan,
        all_balls: Dict[str, Ball],
        court: Court
    ) -> float:
        """Calculate overall break quality (0-1)."""
        quality = 0.0
        factors = 0

        # Having a good pilot is important
        if plan.pilot_ball:
            ball = all_balls[plan.pilot_ball]
            if plan.pilot_target:
                dist = (ball.position - plan.pilot_target).magnitude()
                pilot_quality = max(0, 1 - dist / 10)
                quality += pilot_quality * 0.4
                factors += 0.4

        # Having a pioneer makes 3+ ball break possible
        if plan.pioneer_ball:
            ball = all_balls[plan.pioneer_ball]
            if plan.pioneer_target:
                dist = (ball.position - plan.pioneer_target).magnitude()
                pioneer_quality = max(0, 1 - dist / 12)
                quality += pioneer_quality * 0.35
                factors += 0.35

        # Having a pivot makes 4-ball break possible
        if plan.pivot_ball:
            ball = all_balls[plan.pivot_ball]
            if plan.pivot_target:
                dist = (ball.position - plan.pivot_target).magnitude()
                pivot_quality = max(0, 1 - dist / 15)
                quality += pivot_quality * 0.25
                factors += 0.25

        return quality / factors if factors > 0 else 0.0

    def suggest_shot(
        self,
        striker: Ball,
        plan: BreakPlan,
        all_balls: Dict[str, Ball],
        court: Court,
        is_croquet: bool = False,
        croqueted_ball: Optional[Ball] = None
    ) -> Tuple[str, Vector2, float, Optional[StrokeType]]:
        """
        Suggest best shot based on break plan.

        Args:
            striker: Ball taking the shot
            plan: Current break plan
            all_balls: All balls
            court: The court
            is_croquet: Whether this is a croquet stroke
            croqueted_ball: If croquet, the ball being croqueted

        Returns:
            Tuple of (description, target, power, stroke_type for croquet)
        """
        current_hoop = court.get_hoop_for_ball(striker.hoops_run)

        if is_croquet and croqueted_ball:
            return self._suggest_croquet_shot(
                striker, croqueted_ball, plan, all_balls, court
            )

        # Check if we can run the hoop
        if current_hoop:
            to_hoop = current_hoop.position - striker.position
            dist = to_hoop.magnitude()

            if dist < 5:
                approach_dot = to_hoop.normalize().dot(current_hoop.direction)
                if approach_dot > 0.5:
                    # Good position - run the hoop!
                    target = current_hoop.position + current_hoop.direction * 2
                    return ("Run hoop", target, self._power_for_dist(dist + 3), None)

        # Try to roquet pilot ball
        if plan.pilot_ball and plan.pilot_ball in all_balls:
            pilot = all_balls[plan.pilot_ball]
            dist = (pilot.position - striker.position).magnitude()
            return (f"Roquet pilot ({plan.pilot_ball})", pilot.position,
                    self._power_for_dist(dist + 0.5), None)

        # Try to roquet any ball
        for color, ball in all_balls.items():
            if color != striker.color:
                dist = (ball.position - striker.position).magnitude()
                return (f"Roquet {color}", ball.position,
                        self._power_for_dist(dist + 0.5), None)

        # Default: approach hoop
        if current_hoop:
            target = current_hoop.position - current_hoop.direction * 3
            dist = (target - striker.position).magnitude()
            return ("Approach hoop", target, self._power_for_dist(dist), None)

        return ("Center", Vector2(court.width/2, court.height/2), 5.0, None)

    def _suggest_croquet_shot(
        self,
        striker: Ball,
        croqueted: Ball,
        plan: BreakPlan,
        all_balls: Dict[str, Ball],
        court: Court
    ) -> Tuple[str, Vector2, float, StrokeType]:
        """Suggest the best croquet stroke to play."""
        current_hoop = court.get_hoop_for_ball(striker.hoops_run)
        next_hoop = court.get_hoop_for_ball(striker.hoops_run + 1)

        # Determine where we want each ball to go
        striker_target = None
        croqueted_target = None

        # If croqueted ball is the pilot, send it past hoop, position striker
        if plan.pilot_ball == croqueted.color and current_hoop:
            # Take-off: send croqueted near hoop, striker to good position
            croqueted_target = current_hoop.position - current_hoop.direction * 2
            striker_target = current_hoop.position - current_hoop.direction * 4

            stroke_type, aim, power, split = self.stroke_calc.select_best_stroke(
                striker, croqueted, striker_target, croqueted_target, court
            )

            return (f"Take-off to hoop", croqueted_target, power, stroke_type)

        # If croqueted ball should be pioneer, send to next hoop
        if next_hoop:
            croqueted_target = next_hoop.position - next_hoop.direction * 4

            if current_hoop:
                striker_target = current_hoop.position - current_hoop.direction * 3
            else:
                striker_target = striker.position + Vector2(3, 0)

            stroke_type, aim, power, split = self.stroke_calc.select_best_stroke(
                striker, croqueted, striker_target, croqueted_target, court
            )

            return (f"Send pioneer to next hoop", croqueted_target, power, stroke_type)

        # Default: drive both toward current hoop
        if current_hoop:
            target = current_hoop.position
            dist = (target - croqueted.position).magnitude()
            return ("Drive to hoop", target, self._power_for_dist(dist), StrokeType.DRIVE)

        return ("Drive center", Vector2(court.width/2, court.height/2), 5.0, StrokeType.DRIVE)

    def _power_for_dist(self, dist: float) -> float:
        """Calculate power for distance."""
        import config
        friction = config.FRICTION_COEFFICIENT * config.GRAVITY
        return min(math.sqrt(2 * friction * dist) * 1.2, config.MAX_SHOT_POWER)
