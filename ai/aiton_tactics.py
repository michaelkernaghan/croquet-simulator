"""
Aiton Tactics - Strategic positioning based on Keith Aiton's teachings.

Implements key tactical concepts from "The Basics" chapter:

HOOP APPROACHES (Section 2.3):
- Ideal approach: 1 yard in front of hoop with stop-shot ratio ~1:6
- Right side approaches are easier than left side
- 12 inches in front requires excellent stop-shot control

LEAVES (Section 2.5):
- Diagonal Spread: Balls spread to corners for defensive positioning
- NSL (New Standard Leave): Partner at hoop 2, opponents separated
- MSL (Maugham Standard Leave): Variation with different partner placement

BREAK BUILDING (Sections 2.4-2.6):
- Pioneer 3-4 yards in front of NEXT hoop
- Reception ball positioning determines approach quality
- 3-ball to 4-ball transition is critical for break continuation

STANDARD STROKES (Section 2.7):
- Roll from corner II to hoop 2 is a "standard stroke"
- Stop-shot for pioneer placement
- Take-off for position without moving croqueted ball much
"""
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto

from models.ball import Ball, Vector2
from models.court import Court


class LeaveType(Enum):
    """Types of standard leaves from Aiton."""
    DIAGONAL_SPREAD = auto()  # Balls spread diagonally across court
    NSL = auto()              # New Standard Leave
    MSL = auto()              # Maugham Standard Leave
    DEFENSIVE = auto()        # General defensive positioning


class ApproachSide(Enum):
    """Which side of hoop the approach is from."""
    RIGHT = auto()   # Easier - Aiton emphasizes this
    LEFT = auto()    # Harder
    STRAIGHT = auto() # Ideal


@dataclass
class ApproachQuality:
    """Assessment of a hoop approach position."""
    distance_to_hoop: float      # Yards from hoop
    approach_angle: float        # Radians off-center
    side: ApproachSide           # Left/right/straight
    success_probability: float   # 0-1 chance of running
    ideal_power: float           # Power needed for clean run
    recommendation: str          # What Aiton would advise


@dataclass
class LeavePosition:
    """Target positions for a standard leave."""
    striker_pos: Vector2         # Where striker ball should be
    partner_pos: Vector2         # Where partner should be
    opponent1_pos: Vector2       # Where first opponent should be
    opponent2_pos: Vector2       # Where second opponent should be
    description: str             # Leave name and purpose


@dataclass
class StandardStroke:
    """A named standard stroke pattern from Aiton."""
    name: str
    start_zone: Tuple[float, float, float, float]  # x1, y1, x2, y2 bounding box
    target_position: Vector2
    stroke_type: str  # "roll", "stop", "drive", "take-off"
    description: str


class AitonTactics:
    """
    Implements Keith Aiton's tactical principles for croquet.

    Key insights from the Aiton scans:
    1. Approach from 1 yard is ideal (Section 2.3)
    2. Right side approaches are easier than left
    3. Stop-shot ratio ~1:6 for approaches
    4. Pioneer should be 3-4 yards in front of next hoop
    5. Diagonal Spread, NSL, MSL for leaves
    6. Reception ball quality affects entire break
    """

    # Aiton's ideal approach distance (yards)
    IDEAL_APPROACH_DISTANCE = 1.0

    # Maximum approach distance for high success (yards)
    MAX_GOOD_APPROACH = 3.0

    # Distance where approach becomes risky
    RISKY_APPROACH_DISTANCE = 7.0

    # Ideal pioneer distance from hoop (yards)
    IDEAL_PIONEER_DISTANCE = 4.0

    # Approach angle thresholds (radians)
    STRAIGHT_ANGLE_THRESHOLD = math.radians(10)  # Within 10 degrees = straight
    ACCEPTABLE_ANGLE = math.radians(30)          # Within 30 degrees = acceptable
    DIFFICULT_ANGLE = math.radians(45)           # Beyond 45 degrees = difficult

    def __init__(self, court: Court):
        """Initialize with court reference."""
        self.court = court
        self._init_standard_strokes()

    def _init_standard_strokes(self):
        """Initialize the standard stroke patterns Aiton describes."""
        # "The roll approach from corner II to hoop 2 is another standard stroke"
        self.standard_strokes = [
            StandardStroke(
                name="Corner II to Hoop 2 Roll",
                start_zone=(0, 26, 4, 30),  # Near corner 2 area (NW)
                target_position=Vector2(7, 24),  # Approach to hoop 2
                stroke_type="roll",
                description="Standard roll from corner II to approach hoop 2"
            ),
            StandardStroke(
                name="Corner I to Hoop 1 Roll",
                start_zone=(0, 4, 4, 8),  # Near corner 1 area (SW)
                target_position=Vector2(7, 6),  # Approach to hoop 1
                stroke_type="roll",
                description="Roll from corner I to approach hoop 1"
            ),
            StandardStroke(
                name="Corner IV to Hoop 4 Roll",
                start_zone=(24, 4, 28, 8),  # Near corner 4 area (SE)
                target_position=Vector2(21, 8),  # Approach to hoop 4
                stroke_type="roll",
                description="Roll from corner IV to approach hoop 4"
            ),
            StandardStroke(
                name="Corner III to Hoop 3 Roll",
                start_zone=(24, 26, 28, 30),  # Near corner 3 area (NE)
                target_position=Vector2(21, 29),  # Approach to hoop 3
                stroke_type="roll",
                description="Roll from corner III to approach hoop 3"
            ),
        ]

    def assess_approach(
        self,
        ball: Ball,
        hoop,
        include_recommendation: bool = True
    ) -> ApproachQuality:
        """
        Assess the quality of a hoop approach based on Aiton's principles.

        Key insights from Section 2.3:
        - 1 yard in front is ideal
        - Stop-shot ratio of ~1:6 needed
        - Right side approaches are easier than left
        - 12 inches in front requires excellent control

        Args:
            ball: Ball attempting approach
            hoop: Target hoop
            include_recommendation: Whether to generate advice text

        Returns:
            ApproachQuality assessment
        """
        to_hoop = hoop.position - ball.position
        distance = to_hoop.magnitude()

        # Calculate approach angle relative to hoop direction
        if distance > 0.1:
            approach_dir = to_hoop.normalize()
            # Dot product with hoop direction tells us alignment
            alignment = approach_dir.dot(hoop.direction)
            approach_angle = math.acos(max(-1, min(1, alignment)))
        else:
            approach_angle = 0

        # Determine side (using cross product for left/right)
        # Perpendicular to hoop direction
        perp = Vector2(-hoop.direction.y, hoop.direction.x)
        side_dot = (ball.position - hoop.position).dot(perp)

        if abs(approach_angle) < self.STRAIGHT_ANGLE_THRESHOLD:
            side = ApproachSide.STRAIGHT
        elif side_dot > 0:
            side = ApproachSide.RIGHT  # Easier per Aiton
        else:
            side = ApproachSide.LEFT   # Harder per Aiton

        # Calculate success probability based on Aiton's teaching
        prob = self._calculate_approach_probability(distance, approach_angle, side)

        # Calculate ideal power (enough to go through + 2 yards)
        ideal_power = self._power_for_distance(distance + 2.0)

        # Generate recommendation
        recommendation = ""
        if include_recommendation:
            recommendation = self._get_approach_recommendation(
                distance, approach_angle, side, prob
            )

        return ApproachQuality(
            distance_to_hoop=distance,
            approach_angle=approach_angle,
            side=side,
            success_probability=prob,
            ideal_power=ideal_power,
            recommendation=recommendation
        )

    def _calculate_approach_probability(
        self,
        distance: float,
        angle: float,
        side: ApproachSide
    ) -> float:
        """
        Calculate success probability based on Aiton's principles.

        From Section 2.3:
        - 1 yard approach with stop-shot is "typically realistic"
        - 12 inches needs "excellent stop-shot control"
        - Angle significantly affects difficulty
        """
        # Distance factor - peaks at 1 yard (Aiton's ideal)
        if distance < 0.3:
            # Too close - hard to control
            dist_factor = 0.7
        elif distance <= self.IDEAL_APPROACH_DISTANCE:
            # Ideal zone
            dist_factor = 0.95
        elif distance <= self.MAX_GOOD_APPROACH:
            # Good zone - linear dropoff
            dist_factor = 0.95 - (distance - 1.0) * 0.1
        elif distance <= self.RISKY_APPROACH_DISTANCE:
            # Risky zone
            dist_factor = 0.75 - (distance - 3.0) * 0.1
        else:
            # Long range - difficult
            dist_factor = max(0.3, 0.55 - (distance - 7.0) * 0.05)

        # Angle factor
        if angle < self.STRAIGHT_ANGLE_THRESHOLD:
            angle_factor = 1.0
        elif angle < self.ACCEPTABLE_ANGLE:
            angle_factor = 0.85
        elif angle < self.DIFFICULT_ANGLE:
            angle_factor = 0.65
        else:
            angle_factor = 0.4

        # Side factor - Aiton emphasizes right side is easier
        if side == ApproachSide.STRAIGHT:
            side_factor = 1.0
        elif side == ApproachSide.RIGHT:
            side_factor = 0.95  # Easier
        else:  # LEFT
            side_factor = 0.85  # Harder

        return dist_factor * angle_factor * side_factor

    def _get_approach_recommendation(
        self,
        distance: float,
        angle: float,
        side: ApproachSide,
        probability: float
    ) -> str:
        """Generate Aiton-style recommendation for the approach."""
        if probability > 0.85:
            if distance <= 1.5:
                return "Excellent position - run the hoop with controlled stop-shot"
            else:
                return "Good position - approach with confidence"
        elif probability > 0.65:
            if angle > self.ACCEPTABLE_ANGLE:
                return "Angle is challenging - consider straightening up first"
            elif side == ApproachSide.LEFT:
                return "Left side approach - take extra care with alignment"
            else:
                return "Acceptable approach - moderate power recommended"
        elif probability > 0.45:
            if distance > 5:
                return "Long approach - consider positioning closer first"
            else:
                return "Difficult angle - may be better to reposition"
        else:
            return "Poor approach - strongly recommend repositioning"

    def get_leave_positions(
        self,
        striker_color: str,
        balls: Dict[str, Ball],
        leave_type: LeaveType = LeaveType.NSL
    ) -> LeavePosition:
        """
        Calculate target positions for a standard leave.

        From Section 2.5:
        - Diagonal Spread: Opponent balls separated diagonally
        - NSL: Partner at hoop 2, opponents in corners
        - MSL: Variation with partner placement

        Args:
            striker_color: Color of striker's ball
            balls: All balls on court
            leave_type: Which standard leave to set up

        Returns:
            LeavePosition with targets for all balls
        """
        # Determine partner and opponents
        partner_colors = {"blue": "black", "black": "blue",
                         "red": "yellow", "yellow": "red"}
        partner_color = partner_colors.get(striker_color)

        opponent_colors = [c for c in balls.keys()
                         if c != striker_color and c != partner_color]

        if leave_type == LeaveType.DIAGONAL_SPREAD:
            return self._diagonal_spread_leave(
                striker_color, partner_color, opponent_colors
            )
        elif leave_type == LeaveType.NSL:
            return self._nsl_leave(striker_color, partner_color, opponent_colors)
        elif leave_type == LeaveType.MSL:
            return self._msl_leave(striker_color, partner_color, opponent_colors)
        else:
            return self._defensive_leave(
                striker_color, partner_color, opponent_colors
            )

    def _diagonal_spread_leave(
        self,
        striker: str,
        partner: str,
        opponents: List[str]
    ) -> LeavePosition:
        """
        Diagonal Spread leave from Aiton Figure 2.10.

        Balls spread to corners to maximize opponent's difficulty
        in picking up a break.
        """
        # Striker near corner 4 (SE)
        striker_pos = Vector2(self.court.width - 3, 3)

        # Partner near corner 2 (NW) - diagonal from striker
        partner_pos = Vector2(3, self.court.height - 3)

        # Opponents at other corners
        opp1_pos = Vector2(3, 3)  # Corner 1 (SW)
        opp2_pos = Vector2(self.court.width - 3, self.court.height - 3)  # Corner 3 (NE)

        return LeavePosition(
            striker_pos=striker_pos,
            partner_pos=partner_pos,
            opponent1_pos=opp1_pos,
            opponent2_pos=opp2_pos,
            description="Diagonal Spread - balls separated for defensive leave"
        )

    def _nsl_leave(
        self,
        striker: str,
        partner: str,
        opponents: List[str]
    ) -> LeavePosition:
        """
        New Standard Leave (NSL) from Aiton Figure 2.11.

        - Partner positioned south and west of hoop 2
        - Opponents separated - one near west boundary
        - Striker positioned to shoot at partner next turn
        """
        # Partner: south and west of hoop 2 (good for pickup)
        # Hoop 2 is at (7, 28)
        partner_pos = Vector2(5, 25)

        # Striker: on east yard line, can shoot to partner
        striker_pos = Vector2(self.court.width - 1, 17)

        # Opponent 1: west upright of hoop 1 area
        opp1_pos = Vector2(1, 5)

        # Opponent 2: east boundary, north area
        opp2_pos = Vector2(self.court.width - 1, 30)

        return LeavePosition(
            striker_pos=striker_pos,
            partner_pos=partner_pos,
            opponent1_pos=opp1_pos,
            opponent2_pos=opp2_pos,
            description="NSL - Partner at hoop 2, opponents separated on boundaries"
        )

    def _msl_leave(
        self,
        striker: str,
        partner: str,
        opponents: List[str]
    ) -> LeavePosition:
        """
        Maugham Standard Leave (MSL) from Aiton.

        Similar to NSL but with variation in partner placement.
        Partner is placed just south and west of hoop 2.
        """
        # Partner: variation on NSL position
        partner_pos = Vector2(4, 26)

        # Striker: near peg, can join or shoot partner
        striker_pos = Vector2(16, 17)

        # Opponents further separated
        opp1_pos = Vector2(1, 3)
        opp2_pos = Vector2(self.court.width - 1, 32)

        return LeavePosition(
            striker_pos=striker_pos,
            partner_pos=partner_pos,
            opponent1_pos=opp1_pos,
            opponent2_pos=opp2_pos,
            description="MSL - Maugham Standard Leave variation"
        )

    def _defensive_leave(
        self,
        striker: str,
        partner: str,
        opponents: List[str]
    ) -> LeavePosition:
        """General defensive leave when standard leaves aren't applicable."""
        # Spread balls to make opponent pickup difficult
        striker_pos = Vector2(self.court.width / 2, 5)
        partner_pos = Vector2(self.court.width / 2, self.court.height - 5)
        opp1_pos = Vector2(3, self.court.height / 2)
        opp2_pos = Vector2(self.court.width - 3, self.court.height / 2)

        return LeavePosition(
            striker_pos=striker_pos,
            partner_pos=partner_pos,
            opponent1_pos=opp1_pos,
            opponent2_pos=opp2_pos,
            description="Defensive spread - minimize opponent break chances"
        )

    def get_ideal_pioneer_position(
        self,
        hoop,
        from_position: Vector2 = None
    ) -> Vector2:
        """
        Calculate ideal pioneer position for a hoop.

        From Aiton Section 2.4:
        - Pioneer should be 3-4 yards in front of hoop
        - Position depends on approach direction
        - Keep within inner rectangle bounded by corner hoops
        """
        # 4 yards in front of hoop (opposite to running direction)
        pioneer_pos = hoop.position - hoop.direction * self.IDEAL_PIONEER_DISTANCE

        # Clamp to inner rectangle (7-21 x, 7-28 y)
        pioneer_pos.x = max(7, min(21, pioneer_pos.x))
        pioneer_pos.y = max(7, min(28, pioneer_pos.y))

        return pioneer_pos

    def get_ideal_reception_position(
        self,
        hoop,
        rush_direction: Vector2 = None
    ) -> Vector2:
        """
        Calculate ideal reception (pilot) ball position.

        From Aiton: Reception ball determines approach quality.
        Should be positioned to allow rush toward hoop after roquet.

        Args:
            hoop: Target hoop for approach
            rush_direction: Optional direction to set up rush toward

        Returns:
            Ideal position for reception ball
        """
        # 2-3 yards in front of hoop for approach
        base_pos = hoop.position - hoop.direction * 2.5

        # If rush direction specified, offset slightly to allow rush
        if rush_direction and rush_direction.magnitude() > 0.1:
            # Position so that after roquet, can rush toward hoop
            perp = Vector2(-rush_direction.y, rush_direction.x)
            base_pos = base_pos + perp * 0.5

        return base_pos

    def evaluate_break_pickup(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        deadness: Dict[str, set]
    ) -> Tuple[float, str]:
        """
        Evaluate the quality of a potential break pickup.

        From Aiton Section 2.6:
        - 4th turn pickup is critical
        - Need pioneer positioning and rush availability
        - Consider controlled break vs defensive options

        Args:
            striker: Ball attempting pickup
            balls: All balls
            deadness: Which balls striker is dead on

        Returns:
            Tuple of (quality 0-1, description)
        """
        dead_on = deadness.get(striker.color, set())
        live_balls = [c for c in balls.keys()
                     if c != striker.color and c not in dead_on]

        if not live_balls:
            return (0.0, "No live balls - cannot pick up break")

        # Get target hoop
        target_hoop = self.court.get_hoop_for_ball(striker.hoops_run)
        if not target_hoop:
            return (0.3, "Rover ball - aim for peg")

        # Evaluate each live ball as potential pilot
        best_pilot_score = 0
        best_pilot = None

        for color in live_balls:
            ball = balls[color]
            # Score as pilot (reception ball)
            pilot_score = self._score_pilot_quality(ball, striker, target_hoop)
            if pilot_score > best_pilot_score:
                best_pilot_score = pilot_score
                best_pilot = color

        # Check for pioneer at next hoop
        next_hoop = self.court.get_hoop_for_ball(striker.hoops_run + 1)
        pioneer_score = 0
        if next_hoop:
            for color in live_balls:
                if color != best_pilot:
                    ball = balls[color]
                    p_score = self._score_pioneer_quality(ball, next_hoop)
                    pioneer_score = max(pioneer_score, p_score)

        # Combined quality
        if len(live_balls) >= 3:
            # 4-ball break potential
            quality = best_pilot_score * 0.5 + pioneer_score * 0.3 + 0.2
            desc = f"4-ball break possible with {best_pilot} as pilot"
        elif len(live_balls) >= 2:
            # 3-ball break potential
            quality = best_pilot_score * 0.6 + pioneer_score * 0.3 + 0.1
            desc = f"3-ball break setup with {best_pilot}"
        else:
            # 2-ball only
            quality = best_pilot_score * 0.8
            desc = f"2-ball break only - difficult"

        return (min(1.0, quality), desc)

    def _score_pilot_quality(
        self,
        pilot: Ball,
        striker: Ball,
        hoop
    ) -> float:
        """Score how good a ball is as pilot for approach."""
        to_hoop = hoop.position - pilot.position
        dist_to_hoop = to_hoop.magnitude()

        # Ideal: 2-4 yards from hoop, in front
        if dist_to_hoop < 1 or dist_to_hoop > 10:
            dist_score = 0.3
        elif 2 <= dist_to_hoop <= 4:
            dist_score = 1.0
        else:
            dist_score = 0.7

        # Check if in front of hoop
        if dist_to_hoop > 0.5:
            approach_dot = to_hoop.normalize().dot(hoop.direction)
            if approach_dot < 0:
                return 0.2  # Wrong side

        # Can striker reach this ball?
        striker_dist = (pilot.position - striker.position).magnitude()
        reach_score = max(0.3, 1.0 - striker_dist / 20)

        return dist_score * 0.6 + reach_score * 0.4

    def _score_pioneer_quality(self, pioneer: Ball, hoop) -> float:
        """Score how good a ball is as pioneer."""
        to_hoop = hoop.position - pioneer.position
        dist = to_hoop.magnitude()

        # Ideal: 3-5 yards from hoop
        if 3 <= dist <= 5:
            dist_score = 1.0
        elif 2 <= dist <= 7:
            dist_score = 0.7
        else:
            dist_score = 0.4

        # Check if in front
        if dist > 0.5:
            approach_dot = to_hoop.normalize().dot(hoop.direction)
            if approach_dot < 0:
                return 0.2

        return dist_score

    def find_standard_stroke(
        self,
        ball: Ball
    ) -> Optional[StandardStroke]:
        """
        Check if ball is in position for a standard stroke pattern.

        From Aiton Section 2.7 - certain positions have well-known
        optimal strokes that should be recognized and executed.
        """
        for stroke in self.standard_strokes:
            x1, y1, x2, y2 = stroke.start_zone
            if (x1 <= ball.position.x <= x2 and
                y1 <= ball.position.y <= y2):
                return stroke
        return None

    def get_stop_shot_ratio(self, approach_type: str = "standard") -> float:
        """
        Get the appropriate stop-shot ratio based on Aiton's teaching.

        From Section 2.3:
        - Standard approach: ~1:6 ratio
        - Close approach (12 inches): requires tighter control
        """
        if approach_type == "close":
            return 0.12  # 1:8 ratio for close work
        elif approach_type == "long":
            return 0.20  # 1:5 ratio for longer approaches
        else:
            return 0.167  # 1:6 ratio - Aiton's standard

    def _power_for_distance(self, distance: float) -> float:
        """Calculate power needed for distance (with friction)."""
        import config
        friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY
        return math.sqrt(2 * friction_decel * distance) * 1.1
