"""
Expert Tactics - Advanced tactical patterns from expert match play.

Based on analysis of the Aiton-Maugham Eastern Championships 2007 game,
with commentary revealing expert-level decision making:

LIFT MANAGEMENT:
- After running 1-back or 4-back, opponent gets a lift
- Positioning must account for lift options (corners I, III, or baulk lines)
- Maximum length positions minimize lift effectiveness

POSITION VS SHOOTING:
- "What do you do after you've run this?" - always think ahead
- Wired positions can force opponent into risky shots
- Guarding sides of lawn affects opponent's lift options

TPO (TRIPLE PEEL OUT):
- High-risk tactic to reduce to 2-ball game
- Used when opponent is shooting poorly
- Changes game dynamics significantly

IMPASSE TACTICS:
- Near-hoop standoffs where neither player wants to shoot first
- Moving incrementally closer to hoop
- Forcing opponent to take risky shot or concede position

STROKE QUALITY:
- "Thrashing" vs "smooth" strokes affects success
- Rushed decisions lead to poor execution
- Confidence and rhythm affect shooting percentage
"""
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto

from models.ball import Ball, Vector2
from models.court import Court


class TacticalSituation(Enum):
    """Types of tactical situations from expert play."""
    LIFT_PENDING = auto()       # Opponent will get lift after this hoop
    IMPASSE = auto()            # Near-hoop standoff
    TPO_OPPORTUNITY = auto()    # Triple peel out possible
    GUARDING = auto()           # Protecting a side of lawn
    WIRED_POSITION = auto()     # Taking wired position
    MAXIMUM_LENGTH = auto()     # Maximum distance leave


@dataclass
class LiftConsideration:
    """Information about lift implications."""
    lift_available: bool        # Will opponent get lift?
    lift_corners: List[str]     # Which corners available (I, III)
    baulk_available: bool       # Can lift from baulk?
    recommended_leave: str      # Where to position for best leave
    danger_zones: List[Vector2] # Areas to avoid


@dataclass
class ImpasseAnalysis:
    """Analysis of an impasse situation."""
    is_impasse: bool
    hoop_contested: int         # Which hoop
    distance_advantage: float   # Who is closer (positive = striker)
    shoot_recommendation: bool  # Should striker shoot?
    reasoning: str


@dataclass
class PositionVsShootDecision:
    """Decision analysis for position vs shooting."""
    should_shoot: bool
    shoot_confidence: float     # 0-1 confidence in making shot
    position_value: float       # 0-1 value of taking position
    next_turn_consideration: str # What happens after?
    reasoning: str


class ExpertTactics:
    """
    Implements expert-level tactical decision making.

    Based on Aiton-Maugham match commentary showing:
    - Lift management and leave positioning
    - Position vs shooting decisions
    - Impasse recognition and handling
    - TPO tactical considerations
    """

    # Hoops that grant lifts when run (1-back = hoop 7, 4-back = hoop 10)
    LIFT_HOOPS = {7, 10}  # 1-back and 4-back

    # Lift corners
    CORNER_I = Vector2(1, 1)      # SW corner (baulk A area)
    CORNER_III = Vector2(27, 34)  # NE corner (baulk B area)

    def __init__(self, court: Court):
        """Initialize with court reference."""
        self.court = court

    def analyze_lift_situation(
        self,
        striker: Ball,
        opponent_balls: List[Ball]
    ) -> LiftConsideration:
        """
        Analyze lift implications for current position.

        From Aiton-Maugham game:
        - Turn 14: "mindful of the lift, goes to E boundary, maximum length position"
        - Turn 29: "guard the W side of the lawn in view of the impending lift"

        Args:
            striker: Striker's ball
            opponent_balls: Opponent's balls

        Returns:
            LiftConsideration with recommendations
        """
        # Check if striker just ran or is about to run a lift hoop
        next_hoop = striker.hoops_run + 1
        lift_pending = next_hoop in self.LIFT_HOOPS

        if not lift_pending:
            return LiftConsideration(
                lift_available=False,
                lift_corners=[],
                baulk_available=False,
                recommended_leave="standard",
                danger_zones=[]
            )

        # Lifts available from corners I and III
        lift_corners = ["I", "III"]

        # Baulk lines available for lift
        baulk_available = True

        # Calculate danger zones - areas opponent can easily reach from lift
        danger_zones = []

        # From corner I (SW), opponent threatens SW quadrant
        danger_zones.append(Vector2(7, 7))   # Near hoop 1
        danger_zones.append(Vector2(7, 14))  # W side mid-court

        # From corner III (NE), opponent threatens NE quadrant
        danger_zones.append(Vector2(21, 28))  # Near hoop 3
        danger_zones.append(Vector2(21, 21))  # E side mid-court

        # Recommended leave: maximum length, avoid danger zones
        # "maximum length position" - furthest from lift corners
        if striker.hoops_run + 1 == 7:  # About to run 1-back
            recommended_leave = "E boundary south of III, maximum length from I"
        else:  # About to run 4-back
            recommended_leave = "W boundary north of I, maximum length from III"

        return LiftConsideration(
            lift_available=True,
            lift_corners=lift_corners,
            baulk_available=baulk_available,
            recommended_leave=recommended_leave,
            danger_zones=danger_zones
        )

    def analyze_impasse(
        self,
        striker: Ball,
        opponent: Ball,
        target_hoop
    ) -> ImpasseAnalysis:
        """
        Analyze if current situation is an impasse.

        From Aiton-Maugham game (turns 45-49):
        - Both players near 4-back, neither wanting to shoot first
        - Dave: "Hopeful of an impasse..."
        - Keith: "...or a 9 yarder"

        An impasse occurs when:
        - Both balls near same hoop
        - Neither has clear advantage
        - Shooting risks giving opponent easy position

        Args:
            striker: Striker's ball
            opponent: Opponent's ball
            target_hoop: Hoop both are contesting

        Returns:
            ImpasseAnalysis
        """
        if not target_hoop:
            return ImpasseAnalysis(
                is_impasse=False,
                hoop_contested=0,
                distance_advantage=0,
                shoot_recommendation=True,
                reasoning="No hoop contested"
            )

        striker_dist = (target_hoop.position - striker.position).magnitude()
        opponent_dist = (target_hoop.position - opponent.position).magnitude()

        # Impasse conditions:
        # 1. Both within 10 yards of hoop
        # 2. Neither has overwhelming advantage (within 3 yards of each other)
        # 3. Both are on approach side of hoop

        both_close = striker_dist < 10 and opponent_dist < 10
        similar_distance = abs(striker_dist - opponent_dist) < 3

        # Check if both on approach side
        striker_approach = (target_hoop.position - striker.position).dot(target_hoop.direction) > 0
        opponent_approach = (target_hoop.position - opponent.position).dot(target_hoop.direction) > 0
        both_approaching = striker_approach and opponent_approach

        is_impasse = both_close and similar_distance and both_approaching

        distance_advantage = opponent_dist - striker_dist  # Positive = striker closer

        # Shooting recommendation
        if is_impasse:
            if distance_advantage > 1.5:
                # Striker has position advantage - can afford to wait
                shoot_recommendation = False
                reasoning = "Positional advantage - let opponent shoot first"
            elif distance_advantage < -1.5:
                # Opponent has advantage - need to shoot
                shoot_recommendation = True
                reasoning = "Opponent closer - must contest"
            else:
                # True impasse - consider incremental advance
                shoot_recommendation = False
                reasoning = "True impasse - advance incrementally toward hoop"
        else:
            shoot_recommendation = True
            reasoning = "Not an impasse - normal play"

        return ImpasseAnalysis(
            is_impasse=is_impasse,
            hoop_contested=target_hoop.number if hasattr(target_hoop, 'number') else 0,
            distance_advantage=distance_advantage,
            shoot_recommendation=shoot_recommendation,
            reasoning=reasoning
        )

    def position_vs_shoot_decision(
        self,
        striker: Ball,
        target_ball: Ball,
        shot_distance: float,
        can_take_wired_position: bool,
        target_hoop=None,
        opponent_shooting_form: float = 0.5
    ) -> PositionVsShootDecision:
        """
        Decide whether to shoot or take position.

        From Aiton-Maugham game:
        - Turn 23: "take position wired from Y on the basis that Y won't
                   risk shot from B-baulk"
        - Turn 29: "U decided not to shoot this time and goes to W boundary"
        - Turn 38: "the little voice in my head said 'What do you do after
                   you've run this then eh?'"

        Key insight: Always consider what happens AFTER the shot.

        Args:
            striker: Striker's ball
            target_ball: Ball to potentially shoot at
            shot_distance: Distance to target
            can_take_wired_position: Is wired position available?
            target_hoop: Hoop striker is aiming for
            opponent_shooting_form: How well opponent is shooting (0-1)

        Returns:
            PositionVsShootDecision
        """
        # Calculate shot probability based on distance
        # From commentary: 10 yard shots are risky, 4 yard should hit
        if shot_distance < 5:
            shot_confidence = 0.85
        elif shot_distance < 8:
            shot_confidence = 0.65
        elif shot_distance < 12:
            shot_confidence = 0.45
        else:
            shot_confidence = 0.30

        # Position value
        position_value = 0.5

        if can_take_wired_position:
            # Wired position is valuable
            position_value = 0.75
            # Even more valuable if opponent shooting poorly
            if opponent_shooting_form < 0.4:
                position_value = 0.85

        # Consider what happens after
        next_turn_consideration = ""

        if target_hoop:
            # Can we run the hoop after hitting?
            hoop_dist = (target_hoop.position - striker.position).magnitude()
            if hoop_dist > 10:
                next_turn_consideration = "Long way to hoop after roquet"
                shot_confidence *= 0.9  # Slight penalty
            else:
                next_turn_consideration = "Good position for break after roquet"

        # Decision logic
        # From Maugham turn 38: think about consequences
        if shot_confidence > 0.7:
            should_shoot = True
            reasoning = "High confidence shot - take it"
        elif can_take_wired_position and opponent_shooting_form < 0.5:
            should_shoot = False
            reasoning = "Wired position available and opponent shooting poorly"
        elif shot_confidence > position_value:
            should_shoot = True
            reasoning = "Shot probability exceeds position value"
        else:
            should_shoot = False
            reasoning = "Position more valuable than risky shot"

        return PositionVsShootDecision(
            should_shoot=should_shoot,
            shoot_confidence=shot_confidence,
            position_value=position_value,
            next_turn_consideration=next_turn_consideration,
            reasoning=reasoning
        )

    def calculate_maximum_length_position(
        self,
        avoid_corners: List[str]
    ) -> Vector2:
        """
        Calculate maximum length position avoiding specified corners.

        From turn 14: "goes to E boundary, maximum length position"

        Maximum length = furthest from opponent's lift options.

        Args:
            avoid_corners: Corners opponent will lift from ("I", "III")

        Returns:
            Best position for maximum length leave
        """
        # Court boundaries
        yard_line = 1.0

        # Possible positions along yard lines
        positions = []

        # E boundary positions
        for y in range(7, 29, 2):
            positions.append(Vector2(self.court.width - yard_line, y))

        # W boundary positions
        for y in range(7, 29, 2):
            positions.append(Vector2(yard_line, y))

        # N boundary positions
        for x in range(7, 22, 2):
            positions.append(Vector2(x, self.court.height - yard_line))

        # S boundary positions
        for x in range(7, 22, 2):
            positions.append(Vector2(x, yard_line))

        # Calculate minimum distance to lift corners for each position
        best_pos = positions[0]
        best_min_dist = 0

        for pos in positions:
            min_dist = float('inf')

            if "I" in avoid_corners:
                dist_to_I = (pos - self.CORNER_I).magnitude()
                min_dist = min(min_dist, dist_to_I)

            if "III" in avoid_corners:
                dist_to_III = (pos - self.CORNER_III).magnitude()
                min_dist = min(min_dist, dist_to_III)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = pos

        return best_pos

    def evaluate_tpo_opportunity(
        self,
        striker: Ball,
        partner: Ball,
        opponent_balls: List[Ball],
        opponent_shooting_form: float
    ) -> Tuple[bool, str]:
        """
        Evaluate whether TPO (Triple Peel Out) is advisable.

        From turn 8 commentary:
        "I left 3 balls on in the Southerns final, and Keith finished
        the turn after contact. I thought I'd try something different."

        TPO considerations:
        - Reduces game to 2-ball (simpler but riskier)
        - Good if opponent shooting poorly
        - Bad if opponent likely to establish break

        Args:
            striker: Striker's ball
            partner: Partner's ball (to be pegged out)
            opponent_balls: Opponent's balls
            opponent_shooting_form: How well opponent is shooting

        Returns:
            Tuple of (should_attempt_tpo, reasoning)
        """
        # TPO is risky - only consider in specific situations

        # Factors favoring TPO:
        # 1. Opponent shooting poorly
        # 2. Striker has good position
        # 3. Partner ball is advanced (near rover)

        partner_advanced = partner.hoops_run >= 10  # Near rover
        opponent_struggling = opponent_shooting_form < 0.35

        if partner_advanced and opponent_struggling:
            return (True, "Partner advanced, opponent struggling - TPO viable")

        if opponent_struggling and striker.hoops_run >= 6:
            return (True, "Opponent shooting <35% - consider TPO to simplify")

        # Factors against TPO:
        # 1. Opponent shooting well - will establish break
        # 2. Partner not advanced - wasting pegging out

        if opponent_shooting_form > 0.6:
            return (False, "Opponent shooting well - TPO too risky")

        if partner.hoops_run < 6:
            return (False, "Partner not advanced enough for TPO")

        return (False, "Standard play recommended")

    def get_guarding_position(
        self,
        side_to_guard: str,
        own_hoop,
        opponent_position: Vector2
    ) -> Vector2:
        """
        Calculate position to guard a side of lawn.

        From turn 29: "guard the W side of the lawn in view of
        the impending lift"

        Args:
            side_to_guard: "W", "E", "N", or "S"
            own_hoop: Hoop striker is approaching
            opponent_position: Where opponent is

        Returns:
            Guarding position
        """
        yard_line = 1.0

        if side_to_guard == "W":
            # Guard west boundary
            # Position between opponent and west side
            y = min(max(opponent_position.y, 7), 28)
            return Vector2(yard_line, y)

        elif side_to_guard == "E":
            y = min(max(opponent_position.y, 7), 28)
            return Vector2(self.court.width - yard_line, y)

        elif side_to_guard == "N":
            x = min(max(opponent_position.x, 7), 21)
            return Vector2(x, self.court.height - yard_line)

        else:  # "S"
            x = min(max(opponent_position.x, 7), 21)
            return Vector2(x, yard_line)

    def assess_stroke_quality_effect(
        self,
        is_rushed: bool,
        is_pressured: bool,
        recent_misses: int
    ) -> float:
        """
        Assess how stroke quality affects success probability.

        From commentary:
        - "wild thrash" vs running hoop "smoothly"
        - Turn 42/54: "Another thrash" - rushed strokes fail
        - Confidence degrades with misses (Keith: "0 hits and 10 misses")

        Args:
            is_rushed: Is the player rushing the shot?
            is_pressured: Is there time/tactical pressure?
            recent_misses: How many recent misses?

        Returns:
            Multiplier for success probability (0.5 - 1.0)
        """
        quality_factor = 1.0

        # Rushed strokes are worse
        if is_rushed:
            quality_factor *= 0.75  # "thrash" effect

        # Pressure affects quality
        if is_pressured:
            quality_factor *= 0.9

        # Confidence degrades with misses
        # Keith went 0-10 at one point
        if recent_misses > 0:
            confidence_penalty = min(0.25, recent_misses * 0.03)
            quality_factor *= (1.0 - confidence_penalty)

        return max(0.5, quality_factor)
