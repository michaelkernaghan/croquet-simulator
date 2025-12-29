"""
Tactical Rules - Expert knowledge encoded as rules.

Enhanced with knowledge from:
- AC Laws 7th Edition (via ac-laws-summary.txt)
- Keith Wylie's "Expert Croquet Tactics" (Articles 1-4)

These rules capture croquet strategy knowledge including:
- The four playing styles (Aggressive, Precision, Canny, Monte Carlo)
- Break building with pioneers, pivots, and escape balls
- Lift and contact rules for Advanced Play
- Wiring detection and exploitation
- Leave selection (NSL, defensive 4-back leaves)
- Peeling strategy
- Proper rover/peg-out constraints
"""
import random
import math
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum, auto
from dataclasses import dataclass, field

from models.ball import Ball, Vector2
from models.court import Court


class ShotType(Enum):
    """Types of shots the AI can make."""
    HOOP_RUN = auto()       # Attempt to run the hoop
    HOOP_APPROACH = auto()  # Position in front of hoop
    ROQUET = auto()         # Hit another ball
    RUSH = auto()           # Roquet + send ball somewhere useful
    TAKE_OFF = auto()       # Croquet where you go far, other ball stays
    DRIVE = auto()          # Croquet where both balls go same direction
    ROLL = auto()           # Croquet where both balls go similar distance
    SPLIT = auto()          # Croquet where balls go different directions
    STOP_SHOT = auto()      # Croquet where croqueted ball goes far, you stay
    CLEARANCE = auto()      # Hit opponent ball away
    DEFENSIVE = auto()      # Safe position shot
    PEEL = auto()           # Send partner ball through its hoop
    PIONEER_PLACEMENT = auto()  # Place ball at future hoop
    CANNON = auto()         # Three-ball cannon shot
    WIRE = auto()           # Position to wire balls


class PlayStyle(Enum):
    """
    The four playing styles from Wylie's "Expert Croquet Tactics".

    A balanced player uses all three positive styles appropriately,
    avoiding Monte Carlo (reckless) play.
    """
    AGGRESSIVE = auto()   # Bold, daring strokes - "blood"
    PRECISION = auto()    # Meticulous accuracy - "phlegm"
    CANNY = auto()        # Defensive, percentage play - "melancholy"
    MONTE_CARLO = auto()  # Reckless gambling - "choler" (to be avoided!)


class BreakRole(Enum):
    """Roles of balls in a 4-ball break."""
    STRIKER = auto()      # The ball being played
    PIONEER = auto()      # Ball placed near next hoop
    PIVOT = auto()         # Ball kept near middle of court
    ESCAPE = auto()       # Ball used after running hoop to get to pioneer
    OPPONENT = auto()     # Opponent's ball (can still be used in break)


@dataclass
class TacticalAdvice:
    """Advice from the tactical rules."""
    recommended_shot: ShotType
    target_position: Optional[Vector2]
    target_ball: Optional[str]
    priority: float  # 0-1, how important is this advice
    reason: str
    style: PlayStyle = PlayStyle.PRECISION  # Which style this advice represents
    risk_level: float = 0.5  # 0=safe, 1=risky


@dataclass
class BreakState:
    """Tracks the state of the current break."""
    pioneer_ball: Optional[str] = None      # Ball at next hoop
    pivot_ball: Optional[str] = None        # Ball in center
    escape_ball: Optional[str] = None       # Ball for getting to pioneer
    hoops_in_break: int = 0                 # Hoops run this break
    is_four_ball_break: bool = False
    is_three_ball_break: bool = False
    is_two_ball_break: bool = False


@dataclass
class LeaveType:
    """Types of leaves from Wylie."""
    name: str
    description: str
    positions: Dict[str, Vector2]  # Ball color -> target position
    gives_lift: bool = False       # Does this leave give opponent a lift?


class TacticalRules:
    """
    Expert tactical rules for croquet.

    Enhanced with knowledge from AC Laws and Wylie's Expert Croquet Tactics.

    Key concepts implemented:
    - Four playing styles with appropriate situations for each
    - Break building with proper pioneer/pivot/escape ball management
    - Advanced Play rules (lifts at 1-back and 4-back)
    - Wiring detection and strategic use
    - Leave selection based on game state
    - Peeling strategy for partner ball
    - Proper rover constraints (can't peg out partner if leaving rover vs non-rover)
    """

    # Hoop names for reference
    HOOP_NAMES = {
        0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6",
        6: "1-back", 7: "2-back", 8: "3-back", 9: "4-back",
        10: "penult", 11: "rover"
    }

    # Critical hoops for Advanced Play (give opponent lift)
    LIFT_HOOPS = {6: "1-back", 9: "4-back"}  # hoops_run value -> name

    def __init__(self, skill_level: float = 0.7, aggression: float = 0.5):
        """
        Initialize tactical rules.

        Args:
            skill_level: 0-1, affects which styles are viable
            aggression: 0-1, bias toward aggressive vs canny play
        """
        self.skill_level = skill_level
        self.aggression = aggression
        self.rules_applied = []  # Track which rules fired (for learning)
        self.break_state = BreakState()

    def get_style_weights(self) -> Dict[PlayStyle, float]:
        """
        Get appropriate weights for each playing style based on skill and aggression.

        From Wylie: "A successful style resembles a well-balanced temperament"
        combining aggressive, precision, and canny croquet appropriately.
        """
        # Higher skill enables more aggressive and precision play
        # Lower skill should favor canny (defensive) play
        aggressive_weight = self.skill_level * self.aggression
        precision_weight = self.skill_level * (1 - self.aggression * 0.3)
        canny_weight = (1 - self.skill_level * 0.5) * (1 - self.aggression * 0.5)

        # Monte Carlo should always be low - it's bad play!
        monte_carlo_weight = max(0, 0.1 - self.skill_level * 0.1)

        return {
            PlayStyle.AGGRESSIVE: aggressive_weight,
            PlayStyle.PRECISION: precision_weight,
            PlayStyle.CANNY: canny_weight,
            PlayStyle.MONTE_CARLO: monte_carlo_weight
        }

    def get_advice(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int = 1,
        is_continuation: bool = False,
        is_croquet_stroke: bool = False
    ) -> List[TacticalAdvice]:
        """
        Get tactical advice for the current situation.

        Enhanced with Wylie's strategic principles:
        - Running hoops is the objective
        - Roquets are means to an end, not the goal
        - Build breaks using pioneers and pivot balls
        - Consider defensive leaves when break isn't viable

        Args:
            striker: The ball about to shoot
            all_balls: All balls on the court
            court: The court
            deadness: Which balls striker is dead on
            strokes_remaining: Strokes left in the turn
            is_continuation: Whether this is a continuation stroke
            is_croquet_stroke: Whether this is a croquet stroke

        Returns:
            List of TacticalAdvice sorted by priority
        """
        advice = []
        self.rules_applied = []

        # Get context
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)
        live_balls = self._get_live_balls(striker.color, deadness)
        partner = self._get_partner_ball(striker.color, all_balls)
        opponents = self._get_opponent_balls(striker.color, all_balls)
        style_weights = self.get_style_weights()

        # Update break state
        self._update_break_state(striker, all_balls, target_hoop, live_balls)

        # RULE 0: ROVER - PEG OUT (highest priority)
        if striker.is_rover and not striker.has_pegged_out:
            peg_advice = self._check_peg_out_opportunity(striker, partner, all_balls, court)
            if peg_advice:
                advice.append(peg_advice)
                self.rules_applied.append("peg_out_attempt")
                # Check if we should actually peg out (rover constraint)
                if self._should_peg_out(striker, partner, all_balls):
                    return advice  # Pegging out is the ultimate goal

        # RULE 1: HOOP RUN - Primary objective!
        if target_hoop:
            hoop_advice = self._check_hoop_run_opportunity(
                striker, target_hoop, is_continuation, style_weights
            )
            if hoop_advice:
                advice.append(hoop_advice)
                self.rules_applied.append("hoop_run_opportunity")

        # RULE 2: PEELING - Help partner ball through its hoop
        if partner and not partner.has_pegged_out:
            peel_advice = self._check_peel_opportunity(
                striker, partner, all_balls, court, live_balls
            )
            if peel_advice:
                advice.append(peel_advice)
                self.rules_applied.append("peel_opportunity")

        # RULE 3: RUSH TO HOOP - Best way to build a break
        if target_hoop and live_balls:
            rush_advice = self._check_rush_opportunity(
                striker, all_balls, target_hoop, live_balls, style_weights
            )
            if rush_advice:
                advice.append(rush_advice)
                self.rules_applied.append("rush_opportunity")

        # RULE 4: PIONEER PLACEMENT - Set up for future hoops
        if target_hoop and live_balls:
            pioneer_advice = self._check_pioneer_placement(
                striker, all_balls, court, target_hoop, live_balls
            )
            if pioneer_advice:
                advice.append(pioneer_advice)
                self.rules_applied.append("pioneer_placement")

        # RULE 5: HOOP APPROACH - Position for hoop run
        # NOTE: This ends the turn if no roquet made - should be low priority!
        if target_hoop:
            approach_advice = self._check_hoop_approach(
                striker, target_hoop, court, is_continuation, style_weights,
                has_live_balls=len(live_balls) > 0  # Tell it if roquets available
            )
            if approach_advice:
                advice.append(approach_advice)
                self.rules_applied.append("hoop_approach")

        # RULE 6: ROQUET FOR STROKES - High priority when NOT in hoop position!
        if live_balls:
            roquet_advice = self._check_any_roquet(
                striker, all_balls, live_balls, is_continuation, style_weights,
                hoop=target_hoop  # Pass hoop to check if in position
            )
            if roquet_advice:
                advice.append(roquet_advice)
                self.rules_applied.append("any_roquet")

        # RULE 7: WIRING - Strategic use of hoops/peg to block opponent
        if opponents:
            wire_advice = self._check_wiring_opportunity(
                striker, opponents, court, target_hoop
            )
            if wire_advice:
                advice.append(wire_advice)
                self.rules_applied.append("wiring_opportunity")

        # RULE 8: CLEARANCE - Opponent threatens their hoop
        if opponents:
            clearance_advice = self._check_clearance_needed(
                striker, opponents, court, deadness
            )
            if clearance_advice:
                advice.append(clearance_advice)
                self.rules_applied.append("clearance_needed")

        # RULE 9: DEFENSIVE LEAVE - When break not viable
        if not advice or all(a.priority < 0.5 for a in advice):
            leave_advice = self._check_defensive_leave(
                striker, all_balls, court, target_hoop, live_balls, style_weights
            )
            if leave_advice:
                advice.append(leave_advice)
                self.rules_applied.append("defensive_leave")

        # RULE 10: RUSH SETUP - Position for future rush
        if target_hoop and live_balls and not advice:
            setup_advice = self._check_rush_setup(
                striker, all_balls, target_hoop, live_balls, court
            )
            if setup_advice:
                advice.append(setup_advice)
                self.rules_applied.append("rush_setup")

        # Apply style-based priority adjustments
        advice = self._apply_style_adjustments(advice, style_weights, is_continuation)

        # Sort by priority with small randomness for variety
        for a in advice:
            a.priority += random.uniform(-0.05, 0.05)

        advice.sort(key=lambda a: a.priority, reverse=True)

        return advice

    def _update_break_state(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        target_hoop,
        live_balls: List[str]
    ):
        """Update the break state based on current positions."""
        if not target_hoop:
            return

        # Count balls in useful positions
        balls_in_play = 0
        for color in live_balls:
            ball = all_balls.get(color)
            if ball and not ball.has_pegged_out:
                balls_in_play += 1

        # Determine break type
        self.break_state.is_four_ball_break = balls_in_play >= 3
        self.break_state.is_three_ball_break = balls_in_play == 2
        self.break_state.is_two_ball_break = balls_in_play == 1

        # TODO: Track specific pioneer/pivot/escape ball assignments

    def _check_peg_out_opportunity(
        self,
        striker: Ball,
        partner: Optional[Ball],
        all_balls: Dict[str, Ball],
        court: Court
    ) -> Optional[TacticalAdvice]:
        """
        Check if rover should attempt to peg out.

        From AC Laws: A rover can be pegged out by any ball.
        Constraint: Cannot peg out partner if it would leave opponent
        with a rover vs non-rover advantage.
        """
        if not striker.is_rover:
            return None

        to_peg = court.peg_position - striker.position
        distance = to_peg.magnitude()

        # Check the rover constraint
        if not self._should_peg_out(striker, partner, all_balls):
            return None

        # Priority based on distance - closer is better but always try
        distance_factor = max(0.1, 1.0 - distance / 30)
        priority = 0.95 + distance_factor * 0.04  # 0.95-0.99

        return TacticalAdvice(
            recommended_shot=ShotType.HOOP_RUN,  # Reuse for peg attempts
            target_position=court.peg_position,
            target_ball=None,
            priority=priority,
            reason=f"Peg out! (dist={distance:.1f})",
            style=PlayStyle.AGGRESSIVE,
            risk_level=0.3
        )

    def _should_peg_out(
        self,
        striker: Ball,
        partner: Optional[Ball],
        all_balls: Dict[str, Ball]
    ) -> bool:
        """
        Determine if pegging out is strategically sound.

        From AC Laws: Cannot peg out partner ball if it would leave
        opponent with a rover vs non-rover situation.

        More nuanced: Even for own ball, consider game state.
        """
        if not partner:
            return True  # No partner, always peg out

        # If partner is also a rover, definitely peg out
        if partner.is_rover:
            return True

        # If partner is close to being a rover (say, at penult), ok to peg out
        if partner.hoops_run >= 10:  # At penult or rover
            return True

        # Check opponent state
        opponents = self._get_opponent_balls(striker.color, all_balls)
        opponent_rovers = sum(1 for opp in opponents if opp.is_rover)
        opponent_advanced = sum(1 for opp in opponents if opp.hoops_run >= 6)

        # If opponents have rovers and partner is behind, don't peg out yet
        if opponent_rovers > 0 and partner.hoops_run < 6:
            return False

        # Generally ok to peg out if we're ahead
        return True

    def _check_hoop_run_opportunity(
        self,
        striker: Ball,
        hoop,
        is_continuation: bool,
        style_weights: Dict[PlayStyle, float]
    ) -> Optional[TacticalAdvice]:
        """
        Check if we should attempt to run the hoop.

        From Wylie: Running hoops is THE objective. This should be
        highest priority when in good position.
        """
        to_hoop = hoop.position - striker.position
        distance = to_hoop.magnitude()

        if distance > 12:
            return None  # Too far for direct run

        if distance < 0.3:
            return None  # Too close (in the hoop)

        # Check approach angle - must be from correct side
        approach_dir = to_hoop.normalize()
        dot = approach_dir.dot(hoop.direction)

        if dot < 0.15:
            return None  # Wrong side or very bad angle

        # Calculate priority based on position quality
        angle_quality = max(0, (dot - 0.15) / 0.85)  # 0-1
        distance_quality = max(0, 1.0 - distance / 12)  # 0-1

        # Determine style and risk
        if dot > 0.7 and distance < 4:
            # Excellent position - precision shot
            priority = 0.98
            style = PlayStyle.PRECISION
            risk = 0.1
        elif dot > 0.5 and distance < 6:
            # Good position
            priority = 0.94
            style = PlayStyle.PRECISION
            risk = 0.2
        elif dot > 0.4 and distance < 5:
            # Decent position
            priority = 0.90
            style = PlayStyle.AGGRESSIVE if self.aggression > 0.5 else PlayStyle.PRECISION
            risk = 0.3
        elif distance < 8 and dot > 0.3:
            # Marginal - aggressive style needed
            priority = 0.80 + angle_quality * 0.1
            style = PlayStyle.AGGRESSIVE
            risk = 0.5
        else:
            # Difficult - consider if worth the risk
            priority = 0.70 + angle_quality * 0.1 + distance_quality * 0.05
            style = PlayStyle.AGGRESSIVE
            risk = 0.6

        # Boost priority on continuation strokes
        if is_continuation:
            priority = min(0.99, priority + 0.05)

        # Adjust based on style weights
        priority *= (0.7 + style_weights[style] * 0.3)

        return TacticalAdvice(
            recommended_shot=ShotType.HOOP_RUN,
            target_position=hoop.position + hoop.direction * 2,
            target_ball=None,
            priority=priority,
            reason=f"Run hoop (dist={distance:.1f}, angle={dot:.2f})",
            style=style,
            risk_level=risk
        )

    def _check_peel_opportunity(
        self,
        striker: Ball,
        partner: Ball,
        all_balls: Dict[str, Ball],
        court: Court,
        live_balls: List[str]
    ) -> Optional[TacticalAdvice]:
        """
        Check if we can peel partner ball through its hoop.

        From Wylie Article 1: Peeling is sending a ball other than
        striker's ball through its hoop. Key for triple peels.

        Does NOT give extra strokes but DOES score the point.
        """
        if partner.color not in live_balls:
            return None  # Dead on partner

        partner_hoop = court.get_hoop_for_ball(partner.hoops_run)
        if not partner_hoop:
            return None  # Partner is rover

        # Check if partner is in position to be peeled
        to_hoop = partner_hoop.position - partner.position
        peel_distance = to_hoop.magnitude()

        if peel_distance > 3:
            return None  # Too far from hoop for peel

        # Check alignment
        peel_dir = to_hoop.normalize()
        alignment = peel_dir.dot(partner_hoop.direction)

        if alignment < 0.5:
            return None  # Bad angle for peel

        # Check if we can reach partner
        to_partner = partner.position - striker.position
        reach_distance = to_partner.magnitude()

        if reach_distance > 15:
            return None  # Too far to roquet

        # Calculate priority - peeling is valuable for partner advancement
        alignment_quality = (alignment - 0.5) / 0.5  # 0-1
        position_quality = max(0, 1 - peel_distance / 3)  # 0-1

        # Straight peels are best (from Wylie 1.III.7)
        is_straight_peel = alignment > 0.85

        if is_straight_peel and peel_distance < 1.5:
            priority = 0.85  # Excellent peel opportunity
        elif alignment > 0.7 and peel_distance < 2:
            priority = 0.75
        else:
            priority = 0.60 + alignment_quality * 0.1 + position_quality * 0.05

        return TacticalAdvice(
            recommended_shot=ShotType.PEEL,
            target_position=partner.position,
            target_ball=partner.color,
            priority=priority,
            reason=f"Peel {partner.color} through {self.HOOP_NAMES.get(partner.hoops_run, '?')}",
            style=PlayStyle.PRECISION,
            risk_level=0.4
        )

    def _check_rush_opportunity(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        hoop,
        live_balls: List[str],
        style_weights: Dict[PlayStyle, float]
    ) -> Optional[TacticalAdvice]:
        """
        Check if we can rush a ball toward our hoop.

        From Wylie 3.II.5: A rush is a roquet where you control where
        the roqueted ball goes. Essential for break building.
        """
        best_rush = None
        best_priority = 0

        for ball_color in live_balls:
            ball = all_balls.get(ball_color)
            if not ball:
                continue

            to_ball = ball.position - striker.position
            ball_dist = to_ball.magnitude()

            if ball_dist > 15 or ball_dist < 0.3:
                continue

            rush_direction = to_ball.normalize()
            ball_to_hoop = hoop.position - ball.position
            hoop_dist = ball_to_hoop.magnitude()

            if hoop_dist < 0.5:
                continue  # Ball already at hoop

            hoop_direction = ball_to_hoop.normalize()

            # Alignment: striker -> ball -> hoop
            # From Wylie: need good alignment for controlled rush
            alignment = rush_direction.dot(hoop_direction)

            if alignment < 0.25:
                continue  # Cut rush too difficult

            # Straight rush (alignment > 0.8) is best
            is_straight_rush = alignment > 0.8
            is_good_cut = 0.5 <= alignment <= 0.8

            # Calculate rush quality
            distance_score = max(0, 1.0 - ball_dist / 15)
            alignment_score = max(0, (alignment - 0.25) / 0.75)
            hoop_proximity = max(0, 1.0 - hoop_dist / 20)

            if is_straight_rush and ball_dist < 6:
                priority = 0.88
                style = PlayStyle.PRECISION
                risk = 0.15
            elif is_good_cut and ball_dist < 8:
                priority = 0.80 + alignment_score * 0.05
                style = PlayStyle.AGGRESSIVE
                risk = 0.3
            else:
                priority = 0.65 + distance_score * 0.1 + alignment_score * 0.1
                style = PlayStyle.AGGRESSIVE
                risk = 0.4

            # Adjust for style weights
            priority *= (0.7 + style_weights[style] * 0.3)

            if priority > best_priority:
                best_priority = priority
                rush_target = hoop.position - hoop.direction * 2

                best_rush = TacticalAdvice(
                    recommended_shot=ShotType.RUSH,
                    target_position=ball.position,
                    target_ball=ball_color,
                    priority=priority,
                    reason=f"Rush {ball_color} to hoop (align={alignment:.2f})",
                    style=style,
                    risk_level=risk
                )

        return best_rush

    def _check_pioneer_placement(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        court: Court,
        target_hoop,
        live_balls: List[str]
    ) -> Optional[TacticalAdvice]:
        """
        Check if we should place a ball as a pioneer at the next hoop.

        From Wylie: Pioneer = ball placed near next hoop.
        Critical for 4-ball break organization.
        """
        # Get next hoop after current target
        next_hoop = court.get_hoop_for_ball(striker.hoops_run + 1)
        if not next_hoop:
            return None

        # Check if we already have a pioneer at next hoop
        for color in live_balls:
            ball = all_balls.get(color)
            if ball:
                to_next = next_hoop.position - ball.position
                if to_next.magnitude() < 4:
                    return None  # Already have a pioneer

        # Look for a ball we can send to next hoop
        best_advice = None
        best_priority = 0

        for color in live_balls:
            ball = all_balls.get(color)
            if not ball:
                continue

            to_ball = ball.position - striker.position
            dist = to_ball.magnitude()

            if dist > 12:
                continue  # Too far

            # Priority based on how easy the pioneer placement is
            distance_quality = max(0, 1 - dist / 12)

            # Calculate where we'd send the pioneer
            pioneer_pos = next_hoop.position - next_hoop.direction * 3

            priority = 0.55 + distance_quality * 0.15

            if priority > best_priority:
                best_priority = priority
                best_advice = TacticalAdvice(
                    recommended_shot=ShotType.PIONEER_PLACEMENT,
                    target_position=pioneer_pos,
                    target_ball=color,
                    priority=priority,
                    reason=f"Place {color} as pioneer at next hoop",
                    style=PlayStyle.PRECISION,
                    risk_level=0.25
                )

        return best_advice

    def _check_hoop_approach(
        self,
        striker: Ball,
        hoop,
        court: Court,
        is_continuation: bool,
        style_weights: Dict[PlayStyle, float],
        has_live_balls: bool = True  # Whether there are balls to roquet
    ) -> Optional[TacticalAdvice]:
        """
        Suggest approaching the hoop to get into position.

        CRITICAL: Approaching WITHOUT roqueting ends your turn!
        This should only be high priority when:
        1. It's a continuation stroke (you've already roqueted)
        2. There are no live balls to roquet
        3. You're very close and just need a final approach

        From Wylie: Roqueting first, then using croquet stroke to
        approach, is ALWAYS better than approaching directly.
        """
        to_hoop = hoop.position - striker.position
        dist = to_hoop.magnitude()

        # Target position: 2-4 yards in front of hoop
        approach_pos = hoop.position - hoop.direction * 3

        # Check current angle
        if dist > 0:
            current_angle = to_hoop.normalize().dot(hoop.direction)
        else:
            current_angle = 1.0

        # IMPORTANT: If there are live balls, approaching directly
        # is usually a BAD choice because it ends your turn!
        # Only approach if continuation (already have strokes) or no roquet available

        if has_live_balls and not is_continuation:
            # There are balls to roquet - don't just approach!
            # This would waste our turn without gaining strokes
            # Give very low priority - roquet should be preferred
            priority = 0.25  # Lower than roquet priority
            style = PlayStyle.CANNY
            reason = f"Approach hoop (ENDS TURN - dist={dist:.1f})"
        elif is_continuation:
            # Continuation stroke - approaching is fine since we have strokes
            if dist > 10:
                priority = 0.70
            elif dist > 6:
                priority = 0.65
            elif dist > 3 and current_angle < 0.5:
                priority = 0.60  # Need to improve angle
            else:
                priority = 0.50  # Already close
            style = PlayStyle.PRECISION
            reason = f"Approach hoop (continuation - dist={dist:.1f})"
        else:
            # No live balls - approaching is our only option
            if dist > 10:
                priority = 0.60
            elif dist > 6:
                priority = 0.55
            else:
                priority = 0.45
            style = PlayStyle.PRECISION
            reason = f"Approach hoop (no roquet - dist={dist:.1f})"

        # Adjust for style
        priority *= (0.7 + style_weights[style] * 0.3)

        return TacticalAdvice(
            recommended_shot=ShotType.HOOP_APPROACH,
            target_position=approach_pos,
            target_ball=None,
            priority=priority,
            reason=reason,
            style=style,
            risk_level=0.15
        )

    def _check_any_roquet(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        live_balls: List[str],
        is_continuation: bool,
        style_weights: Dict[PlayStyle, float],
        hoop=None  # Pass target hoop to know if we're in position
    ) -> Optional[TacticalAdvice]:
        """
        Find any ball to roquet for strokes.

        CRITICAL PRINCIPLE: If we're NOT in position to run our hoop,
        roqueting a live ball should be HIGH priority - it's the only
        way to continue the break and get into position!

        From Wylie: Roquets are means to running hoops, but when you
        can't run the hoop, getting a roquet IS the priority.
        """
        best_target = None
        best_dist = float('inf')

        for ball_color in live_balls:
            ball = all_balls.get(ball_color)
            if not ball:
                continue

            dist = (ball.position - striker.position).magnitude()
            if dist < best_dist and dist < 25:
                best_dist = dist
                best_target = ball_color

        if not best_target:
            return None

        # Check if we're in position to run the hoop
        in_hoop_position = False
        if hoop:
            to_hoop = hoop.position - striker.position
            hoop_dist = to_hoop.magnitude()
            if hoop_dist > 0.5 and hoop_dist < 6:
                approach_dir = to_hoop.normalize()
                dot = approach_dir.dot(hoop.direction)
                in_hoop_position = dot > 0.5  # Good approach angle

        # CRITICAL: If NOT in hoop position, roquet priority is HIGH
        # because it's the ONLY way to continue the break!
        if not in_hoop_position:
            # Roquet is essential - prioritize it highly
            if best_dist < 5:
                priority = 0.85  # Very close roquet - almost guaranteed
                risk = 0.1
            elif best_dist < 10:
                priority = 0.75  # Good roquet range
                risk = 0.2
            elif best_dist < 15:
                priority = 0.65  # Medium range
                risk = 0.35
            else:
                priority = 0.55  # Long but still worth trying
                risk = 0.5
        else:
            # In hoop position - roquet is lower priority than running hoop
            if best_dist < 5:
                priority = 0.50
                risk = 0.1
            elif best_dist < 10:
                priority = 0.40
                risk = 0.2
            elif best_dist < 15:
                priority = 0.30
                risk = 0.35
            else:
                priority = 0.20
                risk = 0.5

        # On continuation, still value roquets if not in hoop position
        # (continuation means we have extra strokes to use)
        if is_continuation and in_hoop_position:
            priority = max(0.15, priority - 0.15)

        return TacticalAdvice(
            recommended_shot=ShotType.ROQUET,
            target_position=all_balls[best_target].position,
            target_ball=best_target,
            priority=priority,
            reason=f"Roquet {best_target} (dist={best_dist:.1f})",
            style=PlayStyle.PRECISION if best_dist < 8 else PlayStyle.AGGRESSIVE,
            risk_level=risk
        )

    def _check_wiring_opportunity(
        self,
        striker: Ball,
        opponents: List[Ball],
        court: Court,
        target_hoop
    ) -> Optional[TacticalAdvice]:
        """
        Check if we can wire opponent balls.

        From Wylie 3.II.2: Wiring is using hoops/peg to block shots.
        A ball is "wired" if hoop/peg prevents direct shot.

        From AC Laws: Wiring affects lift entitlements.
        """
        if not target_hoop:
            return None

        # Check if we can create a wire using the hoop
        for opp in opponents:
            to_opp = opp.position - target_hoop.position
            opp_dist = to_opp.magnitude()

            if opp_dist > 3:
                continue  # Too far from hoop to wire

            # Check if opponent is on opposite side of hoop from us
            striker_to_hoop = target_hoop.position - striker.position

            # Calculate wiring position
            wire_pos = target_hoop.position - to_opp.normalize() * 1.5

            priority = 0.45

            return TacticalAdvice(
                recommended_shot=ShotType.WIRE,
                target_position=wire_pos,
                target_ball=opp.color,
                priority=priority,
                reason=f"Wire {opp.color} behind hoop",
                style=PlayStyle.CANNY,
                risk_level=0.2
            )

        return None

    def _check_clearance_needed(
        self,
        striker: Ball,
        opponents: List[Ball],
        court: Court,
        deadness: Dict[str, set]
    ) -> Optional[TacticalAdvice]:
        """
        Check if opponent threatens their hoop and needs clearing.

        From Wylie: Sometimes must clear threatening opponent
        even at cost to own break.
        """
        striker_dead = deadness.get(striker.color, set())

        for opp in opponents:
            opp_hoop = court.get_hoop_for_ball(opp.hoops_run)
            if not opp_hoop:
                continue

            to_hoop = opp_hoop.position - opp.position
            dist = to_hoop.magnitude()

            if dist < 5:
                approach_dot = to_hoop.normalize().dot(opp_hoop.direction)
                if approach_dot > 0.6:
                    # Opponent threatens their hoop!
                    # Check if we can hit them
                    if opp.color in striker_dead:
                        continue  # Dead on them

                    to_opp = opp.position - striker.position
                    opp_dist = to_opp.magnitude()

                    if opp_dist > 20:
                        continue  # Too far

                    # Priority based on threat level
                    threat_level = (1 - dist / 5) * approach_dot
                    priority = 0.50 + threat_level * 0.2

                    return TacticalAdvice(
                        recommended_shot=ShotType.CLEARANCE,
                        target_position=opp.position,
                        target_ball=opp.color,
                        priority=priority,
                        reason=f"Clear threatening {opp.color}",
                        style=PlayStyle.CANNY,
                        risk_level=0.3
                    )

        return None

    def _check_defensive_leave(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        court: Court,
        target_hoop,
        live_balls: List[str],
        style_weights: Dict[PlayStyle, float]
    ) -> Optional[TacticalAdvice]:
        """
        Check if we should make a defensive leave.

        From Wylie 3.I.4: Canny croquet - when break not viable,
        leave positions that give opponent no good shot.

        Types of leaves:
        - New Standard Leave (NSL)
        - Defensive 4-back leaves
        - Corner positions
        """
        # Strategic areas from Wylie Fig 3.34
        # Balls in these areas can't be shot at safely from most of court
        strategic_corners = [
            Vector2(1, 1),       # Corner I
            Vector2(27, 1),      # Corner II
            Vector2(27, 34),     # Corner III
            Vector2(1, 34),      # Corner IV
        ]

        # Find best defensive position
        partner = self._get_partner_ball(striker.color, all_balls)
        if not partner:
            return None

        # Calculate leave position - near partner but on yard line
        partner_pos = partner.position

        # Find nearest strategic area
        best_pos = None
        best_dist = float('inf')
        for corner in strategic_corners:
            dist = (corner - partner_pos).magnitude()
            if dist < best_dist:
                best_dist = dist
                best_pos = corner

        if not best_pos:
            best_pos = Vector2(court.width / 2, 1)  # Default to south yard line

        # Priority based on canny style weight
        priority = 0.40 + style_weights[PlayStyle.CANNY] * 0.2

        return TacticalAdvice(
            recommended_shot=ShotType.DEFENSIVE,
            target_position=best_pos,
            target_ball=None,
            priority=priority,
            reason="Defensive leave - join up safely",
            style=PlayStyle.CANNY,
            risk_level=0.1
        )

    def _check_rush_setup(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        hoop,
        live_balls: List[str],
        court: Court
    ) -> Optional[TacticalAdvice]:
        """
        Find a position that sets up a good rush for next shot.

        From Wylie 3.II.5: When no immediate opportunity, position
        so striker -> ball -> hoop aligns for future rush.
        """
        best_setup = None
        best_priority = 0

        for ball_color in live_balls:
            ball = all_balls.get(ball_color)
            if not ball:
                continue

            ball_to_hoop = hoop.position - ball.position
            hoop_dist = ball_to_hoop.magnitude()

            if hoop_dist < 1 or hoop_dist > 25:
                continue

            # Ideal setup: behind ball, opposite to hoop
            rush_dir = ball_to_hoop.normalize()
            setup_pos = ball.position - rush_dir * 4

            if not court.is_in_bounds(setup_pos, 1):
                continue

            travel_dist = (setup_pos - striker.position).magnitude()

            if travel_dist > 18:
                continue

            # Priority based on setup quality
            hoop_proximity = max(0, 1.0 - hoop_dist / 20)
            travel_penalty = max(0, 1.0 - travel_dist / 18)

            priority = 0.40 + hoop_proximity * 0.15 + travel_penalty * 0.10

            if priority > best_priority:
                best_priority = priority
                best_setup = TacticalAdvice(
                    recommended_shot=ShotType.DEFENSIVE,
                    target_position=setup_pos,
                    target_ball=None,
                    priority=priority,
                    reason=f"Set up rush on {ball_color}",
                    style=PlayStyle.CANNY,
                    risk_level=0.15
                )

        return best_setup

    def _apply_style_adjustments(
        self,
        advice: List[TacticalAdvice],
        style_weights: Dict[PlayStyle, float],
        is_continuation: bool
    ) -> List[TacticalAdvice]:
        """
        Apply style-based priority adjustments.

        From Wylie: Balance of styles is key. Don't overuse any one style.
        """
        for a in advice:
            style_modifier = style_weights.get(a.style, 0.5)

            # Penalize high-risk shots if not aggressive style
            if a.risk_level > 0.5 and style_weights[PlayStyle.AGGRESSIVE] < 0.4:
                a.priority *= 0.85

            # Boost safe shots for canny players
            if a.risk_level < 0.3 and style_weights[PlayStyle.CANNY] > 0.5:
                a.priority *= 1.1

            # On continuation, boost hoop-related shots
            if is_continuation:
                if a.recommended_shot in [ShotType.HOOP_RUN, ShotType.HOOP_APPROACH]:
                    a.priority = min(0.99, a.priority * 1.15)

        return advice

    def get_croquet_advice(
        self,
        striker: Ball,
        roqueted: Ball,
        all_balls: Dict[str, Ball],
        court: Court
    ) -> TacticalAdvice:
        """
        Get advice for a croquet stroke.

        From Wylie: After rushing a ball to the hoop, the croquet stroke should:
        1. Send object ball past hoop (as pioneer for NEXT hoop)
        2. Position striker in front of current hoop to run it

        This is the key to break play:
        Rush -> Croquet to set up -> Run hoop -> Rush to next hoop
        """
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)
        next_hoop = court.get_hoop_for_ball(striker.hoops_run + 1)

        if target_hoop:
            to_hoop = target_hoop.position - roqueted.position
            dist_to_hoop = to_hoop.magnitude()

            if dist_to_hoop < 4:
                # Object ball near hoop - perfect for split
                if next_hoop:
                    # Split: object ball to next hoop, striker to current hoop
                    pioneer_pos = next_hoop.position - next_hoop.direction * 4
                    striker_target = target_hoop.position - target_hoop.direction * 2

                    return TacticalAdvice(
                        recommended_shot=ShotType.SPLIT,
                        target_position=striker_target,
                        target_ball=roqueted.color,
                        priority=0.90,
                        reason="Split: pioneer to next, striker to hoop",
                        style=PlayStyle.PRECISION,
                        risk_level=0.25
                    )
                else:
                    # No next hoop (rover) - take-off to run hoop
                    striker_target = target_hoop.position - target_hoop.direction * 2
                    return TacticalAdvice(
                        recommended_shot=ShotType.TAKE_OFF,
                        target_position=striker_target,
                        target_ball=None,
                        priority=0.85,
                        reason="Take-off to run hoop",
                        style=PlayStyle.PRECISION,
                        risk_level=0.2
                    )

            elif dist_to_hoop < 8:
                # Object ball moderately close - drive both to hoop
                drive_target = target_hoop.position - target_hoop.direction * 1

                return TacticalAdvice(
                    recommended_shot=ShotType.DRIVE,
                    target_position=drive_target,
                    target_ball=roqueted.color,
                    priority=0.80,
                    reason="Drive both to hoop",
                    style=PlayStyle.AGGRESSIVE,
                    risk_level=0.35
                )

            else:
                # Object ball far from hoop - take-off to approach
                approach_pos = target_hoop.position - target_hoop.direction * 3

                return TacticalAdvice(
                    recommended_shot=ShotType.TAKE_OFF,
                    target_position=approach_pos,
                    target_ball=None,
                    priority=0.70,
                    reason="Take-off toward hoop",
                    style=PlayStyle.PRECISION,
                    risk_level=0.2
                )

        # Rover - head toward peg
        if striker.is_rover:
            return TacticalAdvice(
                recommended_shot=ShotType.TAKE_OFF,
                target_position=court.peg_position,
                target_ball=None,
                priority=0.90,
                reason="Take-off toward peg",
                style=PlayStyle.AGGRESSIVE,
                risk_level=0.15
            )

        # Default: standard croquet toward center
        return TacticalAdvice(
            recommended_shot=ShotType.DRIVE,
            target_position=Vector2(court.width / 2, court.height / 2),
            target_ball=roqueted.color,
            priority=0.50,
            reason="Standard croquet stroke",
            style=PlayStyle.PRECISION,
            risk_level=0.25
        )

    def check_lift_entitlement(
        self,
        striker_hoops: int,
        opponent_balls: List[Ball],
        are_wired: bool
    ) -> bool:
        """
        Check if opponent gets a lift based on Advanced Play rules.

        From AC Laws:
        - Lift if striker leaves balls wired from each other
        - Lift after running 1-back or 4-back (Advanced Play)
        """
        # Check for lift hoops (1-back = 6, 4-back = 9)
        if striker_hoops in self.LIFT_HOOPS:
            return True

        # Check for wiring
        if are_wired:
            return True

        return False

    def _get_live_balls(
        self,
        striker_color: str,
        deadness: Dict[str, set]
    ) -> List[str]:
        """Get balls that can be roqueted."""
        all_colors = {"blue", "black", "red", "yellow"}
        all_colors.discard(striker_color)
        dead_on = deadness.get(striker_color, set())
        return [c for c in all_colors if c not in dead_on]

    def _get_partner_ball(
        self,
        color: str,
        all_balls: Dict[str, Ball]
    ) -> Optional[Ball]:
        """Get partner ball."""
        partners = {"blue": "black", "black": "blue", "red": "yellow", "yellow": "red"}
        partner_color = partners.get(color)
        return all_balls.get(partner_color)

    def _get_opponent_balls(
        self,
        color: str,
        all_balls: Dict[str, Ball]
    ) -> List[Ball]:
        """Get opponent balls."""
        if color in ["blue", "black"]:
            opp_colors = ["red", "yellow"]
        else:
            opp_colors = ["blue", "black"]

        return [all_balls[c] for c in opp_colors if c in all_balls]

    def is_wired(
        self,
        ball1: Ball,
        ball2: Ball,
        obstacles: List[Vector2],
        obstacle_radius: float = 0.2
    ) -> bool:
        """
        Check if ball1 is wired from ball2 (cannot hit directly).

        From Wylie 3.II.2: A ball is wired if a hoop or peg
        prevents a direct shot.
        """
        direction = ball2.position - ball1.position
        distance = direction.magnitude()

        if distance < 0.1:
            return False  # Balls touching

        direction = direction.normalize()

        # Check each obstacle
        for obstacle in obstacles:
            # Vector from ball1 to obstacle
            to_obstacle = obstacle - ball1.position

            # Project onto direction
            projection = to_obstacle.dot(direction)

            if projection < 0 or projection > distance:
                continue  # Obstacle not between balls

            # Distance from obstacle to line
            closest_point = ball1.position + direction * projection
            obstacle_dist = (obstacle - closest_point).magnitude()

            # Ball radius is about 0.1 yards
            if obstacle_dist < obstacle_radius + 0.1:
                return True  # Wired!

        return False
