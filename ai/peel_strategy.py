"""
Peel Strategy - Triple Peel planning and execution for Association Croquet.

A Triple Peel (TP) is an advanced technique where the striker peels their
partner ball through three hoops (4-back, penult, rover) while making their
own break. This is considered standard expert play.

From Wylie's "Expert Croquet Tactics":
- The TP is the "standard finish" for expert players
- Peeling should be integrated smoothly into break play
- The key is positioning the peeled ball at each hoop opportunity
- Straight peels, Irish peels, and rush peels are the main techniques

This module provides:
- PeelState: Tracks which peels have been completed
- PeelPlanner: Plans when and how to attempt peels
- PeelOpportunity: Identifies good peel setups
"""
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

from models.ball import Ball, Vector2
from models.court import Court


class PeelType(Enum):
    """Types of peel strokes from Wylie."""
    STRAIGHT = auto()      # Standard peel - hit ball straight through hoop
    RUSH_PEEL = auto()     # Peel during a rush (roquet that sends ball through)
    IRISH = auto()         # Ball in jaws, peel from the side
    SPLIT_PEEL = auto()    # Peel during a split croquet stroke
    STOP_SHOT_PEEL = auto()  # Peel using stop shot technique
    CANNON_PEEL = auto()   # Three-ball cannon that peels


class PeelDifficulty(Enum):
    """Difficulty ratings for peel attempts."""
    EASY = auto()          # Ball well-positioned, straight shot
    MODERATE = auto()      # Some angle or distance challenge
    DIFFICULT = auto()     # Challenging position or angle
    EXPERT = auto()        # Very precise execution required


@dataclass
class PeelState:
    """
    Tracks the state of peeling for a triple peel attempt.

    In a standard TP, the partner ball needs to go through:
    - 4-back (hoop 10)
    - Penult (hoop 11)
    - Rover (hoop 12)

    The striker typically runs hoops 1-back through rover themselves
    while peeling partner through these three hoops.
    """
    partner_color: str
    peels_needed: List[int] = field(default_factory=list)  # Hoop numbers
    peels_completed: List[int] = field(default_factory=list)
    tp_in_progress: bool = False
    tp_started_at_hoop: Optional[int] = None  # Striker's hoop when TP began

    def __post_init__(self):
        """Initialize peels needed based on partner position."""
        # Standard TP: 4-back, penult, rover
        if not self.peels_needed:
            self.peels_needed = [10, 11, 12]

    @property
    def next_peel_hoop(self) -> Optional[int]:
        """Get the next hoop the partner needs to be peeled through."""
        for hoop in self.peels_needed:
            if hoop not in self.peels_completed:
                return hoop
        return None

    @property
    def peels_remaining(self) -> int:
        """Number of peels still needed."""
        return len(self.peels_needed) - len(self.peels_completed)

    @property
    def is_complete(self) -> bool:
        """Check if the triple peel is complete."""
        return self.peels_remaining == 0

    def record_peel(self, hoop_num: int):
        """Record a successful peel."""
        if hoop_num not in self.peels_completed:
            self.peels_completed.append(hoop_num)

    def get_peel_name(self, hoop_num: int) -> str:
        """Get the name of a peel hoop."""
        names = {10: "4-back", 11: "penult", 12: "rover"}
        return names.get(hoop_num, f"hoop {hoop_num}")


@dataclass
class PeelOpportunity:
    """Represents an opportunity to attempt a peel."""
    peel_type: PeelType
    target_hoop: int
    partner_ball: Ball
    difficulty: PeelDifficulty
    setup_position: Vector2  # Where striker should be
    aim_direction: Vector2   # Direction to hit
    power: float
    success_probability: float  # 0-1
    description: str


class PeelPlanner:
    """
    Plans triple peel attempts during a break.

    The planner considers:
    - Partner ball position relative to its next hoop
    - Striker's current break state
    - Available balls for continuing the break after peel
    - Risk vs reward of attempting the peel
    """

    def __init__(self, skill_level: float = 0.75):
        """
        Initialize the peel planner.

        Args:
            skill_level: AI skill level (0-1), affects peel success rates
        """
        self.skill_level = skill_level
        self.peel_state: Optional[PeelState] = None

        # Peel success base rates by type (modified by skill and difficulty)
        self.base_success_rates = {
            PeelType.STRAIGHT: 0.85,
            PeelType.RUSH_PEEL: 0.70,
            PeelType.IRISH: 0.75,
            PeelType.SPLIT_PEEL: 0.65,
            PeelType.STOP_SHOT_PEEL: 0.70,
            PeelType.CANNON_PEEL: 0.50,
        }

        # Difficulty modifiers
        self.difficulty_modifiers = {
            PeelDifficulty.EASY: 1.1,
            PeelDifficulty.MODERATE: 1.0,
            PeelDifficulty.DIFFICULT: 0.8,
            PeelDifficulty.EXPERT: 0.6,
        }

    def should_attempt_tp(
        self,
        striker: Ball,
        partner: Ball,
        all_balls: Dict[str, Ball],
        court: Court
    ) -> Tuple[bool, str]:
        """
        Determine if a triple peel should be attempted.

        Conditions for starting a TP:
        1. Striker is for 1-back or later (hoops_run >= 6)
        2. Partner is for 4-back (hoops_run == 9)
        3. Both balls are in good positions
        4. Skill level is sufficient

        Args:
            striker: The striker ball
            partner: The partner ball
            all_balls: All balls on court
            court: The court

        Returns:
            Tuple of (should_attempt, reason)
        """
        # Check striker progress
        if striker.hoops_run < 6:
            return (False, "Striker not far enough (need 1-back or later)")

        # Check partner position - should be for 4-back to start standard TP
        if partner.hoops_run < 9:
            return (False, f"Partner only at hoop {partner.hoops_run + 1}, need 4-back")

        if partner.hoops_run >= 12:
            return (False, "Partner is already a rover")

        # Check skill level
        if self.skill_level < 0.6:
            return (False, "Skill level too low for TP")

        # Check if partner is reasonably positioned
        partner_hoop = court.get_hoop_for_ball(partner.hoops_run)
        if partner_hoop:
            dist_to_hoop = (partner.position - partner_hoop.position).magnitude()
            if dist_to_hoop > 15:
                return (False, "Partner too far from their hoop")

        return (True, "Good position for triple peel")

    def initialize_tp(self, striker: Ball, partner: Ball) -> PeelState:
        """
        Initialize tracking for a triple peel attempt.

        Args:
            striker: The striker ball
            partner: The partner ball

        Returns:
            PeelState for tracking the TP
        """
        # Determine which peels are needed based on partner's current position
        peels_needed = []
        for hoop in [10, 11, 12]:  # 4-back, penult, rover
            if partner.hoops_run < hoop:
                peels_needed.append(hoop)

        self.peel_state = PeelState(
            partner_color=partner.color,
            peels_needed=peels_needed,
            tp_in_progress=True,
            tp_started_at_hoop=striker.hoops_run
        )

        return self.peel_state

    def find_peel_opportunity(
        self,
        striker: Ball,
        partner: Ball,
        court: Court,
        is_croquet_stroke: bool = False
    ) -> Optional[PeelOpportunity]:
        """
        Find the best peel opportunity given current positions.

        Args:
            striker: The striker ball
            partner: The partner ball
            court: The court
            is_croquet_stroke: Whether this is during a croquet stroke

        Returns:
            PeelOpportunity if a good opportunity exists, None otherwise
        """
        if not self.peel_state or self.peel_state.is_complete:
            return None

        next_peel = self.peel_state.next_peel_hoop
        if not next_peel:
            return None

        target_hoop = court.get_hoop_for_ball(next_peel - 1)  # hoops_run is 0-indexed
        if not target_hoop:
            return None

        # Calculate partner's position relative to hoop
        to_hoop = target_hoop.position - partner.position
        dist_to_hoop = to_hoop.magnitude()

        if dist_to_hoop < 0.5:
            # Partner very close - might be jawsed
            return self._check_irish_peel(partner, target_hoop, next_peel)
        elif dist_to_hoop < 4:
            # Good distance for straight peel
            return self._check_straight_peel(striker, partner, target_hoop, next_peel, is_croquet_stroke)
        elif dist_to_hoop < 8:
            # Moderate distance - could rush peel
            return self._check_rush_peel(striker, partner, target_hoop, next_peel)
        else:
            # Too far for reliable peel
            return None

    def _check_straight_peel(
        self,
        striker: Ball,
        partner: Ball,
        target_hoop,
        hoop_num: int,
        is_croquet: bool
    ) -> Optional[PeelOpportunity]:
        """Check for straight peel opportunity."""
        to_hoop = target_hoop.position - partner.position
        dist = to_hoop.magnitude()

        if dist < 0.1:
            return None

        # Direction from partner to hoop
        peel_dir = to_hoop.normalize()

        # Check alignment with hoop direction
        alignment = peel_dir.dot(target_hoop.direction)

        if alignment < 0.5:
            # Poor alignment
            difficulty = PeelDifficulty.DIFFICULT
        elif alignment < 0.8:
            difficulty = PeelDifficulty.MODERATE
        else:
            difficulty = PeelDifficulty.EASY

        # Calculate where striker should be for the peel
        # Striker should be behind partner, opposite to peel direction
        setup_pos = partner.position - peel_dir * 0.2  # Contact distance

        # Calculate power needed
        power = self._calculate_peel_power(dist, is_croquet)

        # Calculate success probability
        base_rate = self.base_success_rates[PeelType.STRAIGHT]
        modifier = self.difficulty_modifiers[difficulty]
        success_prob = min(1.0, base_rate * modifier * self.skill_level)

        peel_type = PeelType.SPLIT_PEEL if is_croquet else PeelType.STRAIGHT

        return PeelOpportunity(
            peel_type=peel_type,
            target_hoop=hoop_num,
            partner_ball=partner,
            difficulty=difficulty,
            setup_position=setup_pos,
            aim_direction=peel_dir,
            power=power,
            success_probability=success_prob,
            description=f"Straight peel through {self.peel_state.get_peel_name(hoop_num)}"
        )

    def _check_irish_peel(
        self,
        partner: Ball,
        target_hoop,
        hoop_num: int
    ) -> Optional[PeelOpportunity]:
        """
        Check for Irish peel opportunity (ball in jaws).

        An Irish peel is when the ball is in the jaws of the hoop
        and you peel it from the side.
        """
        to_hoop = target_hoop.position - partner.position
        dist = to_hoop.magnitude()

        if dist > 0.5:
            return None  # Not close enough for Irish peel

        # Ball is in or near jaws - Irish peel possible
        # Peel from the side, perpendicular to hoop direction
        side_dir = Vector2(-target_hoop.direction.y, target_hoop.direction.x)

        # Choose the better side based on partner position
        partner_side = (partner.position - target_hoop.position).dot(side_dir)
        if partner_side > 0:
            peel_dir = side_dir * -1  # Peel from the other side
        else:
            peel_dir = side_dir

        setup_pos = partner.position - peel_dir * 0.2

        # Irish peels are generally reliable when ball is jawsed
        difficulty = PeelDifficulty.MODERATE
        base_rate = self.base_success_rates[PeelType.IRISH]
        success_prob = min(1.0, base_rate * self.difficulty_modifiers[difficulty] * self.skill_level)

        return PeelOpportunity(
            peel_type=PeelType.IRISH,
            target_hoop=hoop_num,
            partner_ball=partner,
            difficulty=difficulty,
            setup_position=setup_pos,
            aim_direction=peel_dir,
            power=3.0,  # Moderate power for Irish peel
            success_probability=success_prob,
            description=f"Irish peel through {self.peel_state.get_peel_name(hoop_num)} (jawsed)"
        )

    def _check_rush_peel(
        self,
        striker: Ball,
        partner: Ball,
        target_hoop,
        hoop_num: int
    ) -> Optional[PeelOpportunity]:
        """Check for rush peel opportunity."""
        to_hoop = target_hoop.position - partner.position
        dist = to_hoop.magnitude()

        if dist < 2 or dist > 10:
            return None

        peel_dir = to_hoop.normalize()

        # For rush peel, striker needs to be behind partner with good line to hoop
        striker_to_partner = partner.position - striker.position
        striker_dist = striker_to_partner.magnitude()

        if striker_dist > 8:
            return None  # Too far for controlled rush

        # Check if striker can rush partner toward hoop
        if striker_dist > 0.1:
            rush_dir = striker_to_partner.normalize()
            alignment = rush_dir.dot(peel_dir)

            if alignment < 0.6:
                return None  # Poor rush line
        else:
            alignment = 1.0

        # Determine difficulty based on distance and alignment
        if alignment > 0.9 and dist < 5:
            difficulty = PeelDifficulty.MODERATE
        elif alignment > 0.7:
            difficulty = PeelDifficulty.DIFFICULT
        else:
            difficulty = PeelDifficulty.EXPERT

        # Setup is striker's current position (it's a rush)
        setup_pos = striker.position

        # Power based on distance to hoop
        power = self._calculate_peel_power(dist + striker_dist, False)

        base_rate = self.base_success_rates[PeelType.RUSH_PEEL]
        success_prob = min(1.0, base_rate * self.difficulty_modifiers[difficulty] * self.skill_level)

        return PeelOpportunity(
            peel_type=PeelType.RUSH_PEEL,
            target_hoop=hoop_num,
            partner_ball=partner,
            difficulty=difficulty,
            setup_position=setup_pos,
            aim_direction=peel_dir,
            power=power,
            success_probability=success_prob,
            description=f"Rush peel through {self.peel_state.get_peel_name(hoop_num)}"
        )

    def _calculate_peel_power(self, distance: float, is_croquet: bool) -> float:
        """Calculate power needed for a peel at given distance."""
        # Base power calculation
        import config
        friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY
        base_power = math.sqrt(2 * friction_decel * distance)

        # Croquet strokes transfer less energy to croqueted ball
        if is_croquet:
            base_power *= 1.5  # Need more power for split

        return min(base_power * 1.2, config.MAX_SHOT_POWER)

    def get_peel_setup_advice(
        self,
        striker: Ball,
        partner: Ball,
        court: Court
    ) -> Optional[Dict]:
        """
        Get advice on how to set up for the next peel opportunity.

        Returns positioning advice for where to place balls during
        the break to prepare for peel attempts.
        """
        if not self.peel_state or self.peel_state.is_complete:
            return None

        next_peel = self.peel_state.next_peel_hoop
        if not next_peel:
            return None

        target_hoop = court.get_hoop_for_ball(next_peel - 1)
        if not target_hoop:
            return None

        # Calculate ideal partner position for next peel
        # Usually 2-4 yards in front of hoop, on line
        ideal_partner_pos = target_hoop.position - target_hoop.direction * 3

        # Current position
        partner_to_ideal = ideal_partner_pos - partner.position
        partner_dist = partner_to_ideal.magnitude()

        advice = {
            "next_peel": self.peel_state.get_peel_name(next_peel),
            "partner_position": partner.position,
            "ideal_position": ideal_partner_pos,
            "distance_to_ideal": partner_dist,
            "recommendation": ""
        }

        if partner_dist < 2:
            advice["recommendation"] = "Partner well-positioned for peel"
        elif partner_dist < 5:
            advice["recommendation"] = f"Move partner {partner_dist:.1f}m toward {self.peel_state.get_peel_name(next_peel)}"
        else:
            advice["recommendation"] = f"Partner needs repositioning for peel (currently {partner_dist:.1f}m away)"

        return advice

    def should_prioritize_peel(
        self,
        striker: Ball,
        partner: Ball,
        peel_opportunity: PeelOpportunity,
        court: Court
    ) -> Tuple[bool, str]:
        """
        Determine if a peel attempt should be prioritized over normal break play.

        Args:
            striker: The striker ball
            partner: The partner ball
            peel_opportunity: The available peel opportunity
            court: The court

        Returns:
            Tuple of (should_prioritize, reason)
        """
        if not peel_opportunity:
            return (False, "No peel opportunity")

        # Always prioritize if success probability is very high
        if peel_opportunity.success_probability > 0.85:
            return (True, "Excellent peel opportunity")

        # Prioritize rover peel (final peel) if reasonably good
        if peel_opportunity.target_hoop == 12 and peel_opportunity.success_probability > 0.6:
            return (True, "Rover peel - complete the TP!")

        # Don't risk break for low probability peels
        if peel_opportunity.success_probability < 0.5:
            return (False, "Peel too risky - continue break")

        # Consider game state - if leading comfortably, be more conservative
        # This would need game score information to implement fully

        # Default: attempt if moderate or better probability
        if peel_opportunity.success_probability >= 0.6:
            return (True, f"Good opportunity ({peel_opportunity.success_probability:.0%} success)")

        return (False, "Continue break, set up better peel")
