"""
Opening Strategy - Strategic ball placement for game openings.

Based on authentic Association Croquet opening theory:
- First ball placement options (east boundary, supershot, etc.)
- Second ball responses (tice, duffer tice, corner placements)
- Third and fourth turn responses
"""
import math
import random
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum, auto

from models.ball import Ball, Vector2
from models.court import Court
import config


class OpeningType(Enum):
    """Types of opening placements."""
    # First ball (Blue) options
    EAST_BOUNDARY = auto()      # Safe, non-threatening position
    SUPERSHOT = auto()          # Aggressive center position near H5-peg
    ANTI_DUFFER = auto()        # Between H6 and peg, discourages duffer tice
    CORNER_IV = auto()          # Corner 4 placement

    # Second ball (Red) responses
    STANDARD_TICE = auto()      # West boundary, 8-13 yards north of C1
    DUFFER_TICE = auto()        # Near penultimate (H6), very aggressive
    CORNER_II = auto()          # Defensive C2 placement
    CORNER_IV_RESPONSE = auto() # C4 placement, signals 4th turn intent
    JOIN_PARTNER = auto()       # Join partner ball

    # Third/Fourth turn
    SHOOT_TICE = auto()         # Shoot at the tice
    SHOOT_PARTNER = auto()      # Shoot at partner
    DEFENSIVE_JOIN = auto()     # Join defensively


@dataclass
class OpeningPlan:
    """Plan for an opening shot."""
    opening_type: OpeningType
    target_position: Vector2
    power: float
    description: str
    priority: float  # 0-1, how good is this option


class OpeningPlanner:
    """
    Plans opening shots based on game state and opponent behavior.

    Implements authentic croquet opening theory with adjustments
    based on player skill levels.
    """

    def __init__(self, aggression: float = 0.5):
        """
        Initialize opening planner.

        Args:
            aggression: How aggressive to play (0=defensive, 1=aggressive)
        """
        self.aggression = aggression

    def get_opening_shot(
        self,
        ball: Ball,
        all_balls: Dict[str, Ball],
        balls_in_play: Dict[str, bool],
        court: Court,
        turn_number: int
    ) -> Tuple[Vector2, float, str]:
        """
        Get the best opening shot for the current situation.

        Args:
            ball: The ball to shoot
            all_balls: All balls
            balls_in_play: Which balls have entered play
            court: The court
            turn_number: Current turn number (1-4 for opening)

        Returns:
            Tuple of (target_position, power, description)
        """
        # Count balls in play
        in_play_count = sum(1 for v in balls_in_play.values() if v)

        if in_play_count == 0:
            # First ball - Blue's opening
            return self._first_ball_opening(ball, court)
        elif in_play_count == 1:
            # Second ball - Red's response
            return self._second_ball_response(ball, all_balls, balls_in_play, court)
        elif in_play_count == 2:
            # Third ball - Black's turn
            return self._third_ball_response(ball, all_balls, balls_in_play, court)
        else:
            # Fourth ball - Yellow's turn
            return self._fourth_ball_response(ball, all_balls, balls_in_play, court)

    def _first_ball_opening(
        self,
        ball: Ball,
        court: Court
    ) -> Tuple[Vector2, float, str]:
        """
        First ball (Blue) opening options.

        Main options:
        - East boundary: Safe, between H4 and H5 level
        - Supershot: Aggressive, near center between H5 and peg
        - Anti-duffer: Between H6 and peg, west of east boundary
        """
        options = []

        # East boundary - most common, safe option
        # Position between hoop 4 (y=7) and hoop 5 (y=10.5) level
        east_target = Vector2(court.width - 2, 9)  # 2 yards from east boundary
        options.append(OpeningPlan(
            opening_type=OpeningType.EAST_BOUNDARY,
            target_position=east_target,
            power=self._power_for_distance(ball.position, east_target),
            description="East boundary opening",
            priority=0.7 if self.aggression < 0.6 else 0.5
        ))

        # Supershot - aggressive option
        # Near center, between H5 (y=10.5) and peg (y=17.5)
        supershot_target = Vector2(court.width / 2, 13)
        options.append(OpeningPlan(
            opening_type=OpeningType.SUPERSHOT,
            target_position=supershot_target,
            power=self._power_for_distance(ball.position, supershot_target),
            description="Supershot opening",
            priority=0.8 if self.aggression > 0.6 else 0.4
        ))

        # Anti-duffer - between H6 and peg, 1-2 yards west of east boundary
        anti_duffer_target = Vector2(court.width - 4, 20)
        options.append(OpeningPlan(
            opening_type=OpeningType.ANTI_DUFFER,
            target_position=anti_duffer_target,
            power=self._power_for_distance(ball.position, anti_duffer_target),
            description="Anti-duffer opening",
            priority=0.6
        ))

        # Select based on priority with significant randomness for variety
        # Use weighted random selection instead of just picking the best
        weights = [o.priority + random.uniform(0, 0.5) for o in options]
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                best = options[i]
                break
        else:
            best = options[-1]

        # Add small random offset to target position for natural variation
        offset = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        target = best.target_position + offset
        return (target, best.power, best.description)

    def _second_ball_response(
        self,
        ball: Ball,
        all_balls: Dict[str, Ball],
        balls_in_play: Dict[str, bool],
        court: Court
    ) -> Tuple[Vector2, float, str]:
        """
        Second ball (Red) response to first ball placement.

        Depends on where Blue went:
        - Against east boundary: Standard tice, duffer tice, or C2
        - Against supershot: C2, max distance, or roquet attempt
        """
        # Find the first ball (Blue)
        blue_ball = all_balls.get("blue")
        if not blue_ball:
            return self._default_opening(ball, court)

        blue_pos = blue_ball.position

        # Determine what opening Blue played
        if blue_pos.x > court.width - 5:
            # Blue on east boundary
            return self._respond_to_east_boundary(ball, blue_ball, court)
        elif blue_pos.x < court.width / 2 + 3 and blue_pos.y < 15:
            # Blue played supershot
            return self._respond_to_supershot(ball, blue_ball, court)
        else:
            # Other opening
            return self._standard_tice(ball, court)

    def _respond_to_east_boundary(
        self,
        ball: Ball,
        opponent: Ball,
        court: Court
    ) -> Tuple[Vector2, float, str]:
        """Respond to east boundary opening."""
        options = []

        # Standard tice - west boundary, 8-13 yards north of corner 1
        # Corner 1 is at y=0, so tice at y=8-13
        tice_y = 8 + random.uniform(0, 5)
        tice_target = Vector2(2, tice_y)  # 2 yards from west boundary
        options.append(OpeningPlan(
            opening_type=OpeningType.STANDARD_TICE,
            target_position=tice_target,
            power=self._power_for_distance(ball.position, tice_target),
            description="Standard tice",
            priority=0.7
        ))

        # Duffer tice - near penultimate (H6 at y=24.5), very aggressive
        # About 2 feet north of H6
        duffer_target = Vector2(2, 25)
        options.append(OpeningPlan(
            opening_type=OpeningType.DUFFER_TICE,
            target_position=duffer_target,
            power=self._power_for_distance(ball.position, duffer_target),
            description="Duffer tice",
            priority=0.8 if self.aggression > 0.7 else 0.3
        ))

        # Corner 2 - defensive, 6 inches south of corner 2
        c2_target = Vector2(1, court.height - 1)
        options.append(OpeningPlan(
            opening_type=OpeningType.CORNER_II,
            target_position=c2_target,
            power=self._power_for_distance(ball.position, c2_target),
            description="Corner 2 placement",
            priority=0.5 if self.aggression < 0.4 else 0.3
        ))

        # Corner 4 - aggressive, signals 4th turn break intent
        c4_target = Vector2(court.width - 1, 1)
        options.append(OpeningPlan(
            opening_type=OpeningType.CORNER_IV_RESPONSE,
            target_position=c4_target,
            power=self._power_for_distance(ball.position, c4_target),
            description="Corner 4 placement",
            priority=0.6 if self.aggression > 0.5 else 0.4
        ))

        # Use weighted random selection for variety
        weights = [o.priority + random.uniform(0, 0.4) for o in options]
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                best = options[i]
                break
        else:
            best = options[-1]

        # Add small random offset for natural variation
        offset = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        target = best.target_position + offset
        return (target, best.power, best.description)

    def _respond_to_supershot(
        self,
        ball: Ball,
        opponent: Ball,
        court: Court
    ) -> Tuple[Vector2, float, str]:
        """Respond to supershot opening."""
        options = []

        # Corner 2 - forces difficult croquet stroke
        c2_target = Vector2(1, court.height - 1)
        options.append(OpeningPlan(
            opening_type=OpeningType.CORNER_II,
            target_position=c2_target,
            power=self._power_for_distance(ball.position, c2_target),
            description="C2 response to supershot",
            priority=0.7
        ))

        # Max distance - east boundary near peg level
        max_dist_target = Vector2(court.width - 2, court.height / 2)
        options.append(OpeningPlan(
            opening_type=OpeningType.EAST_BOUNDARY,
            target_position=max_dist_target,
            power=self._power_for_distance(ball.position, max_dist_target),
            description="Max distance response",
            priority=0.6
        ))

        # Supershot lag - 3-4 yards east and 1-2 north of opponent
        lag_target = opponent.position + Vector2(3.5, 1.5)
        options.append(OpeningPlan(
            opening_type=OpeningType.SUPERSHOT,
            target_position=lag_target,
            power=self._power_for_distance(ball.position, lag_target),
            description="Supershot lag",
            priority=0.5
        ))

        # Use weighted random selection for variety
        weights = [o.priority + random.uniform(0, 0.4) for o in options]
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                best = options[i]
                break
        else:
            best = options[-1]

        # Add small random offset for natural variation
        offset = Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        target = best.target_position + offset
        return (target, best.power, best.description)

    def _third_ball_response(
        self,
        ball: Ball,
        all_balls: Dict[str, Ball],
        balls_in_play: Dict[str, bool],
        court: Court
    ) -> Tuple[Vector2, float, str]:
        """
        Third ball (Black) response.

        Options depend on second ball placement:
        - Against standard tice: Shoot from A-baulk
        - Against C2: Shoot partner or join strategically
        - Against duffer tice: Risk assessment for 9-yard shot
        """
        blue_ball = all_balls.get("blue")
        red_ball = all_balls.get("red")

        if not blue_ball or not red_ball:
            return self._default_opening(ball, court)

        # Check if red played a tice (west boundary)
        if red_ball.position.x < 5:
            # Red is on west - consider shooting at tice
            tice_dist = (red_ball.position - ball.position).magnitude()

            if tice_dist < 15 and self.aggression > 0.4:
                # Shoot at the tice
                return (red_ball.position,
                        self._power_for_distance(ball.position, red_ball.position) * 1.05,
                        "Shoot at tice")
            else:
                # Join partner
                join_target = blue_ball.position + Vector2(-2, 0)
                return (join_target,
                        self._power_for_distance(ball.position, join_target),
                        "Join partner")

        # Red in corner - shoot at partner or join
        partner_dist = (blue_ball.position - ball.position).magnitude()
        if partner_dist < 20:
            return (blue_ball.position,
                    self._power_for_distance(ball.position, blue_ball.position) * 1.02,
                    "Shoot at partner")

        # Default to center position
        return self._default_opening(ball, court)

    def _fourth_ball_response(
        self,
        ball: Ball,
        all_balls: Dict[str, Ball],
        balls_in_play: Dict[str, bool],
        court: Court
    ) -> Tuple[Vector2, float, str]:
        """
        Fourth ball (Yellow) response.

        Usually simpler - roquet the best available target.
        """
        # Find best target based on distance and break potential
        best_target = None
        best_score = -float('inf')

        for color, other_ball in all_balls.items():
            if color == ball.color:
                continue
            if not balls_in_play.get(color, False):
                continue

            dist = (other_ball.position - ball.position).magnitude()

            # Score based on distance (closer is better) and team (partner is better)
            is_partner = (ball.color == "yellow" and color == "red") or \
                        (ball.color == "red" and color == "yellow")

            score = 20 - dist  # Base score from distance
            if is_partner:
                score += 5  # Bonus for hitting partner

            # Bonus if target is near a hoop
            for hoop in court.hoops:
                hoop_dist = (other_ball.position - hoop.position).magnitude()
                if hoop_dist < 6:
                    score += 3

            if score > best_score:
                best_score = score
                best_target = other_ball

        if best_target:
            return (best_target.position,
                    self._power_for_distance(ball.position, best_target.position) * 1.03,
                    f"Shoot at {best_target.color}")

        return self._default_opening(ball, court)

    def _standard_tice(
        self,
        ball: Ball,
        court: Court
    ) -> Tuple[Vector2, float, str]:
        """Play a standard tice."""
        tice_y = 10
        tice_target = Vector2(2, tice_y)
        return (tice_target,
                self._power_for_distance(ball.position, tice_target),
                "Standard tice")

    def _default_opening(
        self,
        ball: Ball,
        court: Court
    ) -> Tuple[Vector2, float, str]:
        """Default opening when situation is unclear."""
        # Go toward hoop 1
        hoop1 = court.hoops[0]
        target = hoop1.position - hoop1.direction * 5
        return (target,
                self._power_for_distance(ball.position, target),
                "Approach hoop 1")

    def _power_for_distance(self, start: Vector2, end: Vector2) -> float:
        """Calculate power needed to reach target."""
        dist = (end - start).magnitude()
        friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY
        velocity = math.sqrt(2 * friction_decel * dist)
        return min(velocity * 1.15, config.MAX_SHOT_POWER)

    def is_opening_phase(self, balls_in_play: Dict[str, bool]) -> bool:
        """Check if we're still in the opening phase."""
        in_play_count = sum(1 for v in balls_in_play.values() if v)
        return in_play_count < 4
