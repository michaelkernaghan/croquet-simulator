"""
Position Evaluator - Evaluates board positions for croquet.

Uses a combination of handcrafted features and learnable weights
to evaluate how good a position is for a given ball/side.
"""
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from models.ball import Ball, Vector2
from models.court import Court


@dataclass
class PositionFeatures:
    """Features extracted from a board position."""
    # Hoop progress
    hoops_run: int = 0
    distance_to_next_hoop: float = 0.0
    approach_angle_quality: float = 0.0  # 0-1, how good the approach angle is

    # Ball relationships
    distance_to_partner: float = 0.0
    distance_to_opponents: List[float] = field(default_factory=list)
    can_roquet_count: int = 0  # Number of balls we can roquet (not dead on and not wired)

    # Positional quality
    is_in_good_position: bool = False  # In front of hoop with good angle
    is_wired: bool = False  # Blocked from at least one ball by hoop/peg
    distance_from_boundary: float = 0.0

    # Wire details (for strategic planning)
    wired_from: Dict[str, str] = field(default_factory=dict)  # ball_color -> obstruction_type

    # Strategic
    threatens_hoop: bool = False  # Can run hoop next shot
    partner_threatens_hoop: bool = False
    opponent_threatens_hoop: bool = False


class PositionEvaluator:
    """
    Evaluates board positions using weighted features.

    The weights can be learned through self-play.
    """

    def __init__(self):
        """Initialize with default weights."""
        # Weights for different features (can be learned)
        self.weights = {
            # Hoop progress is very important
            'hoops_run': 10.0,
            'distance_to_next_hoop': -0.5,  # Closer is better
            'approach_angle_quality': 3.0,

            # Ball relationships
            'distance_to_partner': -0.2,  # Closer partner is slightly better
            'can_roquet_count': 2.0,  # More roquet options is good

            # Positional quality
            'is_in_good_position': 5.0,
            'is_wired': -3.0,
            'distance_from_boundary': 0.1,  # Being away from boundary is good

            # Strategic threats
            'threatens_hoop': 8.0,
            'partner_threatens_hoop': 4.0,
            'opponent_threatens_hoop': -6.0,
        }

    def extract_features(
        self,
        ball: Ball,
        all_balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set]
    ) -> PositionFeatures:
        """
        Extract features for a ball's position.

        Args:
            ball: The ball to evaluate
            all_balls: All balls on the court
            court: The court
            deadness: Deadness tracking dict

        Returns:
            PositionFeatures for this position
        """
        features = PositionFeatures()

        # Hoop progress
        features.hoops_run = ball.hoops_run

        # Distance and angle to next hoop
        target_hoop = court.get_hoop_for_ball(ball.hoops_run)
        if target_hoop:
            to_hoop = target_hoop.position - ball.position
            features.distance_to_next_hoop = to_hoop.magnitude()

            # Approach angle quality (0-1)
            if features.distance_to_next_hoop > 0.5:
                approach_dir = to_hoop.normalize()
                dot = approach_dir.dot(target_hoop.direction)
                features.approach_angle_quality = max(0, dot)
                features.is_in_good_position = dot > 0.5 and features.distance_to_next_hoop < 5
                features.threatens_hoop = dot > 0.7 and features.distance_to_next_hoop < 4

        # Partner and opponent distances
        partner_color = self._get_partner(ball.color)
        opponent_colors = self._get_opponents(ball.color)

        if partner_color and partner_color in all_balls:
            partner = all_balls[partner_color]
            features.distance_to_partner = (ball.position - partner.position).magnitude()

            # Check if partner threatens their hoop
            partner_hoop = court.get_hoop_for_ball(partner.hoops_run)
            if partner_hoop:
                to_hoop = partner_hoop.position - partner.position
                if to_hoop.magnitude() < 4:
                    dot = to_hoop.normalize().dot(partner_hoop.direction)
                    features.partner_threatens_hoop = dot > 0.7

        for opp_color in opponent_colors:
            if opp_color in all_balls:
                opp = all_balls[opp_color]
                dist = (ball.position - opp.position).magnitude()
                features.distance_to_opponents.append(dist)

                # Check if opponent threatens their hoop
                opp_hoop = court.get_hoop_for_ball(opp.hoops_run)
                if opp_hoop:
                    to_hoop = opp_hoop.position - opp.position
                    if to_hoop.magnitude() < 4:
                        dot = to_hoop.normalize().dot(opp_hoop.direction)
                        if dot > 0.7:
                            features.opponent_threatens_hoop = True

        # Roquet options (count balls we can hit, accounting for deadness AND wiring)
        dead_on = deadness.get(ball.color, set())
        roquet_count = 0
        any_wired = False
        wired_from = {}

        for c, other_ball in all_balls.items():
            if c != ball.color and c not in dead_on:
                # Check if we're wired from this ball
                is_wired, obstruction = court.is_wired(
                    ball.position, other_ball.position, ball.radius
                )
                if not is_wired:
                    roquet_count += 1
                else:
                    any_wired = True
                    wired_from[c] = obstruction

        features.can_roquet_count = roquet_count
        features.is_wired = any_wired
        features.wired_from = wired_from

        # Boundary distance
        features.distance_from_boundary = min(
            ball.position.x,
            ball.position.y,
            court.width - ball.position.x,
            court.height - ball.position.y
        )

        return features

    def evaluate(
        self,
        ball: Ball,
        all_balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        hoop_modifiers: Optional[Dict[int, Dict[str, float]]] = None
    ) -> float:
        """
        Evaluate a position for a ball.

        Args:
            ball: The ball to evaluate
            all_balls: All balls on the court
            court: The court
            deadness: Deadness tracking
            hoop_modifiers: Optional per-hoop weight modifiers (hoop_num -> {weight_name: modifier})

        Returns:
            Evaluation score (higher is better)
        """
        features = self.extract_features(ball, all_balls, court, deadness)

        # Determine which hoop this ball is targeting (1-12)
        target_hoop_num = ball.hoops_run + 1 if ball.hoops_run < 12 else 0

        # Get hoop-specific modifiers if available
        modifiers = {}
        if hoop_modifiers and target_hoop_num in hoop_modifiers:
            modifiers = hoop_modifiers[target_hoop_num]

        def get_adjusted_weight(weight_name: str) -> float:
            """Get weight adjusted by per-hoop modifier: base * (1 + modifier)"""
            base_weight = self.weights[weight_name]
            modifier = modifiers.get(weight_name, 0.0)
            return base_weight * (1.0 + modifier)

        score = 0.0
        score += get_adjusted_weight('hoops_run') * features.hoops_run
        score += get_adjusted_weight('distance_to_next_hoop') * features.distance_to_next_hoop
        score += get_adjusted_weight('approach_angle_quality') * features.approach_angle_quality
        score += get_adjusted_weight('distance_to_partner') * features.distance_to_partner
        score += get_adjusted_weight('can_roquet_count') * features.can_roquet_count
        score += get_adjusted_weight('is_in_good_position') * (1 if features.is_in_good_position else 0)
        score += get_adjusted_weight('is_wired') * (1 if features.is_wired else 0)
        score += get_adjusted_weight('distance_from_boundary') * features.distance_from_boundary
        score += get_adjusted_weight('threatens_hoop') * (1 if features.threatens_hoop else 0)
        score += get_adjusted_weight('partner_threatens_hoop') * (1 if features.partner_threatens_hoop else 0)
        score += get_adjusted_weight('opponent_threatens_hoop') * (1 if features.opponent_threatens_hoop else 0)

        return score

    def evaluate_side(
        self,
        side: str,
        all_balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        hoop_modifiers: Optional[Dict[int, Dict[str, float]]] = None
    ) -> float:
        """
        Evaluate position for an entire side (two balls).

        Args:
            side: "blue_black" or "red_yellow"
            all_balls: All balls on the court
            court: The court
            deadness: Deadness tracking
            hoop_modifiers: Optional per-hoop weight modifiers

        Returns:
            Combined evaluation score for the side
        """
        if side == "blue_black":
            colors = ["blue", "black"]
        else:
            colors = ["red", "yellow"]

        total = 0.0
        for color in colors:
            if color in all_balls:
                total += self.evaluate(all_balls[color], all_balls, court, deadness, hoop_modifiers)

        return total

    def update_weights(self, weight_updates: Dict[str, float]):
        """
        Update weights based on learning.

        Args:
            weight_updates: Dictionary of weight adjustments
        """
        for key, delta in weight_updates.items():
            if key in self.weights:
                self.weights[key] += delta

    def _get_partner(self, color: str) -> Optional[str]:
        """Get partner ball color."""
        partners = {
            "blue": "black",
            "black": "blue",
            "red": "yellow",
            "yellow": "red"
        }
        return partners.get(color)

    def _get_opponents(self, color: str) -> List[str]:
        """Get opponent ball colors."""
        if color in ["blue", "black"]:
            return ["red", "yellow"]
        return ["blue", "black"]
