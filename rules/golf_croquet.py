"""
Golf Croquet Rules Engine.

Golf Croquet is simpler than Association Croquet:
- All balls contest the same hoop
- First ball through scores 1 point for their side
- Then all balls move to the next hoop
- No roquets, no croquet strokes, no deadness
- Each player gets one stroke per turn
- First to 7 points wins
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import config
from models.ball import Ball, Vector2
from models.court import Court, Hoop


@dataclass
class GolfCroquetState:
    """Current state of a Golf Croquet game."""
    current_hoop: int = 1          # Which hoop everyone is playing for (1-6)
    hoop_sequence_index: int = 0   # Index in HOOP_SEQUENCE
    blue_black_score: int = 0
    red_yellow_score: int = 0


class GolfCroquetRules:
    """
    Manages Golf Croquet game rules.

    Key rules:
    1. All four balls contest the same hoop
    2. Turns alternate: Blue, Red, Black, Yellow
    3. One stroke per turn (no extras)
    4. First ball through the hoop scores 1 point for their side
    5. After a hoop is scored, all balls play for the next hoop
    6. Sequence: 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 (can play twice)
    7. First side to 7 points wins
    """

    def __init__(self):
        self.state = GolfCroquetState()
        self.hoop_sequence = config.HOOP_SEQUENCE

    def get_current_hoop_number(self) -> int:
        """Get the hoop number everyone is playing for."""
        if self.state.hoop_sequence_index < len(self.hoop_sequence):
            return self.hoop_sequence[self.state.hoop_sequence_index]
        return 6  # Default to last hoop

    def check_hoop_scored(
        self,
        ball: Ball,
        court: Court,
        ball_prev_pos: Vector2
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a ball scored the current hoop.

        Args:
            ball: The ball that just moved
            court: The court
            ball_prev_pos: Ball's position before the shot

        Returns:
            Tuple of (scored: bool, event_message: str or None)
        """
        current_hoop_num = self.get_current_hoop_number()
        hoop = court.get_hoop(current_hoop_num)

        if hoop is None:
            return False, None

        # Check if ball went through the hoop
        if self._ball_ran_hoop(ball, hoop, ball_prev_pos):
            # Score the point
            if ball.color in ["blue", "black"]:
                self.state.blue_black_score += 1
                team = "Blue/Black"
            else:
                self.state.red_yellow_score += 1
                team = "Red/Yellow"

            # Move to next hoop
            self.state.hoop_sequence_index += 1

            event = f"{ball.color.capitalize()} scores hoop {current_hoop_num}! ({team})"
            return True, event

        return False, None

    def _ball_ran_hoop(self, ball: Ball, hoop: Hoop, prev_pos: Vector2) -> bool:
        """
        Check if a ball ran through a hoop in the correct direction.

        Uses line segment intersection to detect if ball path crossed
        through the hoop opening.
        """
        curr_pos = ball.position

        # Ball must be close to hoop
        dist_to_hoop = (curr_pos - hoop.position).magnitude()
        if dist_to_hoop > 2.0:  # More than 2 yards away
            return False

        # Check movement direction
        movement = curr_pos - prev_pos
        if movement.magnitude() < 0.1:
            return False

        move_dir = movement.normalize()

        # Must be moving in roughly the hoop's direction
        dot = move_dir.dot(hoop.direction)
        if dot < 0.5:  # Must be within ~60 degrees of correct direction
            return False

        # Check if ball crossed the hoop plane
        # The hoop plane is perpendicular to the hoop direction, passing through hoop position

        # Distance from previous position to hoop plane (along hoop direction)
        prev_to_hoop = hoop.position - prev_pos
        prev_dist = prev_to_hoop.dot(hoop.direction)

        # Distance from current position to hoop plane
        curr_to_hoop = hoop.position - curr_pos
        curr_dist = curr_to_hoop.dot(hoop.direction)

        # Ball crossed the plane if it went from positive to negative (or close to zero)
        # while passing through the hoop width
        crossed_plane = prev_dist > 0 and curr_dist <= 0.5

        if crossed_plane:
            # Check if ball was within hoop width when crossing
            # Perpendicular distance to hoop center line
            perp_dir = Vector2(-hoop.direction.y, hoop.direction.x)
            perp_dist = abs((curr_pos - hoop.position).dot(perp_dir))

            if perp_dist < 1.0:  # Within reasonable distance of hoop center
                return True

        return False

    def is_game_over(self) -> bool:
        """Check if the game is over (someone reached 7 points)."""
        return (self.state.blue_black_score >= config.WINNING_SCORE or
                self.state.red_yellow_score >= config.WINNING_SCORE)

    def get_winner(self) -> Optional[str]:
        """Get the winning team, or None if game not over."""
        if self.state.blue_black_score >= config.WINNING_SCORE:
            return "Blue/Black"
        elif self.state.red_yellow_score >= config.WINNING_SCORE:
            return "Red/Yellow"
        return None

    def get_scores(self) -> Dict[str, int]:
        """Get current scores."""
        return {
            "Blue/Black": self.state.blue_black_score,
            "Red/Yellow": self.state.red_yellow_score,
        }
