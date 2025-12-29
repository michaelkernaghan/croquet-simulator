"""
Rule Engine for Association Croquet.

Handles game rules including:
- Turn management
- Hoop running detection
- Roquet detection
- Deadness tracking
- Stroke allocation
"""
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field

import config
from models.ball import Ball, Vector2
from models.court import Court


class TurnState(Enum):
    """State of the current turn."""
    INITIAL = auto()          # First stroke of turn
    CONTINUATION = auto()     # Continuation stroke (after hoop or croquet)
    CROQUET_REQUIRED = auto() # Must take croquet stroke
    CROQUET_TAKEN = auto()    # Just took croquet, get continuation


@dataclass
class TurnInfo:
    """Information about the current turn."""
    ball_color: str
    strokes_remaining: int = 1
    state: TurnState = TurnState.INITIAL
    balls_roqueted_this_turn: Set[str] = field(default_factory=set)
    hoops_run_this_turn: int = 0
    just_roqueted: Optional[str] = None  # Ball just hit for croquet
    # Liveness: which balls can still be roqueted this turn
    # At turn start, all balls are live. After roqueting, that ball becomes dead.
    # Running a hoop makes all balls live again (clears deadness).
    live_balls: Set[str] = field(default_factory=set)


class RuleEngine:
    """
    Manages Association Croquet rules.

    Key rules:
    1. Turn order: Blue, Red, Black, Yellow (alternating sides)
    2. One stroke per turn unless you earn extras
    3. Running a hoop: Earn 1 continuation stroke, clear deadness
    4. Roquet: Hit another ball you're not dead on, earn croquet + continuation
    5. Deadness: After roqueting a ball, dead on it until you run your next hoop
    6. Croquet stroke: Place your ball touching roqueted ball, hit both
    """

    def __init__(self):
        """Initialize the rule engine."""
        # Deadness matrix: deadness[striker] = set of balls striker is dead on
        self.deadness: Dict[str, Set[str]] = {
            "blue": set(),
            "black": set(),
            "red": set(),
            "yellow": set(),
        }

        self.turn_info: Optional[TurnInfo] = None
        self.turn_number = 0

    def start_turn(self, ball_color: str) -> TurnInfo:
        """
        Start a new turn for the given ball.

        At the start of each turn, balls are "live" only if the striker
        is not dead on them. Deadness persists until the striker runs a hoop.

        Args:
            ball_color: Color of ball taking the turn

        Returns:
            TurnInfo for this turn
        """
        self.turn_number += 1

        # Live balls = all others minus those we're dead on
        all_colors = {"blue", "black", "red", "yellow"}
        all_colors.discard(ball_color)

        # Remove balls we're dead on from live balls
        dead_on = self.deadness.get(ball_color, set())
        live_balls = all_colors - dead_on

        self.turn_info = TurnInfo(
            ball_color=ball_color,
            strokes_remaining=1,
            state=TurnState.INITIAL,
            live_balls=live_balls
        )
        return self.turn_info

    def process_stroke_result(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        court: Court,
        collisions: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Process the result of a stroke.

        Args:
            striker: The ball that was hit
            all_balls: All balls on the court
            court: The court
            collisions: List of collision events that occurred

        Returns:
            Tuple of (turn_continues: bool, events: List[str])
        """
        events = []

        if self.turn_info is None:
            return False, ["Error: No turn in progress"]

        # Check for hoop run
        hoop_run = self._check_hoop_run(striker, court)
        if hoop_run:
            events.append(f"{striker.color.capitalize()} runs hoop {striker.hoops_run}!")
            self.turn_info.hoops_run_this_turn += 1

            # Running a hoop clears all deadness and makes all balls live again!
            self.deadness[striker.color].clear()
            all_colors = {"blue", "black", "red", "yellow"}
            all_colors.discard(striker.color)
            self.turn_info.live_balls = all_colors

            # Earn continuation stroke
            self.turn_info.strokes_remaining += 1
            self.turn_info.state = TurnState.CONTINUATION

        # Check for roquets (ball-ball collisions)
        roquet_ball = self._check_roquet(striker, collisions)
        if roquet_ball:
            # Can only roquet balls that are "live" (not dead on)
            if roquet_ball in self.turn_info.live_balls:
                events.append(f"{striker.color.capitalize()} roquets {roquet_ball}!")
                # Ball is no longer live until we run a hoop
                self.turn_info.live_balls.discard(roquet_ball)
                self.deadness[striker.color].add(roquet_ball)
                self.turn_info.balls_roqueted_this_turn.add(roquet_ball)
                # Set up for croquet stroke
                self.turn_info.just_roqueted = roquet_ball
                self.turn_info.state = TurnState.CROQUET_REQUIRED
                # Earn croquet stroke + continuation (2 strokes)
                self.turn_info.strokes_remaining += 2
            else:
                events.append(f"{striker.color.capitalize()} hits {roquet_ball} (dead - no roquet)")

        # Check for peg out - use path-based check so we don't miss the peg
        if striker.is_rover and not striker.has_pegged_out:
            # Check both current position and path from shot start
            hit_peg = court.check_peg_hit(striker.position, striker.radius)
            if not hit_peg:
                # Also check if ball path crossed the peg
                hit_peg = court.check_peg_hit_path(
                    striker.shot_start_position,
                    striker.position,
                    striker.radius
                )
            if hit_peg:
                striker.peg_out()
                events.append(f"{striker.color.capitalize()} pegs out!")

        # Decrement strokes
        self.turn_info.strokes_remaining -= 1

        # Check if turn continues
        turn_continues = self.turn_info.strokes_remaining > 0

        if not turn_continues:
            self.turn_info.state = TurnState.INITIAL

        return turn_continues, events

    def _check_hoop_run(self, ball: Ball, court: Court, debug: bool = False) -> bool:
        """
        Check if a ball has run its next hoop.

        Uses the ball's shot_start_position and current position to detect
        if it crossed through the hoop in the correct direction during this shot.

        The ball must:
        1. Pass close to the hoop center (within ~0.5 yards)
        2. Cross through the hoop plane
        3. Be moving in roughly the correct direction (within ~45 degrees)
        4. Pass through the actual hoop gap (perpendicular offset < 0.5 yards)
        """
        target_hoop = court.get_hoop_for_ball(ball.hoops_run)
        if target_hoop is None:
            return False

        hoop_pos = target_hoop.position
        start_pos = ball.shot_start_position

        # Check if ball path came near the hoop
        # Calculate minimum distance from ball's path to the hoop
        movement = ball.position - start_pos
        move_len = movement.magnitude()

        if move_len < 0.1:
            return False

        move_dir = movement.normalize()

        # Project hoop position onto the ball's path
        to_hoop = hoop_pos - start_pos
        projection = to_hoop.dot(move_dir)

        # Clamp projection to the actual path length
        projection = max(0, min(move_len, projection))

        # Find closest point on path to hoop
        closest_point = start_pos + move_dir * projection
        min_dist_to_hoop = (closest_point - hoop_pos).magnitude()

        # Ball must have passed within 0.75 yards of hoop center
        # (Real hoops are ~4 inches wide, but we're a bit forgiving for simulation)
        if min_dist_to_hoop > 0.75:
            if debug:
                print(f"  [HOOP DEBUG] {ball.color} too far from hoop {target_hoop.number}: {min_dist_to_hoop:.2f} yards")
            return False

        # Get the required direction for this hoop
        # The court.get_hoop_for_ball already returns the correct direction
        # (standard for first circuit, reversed for second circuit)
        required_dir = target_hoop.direction

        # Must be moving in roughly the correct direction (within ~45 degrees)
        dot = move_dir.dot(required_dir)

        if debug or dot < 0.7:
            # Log direction info for debugging wrong-direction attempts
            dir_name = "NORTH" if required_dir.y > 0 else "SOUTH" if required_dir.y < 0 else "EAST" if required_dir.x > 0 else "WEST"
            move_name = "NORTH" if move_dir.y > 0.5 else "SOUTH" if move_dir.y < -0.5 else "EAST" if move_dir.x > 0.5 else "WEST" if move_dir.x < -0.5 else "DIAGONAL"
            if dot < 0.7:
                print(f"  [HOOP DEBUG] {ball.color} WRONG DIRECTION for hoop {target_hoop.number}!")
                print(f"    Required: {dir_name} {required_dir}, Moving: {move_name} {move_dir}")
                print(f"    Dot product: {dot:.2f} (need > 0.7)")
                print(f"    Start: {start_pos}, End: {ball.position}")

        if dot < 0.7:
            return False

        # Check if ball crossed the hoop plane
        # Distance from start position to hoop plane (along hoop direction)
        start_dist = to_hoop.dot(required_dir)

        # Distance from current position to hoop plane
        curr_to_hoop = hoop_pos - ball.position
        curr_dist = curr_to_hoop.dot(required_dir)

        # Ball crossed the plane if:
        # - start_dist > 0: started on approach side (in front of hoop)
        # - curr_dist <= 0: ended past the hoop (or at least at it)
        crossed_plane = start_dist > 0 and curr_dist <= 0

        if crossed_plane:
            # Check if ball passed within hoop width at the crossing point
            # The crossing point is where the ball was when it crossed the hoop plane
            perp_dir = Vector2(-required_dir.y, required_dir.x)
            perp_dist = abs((closest_point - hoop_pos).dot(perp_dir))

            # Ball must pass through the actual hoop gap
            # Real gap is ~4 inches (0.1 yards), we allow 0.5 yards for simulation
            if perp_dist < 0.5:
                # SUCCESS - log the hoop run
                dir_name = "NORTH" if required_dir.y > 0 else "SOUTH" if required_dir.y < 0 else "?"
                print(f"  [HOOP RUN] {ball.color} ran hoop {target_hoop.number} (direction: {dir_name})")
                print(f"    From {start_pos} to {ball.position}")
                print(f"    Dot: {dot:.2f}, Perp dist: {perp_dist:.2f}")
                ball.run_hoop()
                return True
            elif debug:
                print(f"  [HOOP DEBUG] {ball.color} missed hoop gap: perp_dist={perp_dist:.2f}")
        elif debug:
            print(f"  [HOOP DEBUG] {ball.color} didn't cross plane: start_dist={start_dist:.2f}, curr_dist={curr_dist:.2f}")

        return False

    def _check_roquet(self, striker: Ball, collisions: List[Dict]) -> Optional[str]:
        """
        Check if the striker ball roqueted another ball.

        Returns the color of the roqueted ball, or None.
        """
        for collision in collisions:
            if collision['type'] == 'ball_collision':
                ball1 = collision['ball1']
                ball2 = collision['ball2']

                # Find which ball is the striker
                if ball1 == striker.color:
                    return ball2
                elif ball2 == striker.color:
                    return ball1

        return None

    def get_live_balls(self, striker_color: str) -> List[str]:
        """
        Get list of balls that the striker can roquet (not dead on).

        Args:
            striker_color: Color of the striking ball

        Returns:
            List of ball colors that can be roqueted
        """
        all_colors = {"blue", "black", "red", "yellow"}
        all_colors.discard(striker_color)  # Can't roquet yourself

        dead_on = self.deadness[striker_color]
        return [c for c in all_colors if c not in dead_on]

    def is_dead_on(self, striker_color: str, target_color: str) -> bool:
        """Check if striker is dead on target."""
        return target_color in self.deadness[striker_color]

    def get_deadness_display(self) -> Dict[str, List[str]]:
        """Get deadness info for display."""
        return {color: list(dead) for color, dead in self.deadness.items()}

    def setup_croquet_stroke(
        self,
        striker: Ball,
        roqueted: Ball
    ) -> Vector2:
        """
        Set up for a croquet stroke by placing the striker ball.

        In croquet, you place your ball in contact with the roqueted ball
        and then hit your ball, causing both to move.

        For simplicity, we place the striker behind the roqueted ball
        in the direction of the next hoop.

        Returns:
            New position for striker ball
        """
        # For now, just place striker next to roqueted ball
        # A smarter AI would choose the placement strategically
        offset = Vector2(0.15, 0)  # Place to the side
        return roqueted.position + offset
