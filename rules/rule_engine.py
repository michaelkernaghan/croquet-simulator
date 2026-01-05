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


@dataclass
class LiftEntitlement:
    """
    Tracks lift entitlements under Advanced Play rules.

    AC Laws: After a ball runs 1-back (hoop 7) or 4-back (hoop 10),
    the opponent is entitled to lift one of their balls to either baulk line.
    """
    available: bool = False
    for_side: Optional[str] = None  # "blue_black" or "red_yellow"
    triggered_by_hoop: Optional[int] = None  # 7 or 10
    triggered_by_ball: Optional[str] = None  # Which ball ran the hoop


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

    # Hoops that trigger lift entitlements under Advanced Play
    LIFT_HOOPS = {7, 10}  # 1-back and 4-back

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

        # Advanced Play: Lift entitlements
        self.lift_entitlement = LiftEntitlement()

        # Track which balls have run lift hoops (for wiring lift rule)
        self.lift_hoops_run: Dict[str, Set[int]] = {
            "blue": set(),
            "black": set(),
            "red": set(),
            "yellow": set(),
        }

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
            hoop_just_run = striker.hoops_run  # Ball already incremented by run_hoop()
            events.append(f"{striker.color.capitalize()} runs hoop {hoop_just_run}!")
            self.turn_info.hoops_run_this_turn += 1

            # Running a hoop clears all deadness and makes all balls live again!
            self.deadness[striker.color].clear()
            all_colors = {"blue", "black", "red", "yellow"}
            all_colors.discard(striker.color)
            self.turn_info.live_balls = all_colors

            # Earn continuation stroke
            self.turn_info.strokes_remaining += 1
            self.turn_info.state = TurnState.CONTINUATION

            # ADVANCED PLAY: Check for lift entitlement (1-back or 4-back)
            if hoop_just_run in self.LIFT_HOOPS:
                self._trigger_lift_entitlement(striker.color, hoop_just_run)
                self.lift_hoops_run[striker.color].add(hoop_just_run)
                events.append(f"  [ADVANCED PLAY] Opponent earns lift at baulk!")

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
                # ROVER PEG-OUT CONSTRAINT: Check if pegging out is allowed
                can_peg_out, reason = self.can_peg_out(striker, all_balls)
                if can_peg_out:
                    striker.peg_out()
                    events.append(f"{striker.color.capitalize()} pegs out!")
                else:
                    events.append(f"{striker.color.capitalize()} hits peg but cannot peg out: {reason}")

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
            if dot < 0.7 and debug:
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
                # SUCCESS - hoop run (debug output removed for cleaner training)
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

    def _trigger_lift_entitlement(self, ball_color: str, hoop_run: int) -> None:
        """
        Trigger a lift entitlement for the opponent under Advanced Play.

        When a ball runs 1-back (hoop 7) or 4-back (hoop 10), the opponent
        may lift one of their balls to either baulk line at the start of
        their next turn.

        Args:
            ball_color: Color of the ball that ran the lift hoop
            hoop_run: Which hoop was run (7 or 10)
        """
        # Determine opponent side
        if ball_color in ["blue", "black"]:
            opponent_side = "red_yellow"
        else:
            opponent_side = "blue_black"

        self.lift_entitlement = LiftEntitlement(
            available=True,
            for_side=opponent_side,
            triggered_by_hoop=hoop_run,
            triggered_by_ball=ball_color
        )
        # Lift earned (debug output removed for cleaner training)

    def check_lift_available(self, side: str) -> bool:
        """
        Check if a lift is available for the given side.

        Args:
            side: "blue_black" or "red_yellow"

        Returns:
            True if the side can take a lift
        """
        return self.lift_entitlement.available and self.lift_entitlement.for_side == side

    def use_lift(self, ball_color: str, baulk_line: str, court) -> Vector2:
        """
        Use a lift entitlement to place a ball on a baulk line.

        Args:
            ball_color: Color of ball to lift
            baulk_line: "A" (south) or "B" (north)
            court: The court for getting baulk positions

        Returns:
            New position for the ball
        """
        if not self.lift_entitlement.available:
            raise ValueError("No lift available")

        # Verify ball is on the correct side
        ball_side = "blue_black" if ball_color in ["blue", "black"] else "red_yellow"
        if ball_side != self.lift_entitlement.for_side:
            raise ValueError(f"Lift is for {self.lift_entitlement.for_side}, not {ball_side}")

        # Get baulk position
        if baulk_line == "A":
            # A-baulk: south side, y = 1 yard
            position = Vector2(court.width / 2, 1)
        else:
            # B-baulk: north side, y = court.height - 1
            position = Vector2(court.width / 2, court.height - 1)

        # Clear the lift entitlement
        self.lift_entitlement = LiftEntitlement()
        # Lift used (debug output removed for cleaner training)

        return position

    def clear_lift_entitlement(self) -> None:
        """Clear any pending lift entitlement (if not used)."""
        # Lift expired (debug output removed for cleaner training)
        self.lift_entitlement = LiftEntitlement()

    def check_wiring_lift(
        self,
        ball: Ball,
        all_balls: Dict[str, Ball],
        court: Court
    ) -> bool:
        """
        Check if a ball is wired from ALL other balls (grants opponent a lift).

        Under AC Laws, if the striker ball is wired from all other balls by
        hoops or peg at the start of their turn, they may claim a lift.

        Args:
            ball: The striker ball
            all_balls: All balls on court
            court: The court (for wire detection)

        Returns:
            True if ball is wired from all others
        """
        other_balls = [b for c, b in all_balls.items()
                       if c != ball.color and not b.has_pegged_out]

        if not other_balls:
            return False

        # Check if wired from EVERY other ball
        for other in other_balls:
            is_wired, _ = court.is_wired(ball.position, other.position, ball.radius)
            if not is_wired:
                return False  # Can hit at least one ball

        return True  # Wired from all balls

    def get_wired_balls(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball],
        court: Court
    ) -> Set[str]:
        """
        Get the set of balls that the striker is wired from.

        Args:
            striker: The striker ball
            all_balls: All balls on court
            court: The court (for wire detection)

        Returns:
            Set of ball colors that striker is wired from
        """
        wired_from = set()

        for color, ball in all_balls.items():
            if color == striker.color or ball.has_pegged_out:
                continue

            is_wired, obstruction = court.is_wired(
                striker.position, ball.position, striker.radius
            )
            if is_wired:
                wired_from.add(color)

        return wired_from

    def can_legally_roquet(
        self,
        striker: Ball,
        target: Ball,
        court: Court
    ) -> Tuple[bool, str]:
        """
        Check if the striker can legally roquet the target ball.

        A roquet is illegal if:
        1. Striker is dead on target (already roqueted this turn cycle)
        2. Striker is wired from target (hoop/peg blocks the shot)

        Args:
            striker: The striking ball
            target: The ball to roquet
            court: The court

        Returns:
            Tuple of (can_roquet, reason)
        """
        # Check deadness
        if target.color in self.deadness.get(striker.color, set()):
            return (False, f"Dead on {target.color}")

        # Check wiring
        is_wired, obstruction = court.is_wired(
            striker.position, target.position, striker.radius
        )
        if is_wired:
            return (False, f"Wired from {target.color} by {obstruction}")

        return (True, "Legal roquet")

    def can_peg_out(
        self,
        striker: Ball,
        all_balls: Dict[str, Ball]
    ) -> Tuple[bool, str]:
        """
        Check if a rover ball is allowed to peg out.

        AC Laws constraint: Cannot peg out if it would leave opponent with
        a rover ball when you have a non-rover ball. This prevents unfair
        endgame situations where opponent has an insurmountable advantage.

        The rule in Association Croquet:
        - Both balls of a side must be rovers before either can peg out
        - EXCEPTION: You CAN peg out if:
          a) Partner is also a rover
          b) Partner is at penult (close enough)
          c) You're pegging out an opponent's ball in the same stroke

        Args:
            striker: The ball attempting to peg out
            all_balls: All balls on the court

        Returns:
            Tuple of (can_peg_out, reason)
        """
        if not striker.is_rover:
            return (False, "Not a rover")

        # Determine partner ball
        partner_color = {
            "blue": "black",
            "black": "blue",
            "red": "yellow",
            "yellow": "red"
        }.get(striker.color)

        if not partner_color or partner_color not in all_balls:
            return (True, "No partner ball")  # Edge case

        partner = all_balls[partner_color]

        # If partner already pegged out, always allowed
        if partner.has_pegged_out:
            return (True, "Partner already out")

        # If partner is also a rover, allowed
        if partner.is_rover:
            return (True, "Partner is rover")

        # If partner is at penult (hoop 11, just before rover), allowed
        # This is a common tactical situation in triple peels
        if partner.hoops_run >= 10:  # Penult or just finished penult
            return (True, "Partner at penult or beyond")

        # Check opponent situation
        opponent_colors = (
            ["red", "yellow"] if striker.color in ["blue", "black"]
            else ["blue", "black"]
        )

        opponent_rovers = 0
        for opp_color in opponent_colors:
            if opp_color in all_balls and not all_balls[opp_color].has_pegged_out:
                if all_balls[opp_color].is_rover:
                    opponent_rovers += 1

        # The strategic constraint: don't leave opponents with rover vs your non-rover
        # This is enforced as a rule in some competitions
        if opponent_rovers > 0 and partner.hoops_run < 6:
            return (False, f"Would leave {partner_color} (at hoop {partner.hoops_run + 1}) vs opponent rover")

        # Generally allowed if partner is making progress
        return (True, "Partner making progress")

    def check_peel(
        self,
        croqueted_ball: Ball,
        court: Court,
        start_position: Vector2
    ) -> Optional[Dict]:
        """
        Check if a peel occurred during a croquet stroke.

        A PEEL is when the croqueted ball runs its hoop during a croquet stroke.
        This is an advanced technique used in triple peels (TP) where the
        striker peels their partner ball through 4-back, penult, and rover
        while making their own break.

        Args:
            croqueted_ball: The ball that was croqueted
            court: The court
            start_position: Where the ball started before the stroke

        Returns:
            Dict with peel info if peel occurred, None otherwise
        """
        # Get the ball's target hoop
        target_hoop = court.get_hoop_for_ball(croqueted_ball.hoops_run)
        if not target_hoop:
            return None  # Ball is a rover, can't be peeled through a hoop

        # Check if the ball ran through its hoop
        # We need to see if the ball crossed from the approach side to the run side
        hoop_pos = target_hoop.position
        hoop_dir = target_hoop.direction

        # Calculate positions relative to hoop
        start_to_hoop = start_position - hoop_pos
        end_to_hoop = croqueted_ball.position - hoop_pos

        # Check if ball crossed the hoop plane
        start_side = start_to_hoop.dot(hoop_dir)
        end_side = end_to_hoop.dot(hoop_dir)

        # Ball must have been on approach side (negative) and ended on run side (positive)
        if start_side < 0 and end_side > 0:
            # Check if ball passed through the hoop opening (not around it)
            # Project position onto hoop plane and check if within hoop width
            lateral_dist = abs(start_to_hoop.x * hoop_dir.y - start_to_hoop.y * hoop_dir.x)

            if lateral_dist < target_hoop.width / 2 + croqueted_ball.radius:
                # Peel successful!
                hoop_num = croqueted_ball.hoops_run + 1
                hoop_name = self._get_hoop_name(hoop_num)

                return {
                    "ball": croqueted_ball.color,
                    "hoop_num": hoop_num,
                    "hoop_name": hoop_name,
                    "is_partner_peel": True  # Will be determined by caller
                }

        return None

    def _get_hoop_name(self, hoop_num: int) -> str:
        """Get the AC name for a hoop number."""
        hoop_names = {
            1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
            7: "1-back", 8: "2-back", 9: "3-back", 10: "4-back",
            11: "penult", 12: "rover"
        }
        return hoop_names.get(hoop_num, str(hoop_num))

    def process_peel(
        self,
        croqueted_ball: Ball,
        peel_info: Dict,
        court: Court
    ) -> str:
        """
        Process a successful peel by awarding the hoop to the croqueted ball.

        Args:
            croqueted_ball: The ball that was peeled
            peel_info: Info about the peel from check_peel()
            court: The court

        Returns:
            Description of the peel
        """
        # Award the hoop to the peeled ball
        croqueted_ball.run_hoop(peel_info["hoop_num"])

        # Generate description
        description = f"PEEL! {croqueted_ball.color.capitalize()} peeled through {peel_info['hoop_name']}"

        # Check if this completes a triple peel milestone
        if peel_info["hoop_num"] == 10:  # 4-back
            description += " (TP: 4-back complete)"
        elif peel_info["hoop_num"] == 11:  # penult
            description += " (TP: penult complete)"
        elif peel_info["hoop_num"] == 12:  # rover
            description += " (TP: rover complete - TRIPLE PEEL!)"

        return description

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
