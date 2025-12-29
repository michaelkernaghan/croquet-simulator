"""
Court model representing the croquet playing field.
"""
from typing import List, Tuple, Dict, Optional
import config
from models.ball import Vector2


class Hoop:
    """Represents a croquet hoop with position and direction."""

    def __init__(self, number: int, position: Tuple[float, float], direction: Tuple[float, float]):
        """
        Initialize a hoop.

        Args:
            number: Hoop number (1-6)
            position: Position in yards (x, y)
            direction: Unit vector indicating "forward" direction through hoop
        """
        self.number = number
        self.position = Vector2(position[0], position[1])
        self.direction = Vector2(direction[0], direction[1])
        self.width = config.HOOP_WIDTH_YARDS

    def get_pixel_position(self) -> Tuple[int, int]:
        """Convert yard position to screen pixel position."""
        px = int(self.position.x * config.YARDS_TO_PIXELS + config.COURT_OFFSET_X)
        py = int((config.COURT_HEIGHT_YARDS - self.position.y) * config.YARDS_TO_PIXELS + config.COURT_OFFSET_Y)
        return (px, py)

    def check_ball_run(self, ball_prev: Vector2, ball_curr: Vector2, ball_radius: float) -> bool:
        """
        Check if a ball has run through this hoop in the correct direction.

        Args:
            ball_prev: Ball's previous position
            ball_curr: Ball's current position
            ball_radius: Ball's radius

        Returns:
            True if the ball ran through the hoop correctly
        """
        # Calculate movement vector
        movement = ball_curr - ball_prev
        if movement.magnitude() < 0.001:
            return False

        # Check if ball crossed the hoop plane
        # Simplified: check if ball is now on the "forward" side of the hoop
        # and was on the "back" side before

        to_hoop_prev = self.position - ball_prev
        to_hoop_curr = self.position - ball_curr

        # Distance to hoop center
        dist_curr = to_hoop_curr.magnitude()

        # Must be close enough to the hoop
        if dist_curr > self.width * 2:
            return False

        # Check direction of approach matches hoop direction
        movement_normalized = movement.normalize()
        dot_product = movement_normalized.dot(self.direction)

        # Must be moving in roughly the correct direction (within 90 degrees)
        if dot_product < 0.3:
            return False

        # Check if ball crossed from back to front of hoop
        prev_side = to_hoop_prev.dot(self.direction)
        curr_side = to_hoop_curr.dot(self.direction)

        # Was behind (positive dot = approaching from correct side) and now in front (negative)
        return prev_side > -ball_radius and curr_side < ball_radius

    def __repr__(self):
        return f"Hoop({self.number}, pos={self.position})"


class Court:
    """
    Represents the croquet court with dimensions, hoops, and peg.
    """

    def __init__(self):
        """Initialize the court with standard dimensions."""
        self.width = config.COURT_WIDTH_YARDS
        self.height = config.COURT_HEIGHT_YARDS
        self.boundary_margin = config.BOUNDARY_MARGIN_YARDS

        # Create hoops
        self.hoops: List[Hoop] = []
        for hoop_data in config.HOOP_POSITIONS:
            hoop = Hoop(
                number=hoop_data["num"],
                position=hoop_data["pos"],
                direction=hoop_data["direction"]
            )
            self.hoops.append(hoop)

        # Peg position
        self.peg_position = Vector2(
            config.PEG_POSITION[0],
            config.PEG_POSITION[1]
        )

    def get_hoop(self, number: int) -> Optional[Hoop]:
        """
        Get a hoop by its number (1-6).

        Args:
            number: Hoop number (1-6)

        Returns:
            The hoop, or None if not found
        """
        for hoop in self.hoops:
            if hoop.number == number:
                return hoop
        return None

    def get_hoop_for_ball(self, hoops_run: int) -> Optional[Hoop]:
        """
        Get the next hoop a ball needs to run based on hoops completed.

        In Association Croquet, balls run hoops 1-6, then 1-back through rover
        (same physical hoops but in DIFFERENT ORDER and OPPOSITE direction).

        Hoop order:
        - Hoops 1-6: Run in standard order (1,2,3,4,5,6) with standard directions
        - Hoops 7-12: Second circuit with different order and reversed directions
            - Hoop 7 (1-back): Physical hoop 2 (NW), run NORTH
            - Hoop 8 (2-back): Physical hoop 1 (SW), run SOUTH
            - Hoop 9 (3-back): Physical hoop 4 (SE), run SOUTH
            - Hoop 10 (4-back): Physical hoop 3 (NE), run NORTH
            - Hoop 11 (penultimate): Physical hoop 6 (Center-N), run NORTH
            - Hoop 12 (rover): Physical hoop 5 (Center-S), run SOUTH

        Args:
            hoops_run: Number of hoops the ball has run (0-12)

        Returns:
            The next hoop to run (with correct direction), or None if all hoops completed
        """
        if hoops_run >= 12:
            return None  # Ball is a rover, needs to peg out

        # First circuit: hoops 1-6 in standard order and direction
        if hoops_run < 6:
            return self.get_hoop(hoops_run + 1)

        # Second circuit: use the AC_SECOND_CIRCUIT mapping
        # hoops_run 6 = next is hoop 7 (1-back), etc.
        circuit_info = config.AC_SECOND_CIRCUIT.get(hoops_run)
        if circuit_info is None:
            return None

        physical_hoop_num, direction = circuit_info
        base_hoop = self.get_hoop(physical_hoop_num)

        if base_hoop is None:
            return None

        # Create a hoop with the correct position and reversed direction
        back_hoop = Hoop(
            number=base_hoop.number,
            position=(base_hoop.position.x, base_hoop.position.y),
            direction=direction
        )
        return back_hoop

    def is_in_bounds(self, position: Vector2, radius: float = 0) -> bool:
        """
        Check if a position is within the court boundaries.

        Args:
            position: Position to check (in yards)
            radius: Optional radius to account for ball size

        Returns:
            True if position is within bounds
        """
        return (
            radius <= position.x <= self.width - radius and
            radius <= position.y <= self.height - radius
        )

    def clamp_to_bounds(self, position: Vector2, radius: float = 0) -> Vector2:
        """
        Clamp a position to be within court boundaries.

        Args:
            position: Position to clamp
            radius: Radius to account for ball size

        Returns:
            Clamped position
        """
        x = max(radius, min(self.width - radius, position.x))
        y = max(radius, min(self.height - radius, position.y))
        return Vector2(x, y)

    def get_boundary_collision(self, position: Vector2, velocity: Vector2, radius: float) -> Optional[Dict]:
        """
        Check if a ball at position with given velocity will hit a boundary.

        Args:
            position: Ball position
            velocity: Ball velocity
            radius: Ball radius

        Returns:
            Dict with 'normal' (reflection normal) and 'position' (corrected position),
            or None if no collision
        """
        collision = None

        # Left boundary
        if position.x - radius <= 0:
            collision = {
                'normal': Vector2(1, 0),
                'position': Vector2(radius, position.y)
            }
        # Right boundary
        elif position.x + radius >= self.width:
            collision = {
                'normal': Vector2(-1, 0),
                'position': Vector2(self.width - radius, position.y)
            }

        # Bottom boundary
        if position.y - radius <= 0:
            normal = Vector2(0, 1)
            if collision:
                # Corner collision - average the normals
                collision['normal'] = (collision['normal'] + normal).normalize()
            else:
                collision = {
                    'normal': normal,
                    'position': Vector2(position.x, radius)
                }
            if collision:
                collision['position'].y = radius

        # Top boundary
        elif position.y + radius >= self.height:
            normal = Vector2(0, -1)
            if collision:
                collision['normal'] = (collision['normal'] + normal).normalize()
            else:
                collision = {
                    'normal': normal,
                    'position': Vector2(position.x, self.height - radius)
                }
            if collision:
                collision['position'].y = self.height - radius

        return collision

    def get_peg_pixel_position(self) -> Tuple[int, int]:
        """Get the peg position in screen pixels."""
        px = int(self.peg_position.x * config.YARDS_TO_PIXELS + config.COURT_OFFSET_X)
        py = int((config.COURT_HEIGHT_YARDS - self.peg_position.y) * config.YARDS_TO_PIXELS + config.COURT_OFFSET_Y)
        return (px, py)

    def check_peg_hit(self, ball_pos: Vector2, ball_radius: float) -> bool:
        """
        Check if a ball has hit the peg.

        Args:
            ball_pos: Ball position
            ball_radius: Ball radius

        Returns:
            True if ball is touching the peg
        """
        # Peg is about 1.5 inches diameter in real life, but we make it
        # easier to hit in the simulation (0.3 yards = ~11 inches)
        peg_radius = 0.3  # Forgiving peg radius for simulation
        distance = (ball_pos - self.peg_position).magnitude()
        return distance <= ball_radius + peg_radius

    def check_peg_hit_path(self, start_pos: Vector2, end_pos: Vector2, ball_radius: float) -> bool:
        """
        Check if a ball's path crossed the peg.

        This checks if the ball passed through the peg during its movement,
        not just at its final position.

        Args:
            start_pos: Ball starting position
            end_pos: Ball ending position
            ball_radius: Ball radius

        Returns:
            True if ball path intersected the peg
        """
        peg_radius = 0.3  # Forgiving peg radius

        # Vector from start to end
        movement = end_pos - start_pos
        move_len = movement.magnitude()

        if move_len < 0.01:
            # Ball didn't move, check position
            return self.check_peg_hit(end_pos, ball_radius)

        move_dir = movement.normalize()

        # Project peg position onto the ball's path
        to_peg = self.peg_position - start_pos
        projection = to_peg.dot(move_dir)

        # Clamp to actual path
        projection = max(0, min(move_len, projection))

        # Find closest point on path to peg
        closest_point = start_pos + move_dir * projection
        min_dist = (closest_point - self.peg_position).magnitude()

        return min_dist <= ball_radius + peg_radius

    def _line_intersects_circle(
        self,
        line_start: Vector2,
        line_end: Vector2,
        circle_center: Vector2,
        circle_radius: float
    ) -> bool:
        """
        Check if a line segment intersects a circle.

        Args:
            line_start: Start point of line segment
            line_end: End point of line segment
            circle_center: Center of circle
            circle_radius: Radius of circle

        Returns:
            True if line segment intersects the circle
        """
        # Vector from start to end
        line_vec = line_end - line_start
        line_len = line_vec.magnitude()

        if line_len < 0.001:
            # Degenerate line, check point distance
            return (line_start - circle_center).magnitude() <= circle_radius

        line_dir = line_vec.normalize()

        # Project circle center onto line
        to_circle = circle_center - line_start
        projection = to_circle.dot(line_dir)

        # Clamp to line segment
        projection = max(0, min(line_len, projection))

        # Find closest point on line segment to circle center
        closest_point = line_start + line_dir * projection
        min_dist = (closest_point - circle_center).magnitude()

        return min_dist <= circle_radius

    def get_hoop_uprights(self, hoop: 'Hoop') -> Tuple[Vector2, Vector2]:
        """
        Get the positions of the two uprights of a hoop.

        Args:
            hoop: The hoop to get uprights for

        Returns:
            Tuple of (left_upright, right_upright) positions
        """
        # Hoop direction is the "forward" direction through the hoop
        # Uprights are perpendicular to this direction, on either side
        hoop_half_width = hoop.width / 2

        # Get perpendicular direction (rotate direction 90 degrees)
        perp = Vector2(-hoop.direction.y, hoop.direction.x)

        left_upright = hoop.position + perp * hoop_half_width
        right_upright = hoop.position - perp * hoop_half_width

        return (left_upright, right_upright)

    def is_wired(
        self,
        from_pos: Vector2,
        to_pos: Vector2,
        ball_radius: float = 0.1
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a line-of-sight from one position to another is blocked
        by a hoop upright or the peg.

        In croquet, a ball is "wired" when it cannot hit another ball because
        a hoop or peg is in the way.

        Args:
            from_pos: Starting position (striker ball)
            to_pos: Target position (target ball)
            ball_radius: Radius of balls for collision calculation

        Returns:
            Tuple of (is_wired, obstruction_type) where:
            - is_wired: True if line of sight is blocked
            - obstruction_type: "hoop_N" or "peg" or None
        """
        # Check for peg obstruction
        # Use a slightly larger radius for the wire check (accounts for ball width)
        peg_radius = 0.3 + ball_radius
        if self._line_intersects_circle(from_pos, to_pos, self.peg_position, peg_radius):
            return (True, "peg")

        # Check for hoop upright obstructions
        # Hoop uprights are thin posts, use small radius
        upright_radius = 0.05 + ball_radius  # ~2 inches for upright + ball radius

        for hoop in self.hoops:
            left_upright, right_upright = self.get_hoop_uprights(hoop)

            if self._line_intersects_circle(from_pos, to_pos, left_upright, upright_radius):
                return (True, f"hoop_{hoop.number}")

            if self._line_intersects_circle(from_pos, to_pos, right_upright, upright_radius):
                return (True, f"hoop_{hoop.number}")

        return (False, None)

    def get_wire_info(
        self,
        striker_pos: Vector2,
        all_ball_positions: Dict[str, Vector2],
        striker_color: str,
        ball_radius: float = 0.1
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Check wire status for a striker ball against all other balls.

        Args:
            striker_pos: Position of the striker ball
            all_ball_positions: Dict of ball color -> position
            striker_color: Color of the striker ball
            ball_radius: Ball radius

        Returns:
            Dict of target_color -> (is_wired, obstruction_type)
        """
        wire_info = {}

        for target_color, target_pos in all_ball_positions.items():
            if target_color != striker_color:
                is_blocked, obstruction = self.is_wired(
                    striker_pos, target_pos, ball_radius
                )
                wire_info[target_color] = (is_blocked, obstruction)

        return wire_info

    def __repr__(self):
        return f"Court({self.width}x{self.height} yards, {len(self.hoops)} hoops)"
