"""
Physics Engine - Handles ball movement, friction, and collisions.
"""
from typing import Dict, List, Optional, Tuple, Callable
import random
import config
from models.ball import Ball, Vector2
from models.court import Court


class PhysicsEngine:
    """
    Manages physics simulation for all balls.

    Handles:
    - Ball movement with velocity
    - Friction deceleration
    - Ball-ball collisions (elastic)
    - Ball-boundary collisions (bounce)
    """

    def __init__(self, court: Court):
        """
        Initialize physics engine.

        Args:
            court: The court for boundary checking
        """
        self.court = court
        # Optional callback for strategic yard line placement
        # Signature: (ball, exit_position, boundary_hit) -> Vector2
        self.yard_line_placement_callback: Optional[Callable] = None

    def update(self, balls: Dict[str, Ball], dt: float) -> List[Dict]:
        """
        Update all ball positions and handle collisions.

        Args:
            balls: Dictionary of all balls
            dt: Time step in seconds

        Returns:
            List of collision events that occurred
        """
        events = []

        # Update each ball
        for ball in balls.values():
            if ball.is_moving:
                # Apply friction
                ball.apply_friction(dt)

                # Update position
                ball.update(dt)

                # Check boundary collision
                boundary_event = self._handle_boundary_collision(ball)
                if boundary_event:
                    events.append(boundary_event)

                # Check hoop collisions (hitting uprights)
                hoop_event = self._handle_hoop_collision(ball)
                if hoop_event:
                    events.append(hoop_event)

        # Check ball-ball collisions
        ball_list = list(balls.values())
        for i, ball1 in enumerate(ball_list):
            for ball2 in ball_list[i + 1:]:
                collision_event = self._handle_ball_collision(ball1, ball2)
                if collision_event:
                    events.append(collision_event)

        return events

    def _handle_boundary_collision(self, ball: Ball) -> Optional[Dict]:
        """
        Check and handle boundary collision for a ball.

        In croquet, balls do NOT bounce off boundaries. When a ball goes out
        of bounds, it is placed on the yard line (1 yard from the boundary)
        at the point where it went out, and it stops.

        If a yard_line_placement_callback is set, it will be called to allow
        strategic placement of the ball on the yard line.

        Args:
            ball: Ball to check

        Returns:
            Collision event dict if collision occurred, None otherwise
        """
        collision = self.court.get_boundary_collision(
            ball.position,
            ball.velocity,
            ball.radius
        )

        if collision:
            # Ball went out of bounds - place on yard line (1 yard from boundary)
            # The yard line is 1 yard inside the court boundary
            yard_line_distance = config.BOUNDARY_MARGIN_YARDS  # 1 yard

            # Determine which boundary was hit
            exit_position = ball.position.copy()
            boundary_hit = self._determine_boundary_hit(ball.position)

            # Place ball at yard line based on which boundary was hit
            # Keep the coordinate along the boundary, but move 1 yard inward
            x = ball.position.x
            y = ball.position.y

            # Clamp x to yard line if near left/right boundary
            # Use yard_line_distance as the threshold (not ball.radius)
            # Left boundary - place at x = 1
            if x < yard_line_distance:
                x = yard_line_distance
            # Right boundary - place at x = width - 1
            elif x > self.court.width - yard_line_distance:
                x = self.court.width - yard_line_distance

            # Clamp y to yard line if near top/bottom boundary
            # Bottom boundary - place at y = 1
            if y < yard_line_distance:
                y = yard_line_distance
            # Top boundary - place at y = height - 1
            elif y > self.court.height - yard_line_distance:
                y = self.court.height - yard_line_distance

            default_position = Vector2(x, y)

            # Use strategic placement callback if available
            if self.yard_line_placement_callback:
                strategic_position = self.yard_line_placement_callback(
                    ball, exit_position, boundary_hit
                )
                if strategic_position:
                    ball.position = strategic_position
                else:
                    ball.position = default_position
            else:
                ball.position = default_position

            # Ball STOPS when it goes out - no bounce
            ball.velocity = Vector2(0, 0)

            return {
                'type': 'boundary',
                'ball': ball.color,
                'position': ball.position.to_tuple(),
                'out_of_bounds': True,  # Mark that ball went out
                'boundary_hit': boundary_hit
            }

        return None

    def _determine_boundary_hit(self, position: Vector2) -> str:
        """Determine which boundary the ball hit."""
        boundaries = []
        if position.x <= 0:
            boundaries.append('west')
        elif position.x >= self.court.width:
            boundaries.append('east')
        if position.y <= 0:
            boundaries.append('south')
        elif position.y >= self.court.height:
            boundaries.append('north')
        return '_'.join(boundaries) if boundaries else 'unknown'

    def get_valid_yard_line_positions(self, boundary_hit: str, exit_point: Vector2) -> List[Vector2]:
        """
        Get all valid yard line positions for a ball that went out of bounds.

        In Association Croquet, the ball can be placed anywhere on the yard line
        of the boundary where it went out.

        Args:
            boundary_hit: Which boundary was hit ('north', 'south', 'east', 'west', or corner)
            exit_point: Where the ball exited

        Returns:
            List of valid placement positions along the yard line
        """
        yard_distance = config.BOUNDARY_MARGIN_YARDS
        positions = []

        # Sample positions every 2 yards along the appropriate yard line
        step = 2.0

        if 'north' in boundary_hit:
            # North yard line: y = height - 1, full width
            y = self.court.height - yard_distance
            for x in range(int(yard_distance), int(self.court.width - yard_distance) + 1, int(step)):
                positions.append(Vector2(float(x), y))
        elif 'south' in boundary_hit:
            # South yard line: y = 1, full width
            y = yard_distance
            for x in range(int(yard_distance), int(self.court.width - yard_distance) + 1, int(step)):
                positions.append(Vector2(float(x), y))

        if 'east' in boundary_hit and 'north' not in boundary_hit and 'south' not in boundary_hit:
            # East yard line only: x = width - 1, full height
            x = self.court.width - yard_distance
            for y in range(int(yard_distance), int(self.court.height - yard_distance) + 1, int(step)):
                positions.append(Vector2(x, float(y)))
        elif 'west' in boundary_hit and 'north' not in boundary_hit and 'south' not in boundary_hit:
            # West yard line only: x = 1, full height
            x = yard_distance
            for y in range(int(yard_distance), int(self.court.height - yard_distance) + 1, int(step)):
                positions.append(Vector2(x, float(y)))

        # If no positions found (shouldn't happen), use exit point clamped to yard line
        if not positions:
            positions.append(exit_point)

        return positions

    def _handle_ball_collision(self, ball1: Ball, ball2: Ball) -> Optional[Dict]:
        """
        Check and handle collision between two balls.

        Args:
            ball1: First ball
            ball2: Second ball

        Returns:
            Collision event dict if collision occurred, None otherwise
        """
        # Check if balls are overlapping
        delta = ball2.position - ball1.position
        distance = delta.magnitude()

        # Combined radius (scaled up for visibility)
        min_distance = (ball1.radius + ball2.radius) * 2

        if distance >= min_distance or distance == 0:
            return None

        # Normal vector from ball1 to ball2
        normal = delta.normalize()

        # Relative velocity
        rel_vel = ball1.velocity - ball2.velocity

        # Velocity component along collision normal
        vel_along_normal = rel_vel.dot(normal)

        # Don't resolve if balls are separating
        if vel_along_normal < 0:
            return None

        # Calculate impulse (equal mass assumption)
        impulse = vel_along_normal * config.RESTITUTION

        # Apply impulse to both balls
        ball1.velocity = ball1.velocity - normal * impulse
        ball2.velocity = ball2.velocity + normal * impulse

        # Separate balls to prevent overlap
        overlap = min_distance - distance
        if overlap > 0:
            separation = normal * (overlap / 2 + 0.01)
            ball1.position = ball1.position - separation
            ball2.position = ball2.position + separation

        return {
            'type': 'ball_collision',
            'ball1': ball1.color,
            'ball2': ball2.color,
            'position': ((ball1.position + ball2.position) * 0.5).to_tuple()
        }

    def _handle_hoop_collision(self, ball: Ball) -> Optional[Dict]:
        """
        Check and handle collision between a ball and hoop uprights.

        In real croquet, a ball can:
        1. Pass cleanly through the hoop
        2. Hit an upright and bounce off (failed attempt)
        3. Stop in the jaws of the hoop (stuck)
        4. Graze an upright and still make it through

        Args:
            ball: Ball to check

        Returns:
            Event dict if collision occurred, None otherwise
        """
        # Check all hoops for collision
        for hoop in self.court.hoops:
            # Get upright positions
            left_upright, right_upright = self.court.get_hoop_uprights(hoop)

            # Upright radius - thin metal posts ~0.05 yards (about 2 inches)
            upright_radius = 0.05

            # Combined collision radius
            collision_dist = ball.radius + upright_radius

            # Check left upright
            left_delta = ball.position - left_upright
            left_dist = left_delta.magnitude()

            if left_dist < collision_dist and left_dist > 0:
                # Hit left upright - bounce off
                event = self._bounce_off_upright(ball, left_upright, left_delta, left_dist, collision_dist, hoop)
                if event:
                    return event

            # Check right upright
            right_delta = ball.position - right_upright
            right_dist = right_delta.magnitude()

            if right_dist < collision_dist and right_dist > 0:
                # Hit right upright - bounce off
                event = self._bounce_off_upright(ball, right_upright, right_delta, right_dist, collision_dist, hoop)
                if event:
                    return event

            # Check if ball is in the jaws (between uprights, near hoop center)
            event = self._check_jaws(ball, hoop, left_upright, right_upright)
            if event:
                return event

        return None

    def _bounce_off_upright(
        self,
        ball: Ball,
        upright_pos: Vector2,
        delta: Vector2,
        distance: float,
        collision_dist: float,
        hoop
    ) -> Optional[Dict]:
        """
        Handle a ball bouncing off a hoop upright.

        Args:
            ball: The ball
            upright_pos: Position of the upright
            delta: Vector from upright to ball
            distance: Current distance
            collision_dist: Distance at which collision occurs
            hoop: The hoop being hit

        Returns:
            Event dict describing the collision
        """
        # Velocity towards upright
        if ball.velocity.magnitude() < 0.01:
            return None

        normal = delta.normalize()
        vel_towards = -ball.velocity.dot(normal)

        # Only bounce if moving towards the upright
        if vel_towards <= 0:
            return None

        # Reflect velocity with some energy loss
        restitution = 0.6  # Uprights absorb more energy than ball-ball collisions
        ball.velocity = ball.velocity + normal * (vel_towards * (1 + restitution))

        # Separate ball from upright
        overlap = collision_dist - distance
        if overlap > 0:
            ball.position = ball.position + normal * (overlap + 0.01)

        # Debug output removed - too verbose for training

        return {
            'type': 'hoop_hit',
            'ball': ball.color,
            'hoop': hoop.number,
            'result': 'bounced_off',
            'position': ball.position.to_tuple()
        }

    def _check_jaws(
        self,
        ball: Ball,
        hoop,
        left_upright: Vector2,
        right_upright: Vector2
    ) -> Optional[Dict]:
        """
        Check if a ball has stopped in the jaws of a hoop.

        A ball is "in the jaws" when it's between the uprights,
        at or past the hoop plane, and moving slowly.

        Args:
            ball: The ball to check
            hoop: The hoop
            left_upright: Left upright position
            right_upright: Right upright position

        Returns:
            Event dict if ball is stuck in jaws, None otherwise
        """
        # Only check slow-moving balls
        speed = ball.velocity.magnitude()
        if speed > 0.5:  # Ball still moving too fast
            return None

        hoop_pos = hoop.position

        # Distance to hoop center
        to_hoop = hoop_pos - ball.position
        dist_to_hoop = to_hoop.magnitude()

        # Must be very close to hoop center
        if dist_to_hoop > hoop.width * 0.8:
            return None

        # Check if ball is between the uprights (perpendicular check)
        perp_dir = Vector2(-hoop.direction.y, hoop.direction.x)
        perp_dist = abs((ball.position - hoop_pos).dot(perp_dir))

        # Ball must be within the hoop gap
        gap_width = hoop.width / 2 - 0.05  # Subtract upright radius
        if perp_dist > gap_width:
            return None

        # Check position along hoop direction (is it IN the hoop, not past it?)
        along_dist = (ball.position - hoop_pos).dot(hoop.direction)

        # Ball is in jaws if it's within the hoop thickness (~0.2 yards)
        if abs(along_dist) < 0.2:
            # Ball is in the jaws!
            if speed < 0.1:
                # Ball has effectively stopped in the jaws
                ball.velocity = Vector2(0, 0)
                # Debug output removed - too verbose for training

                return {
                    'type': 'jaws',
                    'ball': ball.color,
                    'hoop': hoop.number,
                    'position': ball.position.to_tuple()
                }

        return None

    def check_hoop_approach(
        self,
        ball: Ball,
        target_hoop,
        shot_angle: float,
        shot_power: float
    ) -> Tuple[str, float]:
        """
        Predict the outcome of a hoop shot based on approach angle and power.

        REFINED BASED ON AITON'S TEACHINGS (Section 2.3):
        - Ideal approach: 1 yard in front of hoop
        - Stop-shot ratio ~1:6 for approaches
        - Right side approaches are easier than left
        - 12 inches in front requires excellent control
        - Angle significantly affects difficulty

        Returns:
            Tuple of (outcome, probability) where outcome is one of:
            - 'clean': Ball goes through cleanly
            - 'hit_upright': Ball hits upright and bounces
            - 'jaws': Ball stops in jaws
            - 'miss': Ball misses the hoop entirely

        The probability represents likelihood of success.
        """
        import math

        hoop_pos = target_hoop.position
        hoop_dir = target_hoop.direction

        # Calculate approach angle relative to hoop
        shot_dir = Vector2(math.cos(shot_angle), math.sin(shot_angle))
        alignment = shot_dir.dot(hoop_dir)

        # Distance to hoop
        dist_to_hoop = (hoop_pos - ball.position).magnitude()

        # AITON DISTANCE FACTOR:
        # - 1 yard is ideal ("typically realistic" per Aiton)
        # - 12 inches (0.33 yards) needs "excellent control"
        # - Beyond 3 yards becomes progressively harder
        if dist_to_hoop < 0.33:
            # Too close - hard to control (Aiton: needs excellent control)
            dist_factor = 0.75
        elif dist_to_hoop <= 1.0:
            # IDEAL ZONE per Aiton
            dist_factor = 0.95
        elif dist_to_hoop <= 3.0:
            # Good zone - linear dropoff from ideal
            dist_factor = 0.95 - (dist_to_hoop - 1.0) * 0.075
        elif dist_to_hoop <= 7.0:
            # Medium range - still playable
            dist_factor = 0.80 - (dist_to_hoop - 3.0) * 0.075
        else:
            # Long range - difficult
            dist_factor = max(0.35, 0.50 - (dist_to_hoop - 7.0) * 0.03)

        # AITON ANGLE FACTOR:
        # - Straight approach is easiest
        # - Angled approaches progressively harder
        # - Beyond 45 degrees is very difficult
        approach_angle_rad = math.acos(max(0, min(1, alignment)))
        approach_angle_deg = math.degrees(approach_angle_rad)

        if approach_angle_deg < 10:
            angle_factor = 1.0  # Straight - excellent
        elif approach_angle_deg < 30:
            angle_factor = 0.90 - (approach_angle_deg - 10) * 0.005  # Good
        elif approach_angle_deg < 45:
            angle_factor = 0.80 - (approach_angle_deg - 30) * 0.01   # Acceptable
        elif approach_angle_deg < 60:
            angle_factor = 0.65 - (approach_angle_deg - 45) * 0.015  # Difficult
        else:
            angle_factor = max(0.3, 0.50 - (approach_angle_deg - 60) * 0.01)  # Very hard

        # AITON SIDE FACTOR:
        # Right side approaches are easier than left (Section 2.3)
        perp_dir = Vector2(-hoop_dir.y, hoop_dir.x)
        ball_offset = ball.position - hoop_pos
        side_dot = ball_offset.dot(perp_dir)

        if abs(side_dot) < 0.5:
            side_factor = 1.0   # Straight - no penalty
        elif side_dot > 0:
            side_factor = 0.95  # Right side - easier per Aiton
        else:
            side_factor = 0.88  # Left side - harder per Aiton

        # Power factor - need controlled stroke
        ideal_power = dist_to_hoop * 1.3 + 2.0  # Enough to go through
        power_ratio = shot_power / ideal_power if ideal_power > 0 else 1.0

        if 0.85 <= power_ratio <= 1.2:
            power_factor = 0.95  # Good controlled power
        elif 0.7 <= power_ratio <= 1.4:
            power_factor = 0.80  # Acceptable
        elif 0.5 <= power_ratio <= 1.8:
            power_factor = 0.65  # Risky
        else:
            power_factor = 0.45  # Poor control

        # Combined probability using Aiton's principles
        success_prob = dist_factor * angle_factor * side_factor * power_factor

        # Determine outcome based on probability
        if success_prob > 0.82:
            return ('clean', success_prob)
        elif success_prob > 0.60:
            # Might hit upright or get through
            if alignment > 0.92:
                return ('clean', success_prob)
            else:
                return ('hit_upright', success_prob * 0.85)
        elif success_prob > 0.40:
            # Likely to hit upright or stop in jaws
            if power_ratio < 0.75:
                return ('jaws', success_prob * 0.75)
            else:
                return ('hit_upright', success_prob * 0.65)
        else:
            return ('miss', success_prob)

    def are_all_balls_stopped(self, balls: Dict[str, Ball]) -> bool:
        """
        Check if all balls have stopped moving.

        Args:
            balls: Dictionary of all balls

        Returns:
            True if all balls are stationary
        """
        return all(not ball.is_moving for ball in balls.values())

    def is_in_yard_line_area(self, position: Vector2) -> bool:
        """
        Check if a position is in the yard line area (between boundary and yard line).

        In AC, balls in this area must be "marked in" to the yard line,
        except for the striker's ball when they still have strokes.

        Args:
            position: Position to check

        Returns:
            True if position is in the yard line area
        """
        yard_distance = config.BOUNDARY_MARGIN_YARDS

        # Check if within yard line area on any side
        in_west_area = position.x < yard_distance
        in_east_area = position.x > self.court.width - yard_distance
        in_south_area = position.y < yard_distance
        in_north_area = position.y > self.court.height - yard_distance

        return in_west_area or in_east_area or in_south_area or in_north_area

    def mark_ball_in(self, ball: Ball) -> Optional[Dict]:
        """
        Mark a ball in to the yard line if it's in the yard line area.

        In AC, when a ball comes to rest between the boundary and the yard line,
        it must be placed on the yard line at the nearest point.

        Args:
            ball: Ball to potentially mark in

        Returns:
            Event dict if ball was marked in, None otherwise
        """
        if not self.is_in_yard_line_area(ball.position):
            return None

        yard_distance = config.BOUNDARY_MARGIN_YARDS
        x = ball.position.x
        y = ball.position.y
        was_marked_in = False

        # Mark in from west yard line area
        if x < yard_distance:
            x = yard_distance
            was_marked_in = True
        # Mark in from east yard line area
        elif x > self.court.width - yard_distance:
            x = self.court.width - yard_distance
            was_marked_in = True

        # Mark in from south yard line area
        if y < yard_distance:
            y = yard_distance
            was_marked_in = True
        # Mark in from north yard line area
        elif y > self.court.height - yard_distance:
            y = self.court.height - yard_distance
            was_marked_in = True

        if was_marked_in:
            ball.position = Vector2(x, y)
            return {
                'type': 'marked_in',
                'ball': ball.color,
                'position': ball.position.to_tuple()
            }

        return None

    def mark_in_all_balls(
        self,
        balls: Dict[str, Ball],
        striker_color: Optional[str] = None,
        striker_has_strokes: bool = False
    ) -> List[Dict]:
        """
        Mark in all balls that are in the yard line area.

        In AC rules:
        - All balls in the yard line area must be marked in to the yard line
        - EXCEPT the striker's ball if they still have strokes remaining

        Args:
            balls: Dictionary of all balls
            striker_color: Color of current striker's ball (if any)
            striker_has_strokes: Whether striker still has strokes remaining

        Returns:
            List of mark-in events
        """
        events = []

        for color, ball in balls.items():
            # Skip striker's ball if they still have strokes
            if striker_has_strokes and color == striker_color:
                continue

            # Skip balls that have pegged out
            if ball.has_pegged_out:
                continue

            event = self.mark_ball_in(ball)
            if event:
                events.append(event)

        return events

    def shoot_ball(self, ball: Ball, velocity: Vector2):
        """
        Apply an initial velocity to a ball (take a shot).

        Args:
            ball: Ball to shoot
            velocity: Initial velocity vector
        """
        # Record where the shot started (for hoop detection)
        ball.shot_start_position = ball.position.copy()
        ball.set_velocity(velocity)

    def execute_croquet_stroke(
        self,
        striker: Ball,
        croqueted: Ball,
        striker_velocity: Vector2,
        croqueted_velocity: Vector2
    ):
        """
        Execute a croquet stroke where both balls are set in motion.

        Args:
            striker: The striker's ball
            croqueted: The ball being croqueted
            striker_velocity: Velocity for striker ball
            croqueted_velocity: Velocity for croqueted ball
        """
        # Record shot start positions
        striker.shot_start_position = striker.position.copy()
        croqueted.shot_start_position = croqueted.position.copy()

        # Set both balls in motion
        striker.set_velocity(striker_velocity)
        croqueted.set_velocity(croqueted_velocity)
