"""
Physics Engine - Handles ball movement, friction, and collisions.
"""
from typing import Dict, List, Optional, Tuple, Callable
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

            # Default placement: where the ball exited, clamped to yard line
            x = ball.position.x
            y = ball.position.y

            # Left boundary - place at x = 1
            if x - ball.radius <= 0:
                x = yard_line_distance
            # Right boundary - place at x = width - 1
            elif x + ball.radius >= self.court.width:
                x = self.court.width - yard_line_distance

            # Bottom boundary - place at y = 1
            if y - ball.radius <= 0:
                y = yard_line_distance
            # Top boundary - place at y = height - 1
            elif y + ball.radius >= self.court.height:
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

    def are_all_balls_stopped(self, balls: Dict[str, Ball]) -> bool:
        """
        Check if all balls have stopped moving.

        Args:
            balls: Dictionary of all balls

        Returns:
            True if all balls are stationary
        """
        return all(not ball.is_moving for ball in balls.values())

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
