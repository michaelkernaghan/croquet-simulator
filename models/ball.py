"""
Ball entity representing a croquet ball with physics properties.
"""
import math
from typing import Tuple
import config


class Vector2:
    """Simple 2D vector class for physics calculations."""

    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def __add__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector2':
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> 'Vector2':
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'Vector2':
        if scalar == 0:
            return Vector2(0, 0)
        return Vector2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> 'Vector2':
        return Vector2(-self.x, -self.y)

    def __repr__(self):
        return f"Vector2({self.x:.2f}, {self.y:.2f})"

    def dot(self, other: 'Vector2') -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y

    def magnitude(self) -> float:
        """Length of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> 'Vector2':
        """Return unit vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vector2(0, 0)
        return self / mag

    def copy(self) -> 'Vector2':
        """Return a copy of this vector."""
        return Vector2(self.x, self.y)

    @staticmethod
    def from_angle(angle: float, magnitude: float = 1.0) -> 'Vector2':
        """Create vector from angle (radians) and magnitude."""
        return Vector2(
            math.cos(angle) * magnitude,
            math.sin(angle) * magnitude
        )

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)


class Ball:
    """
    Represents a croquet ball with position, velocity, and game state.
    """

    def __init__(self, color: str, position: Tuple[float, float]):
        """
        Initialize a ball.

        Args:
            color: Ball color identifier ("blue", "black", "red", "yellow")
            position: Initial position in yards (x, y)
        """
        self.color = color
        self.position = Vector2(position[0], position[1])
        self.velocity = Vector2(0, 0)
        self.previous_position = self.position.copy()

        # Position at start of current shot (for hoop detection)
        self.shot_start_position = self.position.copy()

        # Physical properties
        self.radius = config.BALL_RADIUS_YARDS
        self.mass = config.BALL_MASS

        # Game state
        self.hoops_run = 0  # Number of hoops completed (0-12)
        self.is_rover = False  # Has run all 12 hoops
        self.has_pegged_out = False  # Hit the peg after becoming rover

    @property
    def is_moving(self) -> bool:
        """Check if the ball is currently moving."""
        return self.velocity.magnitude() > config.MIN_VELOCITY

    def update(self, dt: float):
        """
        Update ball position based on velocity.

        Args:
            dt: Time step in seconds
        """
        if not self.is_moving:
            self.velocity = Vector2(0, 0)
            return

        # Store previous position for collision detection
        self.previous_position = self.position.copy()

        # Update position
        self.position = self.position + self.velocity * dt

    def apply_friction(self, dt: float):
        """
        Apply friction to slow the ball down.

        Args:
            dt: Time step in seconds
        """
        if not self.is_moving:
            return

        # Friction force opposes motion
        friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY
        speed = self.velocity.magnitude()

        # Calculate new speed after friction
        new_speed = max(0, speed - friction_decel * dt)

        if new_speed < config.MIN_VELOCITY:
            self.velocity = Vector2(0, 0)
        else:
            # Maintain direction, reduce magnitude
            self.velocity = self.velocity.normalize() * new_speed

    def apply_impulse(self, impulse: Vector2):
        """
        Apply an impulse (instant velocity change) to the ball.

        Args:
            impulse: Velocity change vector
        """
        self.velocity = self.velocity + impulse

    def set_velocity(self, velocity: Vector2):
        """Set the ball's velocity directly."""
        self.velocity = velocity.copy()

    def stop(self):
        """Stop the ball immediately."""
        self.velocity = Vector2(0, 0)

    def get_next_hoop_number(self) -> int:
        """Get the number of the next hoop to run (1-12), or 0 if finished."""
        if self.hoops_run >= 12:
            return 0  # Ready to peg out
        return self.hoops_run + 1

    def run_hoop(self):
        """Record that this ball has run its next hoop."""
        if self.hoops_run < 12:
            self.hoops_run += 1
            if self.hoops_run >= 12:
                self.is_rover = True

    def peg_out(self):
        """Record that this ball has pegged out (finished)."""
        self.has_pegged_out = True

    def get_pixel_position(self) -> Tuple[int, int]:
        """Convert yard position to screen pixel position."""
        px = int(self.position.x * config.YARDS_TO_PIXELS + config.COURT_OFFSET_X)
        # Flip Y axis (screen Y increases downward, court Y increases upward)
        py = int((config.COURT_HEIGHT_YARDS - self.position.y) * config.YARDS_TO_PIXELS + config.COURT_OFFSET_Y)
        return (px, py)

    def distance_to(self, other: 'Ball') -> float:
        """Calculate distance to another ball in yards."""
        return (self.position - other.position).magnitude()

    def __repr__(self):
        return f"Ball({self.color}, pos={self.position}, hoops={self.hoops_run})"
