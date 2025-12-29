"""
Ball renderer - draws croquet balls with colors and labels.
"""
import pygame
from typing import Dict, Optional
import config
from models.ball import Ball


class BallRenderer:
    """Renders croquet balls with proper colors, shadows, and labels."""

    def __init__(self, screen: pygame.Surface):
        """
        Initialize the ball renderer.

        Args:
            screen: Pygame surface to draw on
        """
        self.screen = screen
        self.font = None

    def _ensure_font(self):
        """Lazily initialize font (after pygame.init())."""
        if self.font is None:
            self.font = pygame.font.Font(None, 18)

    def draw(self, balls: Dict[str, Ball], active_ball: Optional[str] = None):
        """
        Draw all balls.

        Args:
            balls: Dictionary of ball color to Ball object
            active_ball: Color of the currently active ball (highlighted)
        """
        # Draw shadows first (so they appear under balls)
        for ball in balls.values():
            if not ball.has_pegged_out:
                self._draw_shadow(ball)

        # Draw balls
        for color, ball in balls.items():
            if not ball.has_pegged_out:
                is_active = (color == active_ball)
                self._draw_ball(ball, is_active)

    def _draw_shadow(self, ball: Ball):
        """
        Draw a shadow under the ball.

        Args:
            ball: The ball to draw shadow for
        """
        px, py = ball.get_pixel_position()
        shadow_offset = 3
        shadow_color = (20, 50, 20, 128)  # Semi-transparent dark

        # Draw shadow as slightly offset darker circle
        shadow_surface = pygame.Surface((config.BALL_RADIUS_PX * 2 + 4, config.BALL_RADIUS_PX * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(
            shadow_surface,
            shadow_color,
            (config.BALL_RADIUS_PX + 2, config.BALL_RADIUS_PX + 2),
            config.BALL_RADIUS_PX
        )
        self.screen.blit(
            shadow_surface,
            (px - config.BALL_RADIUS_PX - 2 + shadow_offset, py - config.BALL_RADIUS_PX - 2 + shadow_offset)
        )

    def _draw_ball(self, ball: Ball, is_active: bool):
        """
        Draw a single ball.

        Args:
            ball: The ball to draw
            is_active: Whether this ball is currently active (gets highlight)
        """
        self._ensure_font()
        px, py = ball.get_pixel_position()
        radius = config.BALL_RADIUS_PX
        color = config.BALL_COLORS.get(ball.color, config.WHITE)

        # Draw active ball highlight (ring around ball)
        if is_active:
            pygame.draw.circle(
                self.screen,
                config.WHITE,
                (px, py),
                radius + 4,
                3
            )

        # Draw main ball
        pygame.draw.circle(self.screen, color, (px, py), radius)

        # Draw outline (helps black ball be visible)
        outline_color = config.WHITE if ball.color == "black" else config.BLACK
        pygame.draw.circle(self.screen, outline_color, (px, py), radius, 2)

        # Draw highlight (gives 3D effect)
        highlight_offset = radius // 3
        highlight_radius = radius // 3
        highlight_color = self._lighten_color(color, 0.4)
        pygame.draw.circle(
            self.screen,
            highlight_color,
            (px - highlight_offset, py - highlight_offset),
            highlight_radius
        )

        # Draw ball label (first letter)
        label_char = ball.color[0].upper()  # B, K (for black), R, Y
        if ball.color == "black":
            label_char = "K"  # Use K for black to distinguish from blue

        label_color = config.WHITE if ball.color in ("black", "blue") else config.BLACK
        label = self.font.render(label_char, True, label_color)
        label_rect = label.get_rect(center=(px, py))
        self.screen.blit(label, label_rect)

        # Draw hoops progress indicator (small dots below ball)
        self._draw_progress_indicator(ball, px, py + radius + 8)

    def _draw_progress_indicator(self, ball: Ball, x: int, y: int):
        """
        Draw small indicator showing ball's hoop progress.

        Args:
            ball: The ball
            x: Center X position
            y: Y position for indicator
        """
        # Show hoops run as filled/empty dots
        dot_radius = 2
        dot_spacing = 6
        total_dots = 6  # Show 6 dots (one per hoop in current circuit)

        hoops_in_circuit = ball.hoops_run % 6 if ball.hoops_run < 12 else 6
        circuit = ball.hoops_run // 6  # 0 = first circuit, 1 = second, 2 = done

        start_x = x - (total_dots * dot_spacing) // 2

        for i in range(total_dots):
            dot_x = start_x + i * dot_spacing
            is_filled = i < hoops_in_circuit

            if is_filled:
                pygame.draw.circle(self.screen, config.WHITE, (dot_x, y), dot_radius)
            else:
                pygame.draw.circle(self.screen, config.WHITE, (dot_x, y), dot_radius, 1)

        # Show circuit indicator (I or II)
        if circuit > 0 and self.font:
            circuit_label = "II" if circuit >= 1 else "I"
            if ball.is_rover:
                circuit_label = "R"  # Rover
            label = self.font.render(circuit_label, True, config.WHITE)
            label_rect = label.get_rect(midleft=(start_x + total_dots * dot_spacing + 5, y))
            self.screen.blit(label, label_rect)

    def _lighten_color(self, color: tuple, factor: float) -> tuple:
        """
        Lighten a color by a factor.

        Args:
            color: RGB tuple
            factor: Amount to lighten (0-1)

        Returns:
            Lightened RGB tuple
        """
        r, g, b = color
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return (r, g, b)
