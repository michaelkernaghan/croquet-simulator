"""
Main renderer that coordinates all rendering components.
"""
import pygame
from typing import Dict, Optional
import config
from models.court import Court
from models.ball import Ball
from view.court_renderer import CourtRenderer
from view.ball_renderer import BallRenderer


class Renderer:
    """Main renderer that coordinates drawing of all game elements."""

    def __init__(self, screen: pygame.Surface):
        """
        Initialize the renderer.

        Args:
            screen: Pygame surface to draw on
        """
        self.screen = screen
        self.court_renderer = CourtRenderer(screen)
        self.ball_renderer = BallRenderer(screen)
        self.font = None
        self.small_font = None

    def _ensure_fonts(self):
        """Lazily initialize fonts (after pygame.init())."""
        if self.font is None:
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)

    def render(self, court: Court, balls: Dict[str, Ball], active_ball: Optional[str] = None,
               turn_info: Optional[Dict] = None):
        """
        Render the complete game state.

        Args:
            court: The court to render
            balls: Dictionary of balls
            active_ball: Currently active ball color
            turn_info: Optional turn information to display
        """
        # Clear screen
        self.screen.fill(config.BLACK)

        # Draw court (with hoop clips showing which ball needs each hoop)
        self.court_renderer.draw(court, balls)

        # Draw balls
        self.ball_renderer.draw(balls, active_ball)

        # Draw UI overlay
        if turn_info:
            self._draw_turn_info(turn_info)

        # Draw title
        self._draw_title()

    def _draw_title(self):
        """Draw the game title."""
        self._ensure_fonts()
        title = self.font.render("Association Croquet", True, config.WHITE)
        title_rect = title.get_rect(centerx=config.SCREEN_WIDTH // 2, top=10)
        self.screen.blit(title, title_rect)

    def _draw_turn_info(self, turn_info: Dict):
        """
        Draw turn information overlay for Association Croquet.

        Args:
            turn_info: Dictionary with turn details
        """
        self._ensure_fonts()

        # Position in top-right area
        x = config.SCREEN_WIDTH - 200
        y = 50

        # Current turn
        current_ball = turn_info.get("current_ball", "blue")
        ball_color = config.BALL_COLORS.get(current_ball, config.WHITE)

        turn_text = f"Turn: {current_ball.capitalize()}"
        label = self.small_font.render(turn_text, True, ball_color)
        self.screen.blit(label, (x, y))

        # Strokes remaining
        strokes = turn_info.get("strokes_remaining", 1)
        strokes_text = f"Strokes: {strokes}"
        label = self.small_font.render(strokes_text, True, config.WHITE)
        self.screen.blit(label, (x, y + 25))

        # Scores
        y += 60
        label = self.small_font.render("Scores:", True, config.WHITE)
        self.screen.blit(label, (x, y))

        scores = turn_info.get("scores", {})
        y += 25
        for team, score in scores.items():
            score_text = f"{team}: {score}"
            label = self.small_font.render(score_text, True, config.WHITE)
            self.screen.blit(label, (x, y))
            y += 20

    def draw_message(self, message: str, duration_hint: bool = False):
        """
        Draw a message overlay (for events like hoop runs).

        Args:
            message: Message to display
            duration_hint: If True, message is temporary
        """
        self._ensure_fonts()

        # Semi-transparent background
        overlay = pygame.Surface((400, 60), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        # Center on screen
        overlay_rect = overlay.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2))
        self.screen.blit(overlay, overlay_rect)

        # Message text
        label = self.font.render(message, True, config.WHITE)
        label_rect = label.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2))
        self.screen.blit(label, label_rect)
