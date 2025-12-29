"""
Court renderer - draws the croquet court, hoops, and peg.
"""
import pygame
import math
from typing import Tuple
import config
from models.court import Court, Hoop


class CourtRenderer:
    """Renders the croquet court with hoops and peg."""

    def __init__(self, screen: pygame.Surface):
        """
        Initialize the court renderer.

        Args:
            screen: Pygame surface to draw on
        """
        self.screen = screen
        self.font = None

    def _ensure_font(self):
        """Lazily initialize font (after pygame.init())."""
        if self.font is None:
            self.font = pygame.font.Font(None, 20)

    def draw(self, court: Court):
        """
        Draw the complete court.

        Args:
            court: The court to draw
        """
        self._draw_grass()
        self._draw_boundary_lines(court)
        self._draw_yard_lines(court)
        self._draw_hoops(court)
        self._draw_peg(court)

    def _draw_grass(self):
        """Draw the grass background."""
        # Draw court area
        court_rect = pygame.Rect(
            config.COURT_OFFSET_X,
            config.COURT_OFFSET_Y,
            config.COURT_WIDTH_PX,
            config.COURT_HEIGHT_PX
        )
        pygame.draw.rect(self.screen, config.GRASS_GREEN, court_rect)

        # Slightly darker border area outside court
        border_color = (35, 75, 36)
        self.screen.fill(border_color, self.screen.get_rect())
        pygame.draw.rect(self.screen, config.GRASS_GREEN, court_rect)

    def _draw_boundary_lines(self, court: Court):
        """Draw the white boundary lines."""
        # Outer boundary
        outer_rect = pygame.Rect(
            config.COURT_OFFSET_X,
            config.COURT_OFFSET_Y,
            config.COURT_WIDTH_PX,
            config.COURT_HEIGHT_PX
        )
        pygame.draw.rect(self.screen, config.BOUNDARY_WHITE, outer_rect, 3)

        # Yard line (1 yard inside boundary)
        margin_px = int(config.BOUNDARY_MARGIN_YARDS * config.YARDS_TO_PIXELS)
        inner_rect = pygame.Rect(
            config.COURT_OFFSET_X + margin_px,
            config.COURT_OFFSET_Y + margin_px,
            config.COURT_WIDTH_PX - 2 * margin_px,
            config.COURT_HEIGHT_PX - 2 * margin_px
        )
        pygame.draw.rect(self.screen, config.BOUNDARY_WHITE, inner_rect, 1)

    def _draw_yard_lines(self, court: Court):
        """Draw the baulk lines (where balls enter play)."""
        # Baulk lines run HALF the width of the court, 1 yard in from boundary
        #
        # A-baulk (South): y=1, from x=0 (west) to x=14 (center)
        # B-baulk (North): y=34, from x=14 (center) to x=28 (east)

        baulk_color = (200, 200, 100)  # Yellowish for visibility

        # A-baulk (south yard-line, west half) - y=1, x from 0 to 14
        a_start_px = self._yards_to_pixels(config.BAULK_A_START[0], config.BAULK_A_START[1])
        a_end_px = self._yards_to_pixels(config.BAULK_A_END[0], config.BAULK_A_END[1])
        pygame.draw.line(self.screen, baulk_color, a_start_px, a_end_px, 3)

        # B-baulk (north yard-line, east half) - y=34, x from 14 to 28
        b_start_px = self._yards_to_pixels(config.BAULK_B_START[0], config.BAULK_B_START[1])
        b_end_px = self._yards_to_pixels(config.BAULK_B_END[0], config.BAULK_B_END[1])
        pygame.draw.line(self.screen, baulk_color, b_start_px, b_end_px, 3)

        # Draw small markers at baulk ends
        for pos in [a_start_px, a_end_px, b_start_px, b_end_px]:
            pygame.draw.circle(self.screen, baulk_color, pos, 4)

    def _yards_to_pixels(self, x_yards: float, y_yards: float) -> Tuple[int, int]:
        """Convert yard coordinates to pixel coordinates."""
        px = int(x_yards * config.YARDS_TO_PIXELS + config.COURT_OFFSET_X)
        # Flip Y axis (screen Y increases downward, court Y increases upward)
        py = int((config.COURT_HEIGHT_YARDS - y_yards) * config.YARDS_TO_PIXELS + config.COURT_OFFSET_Y)
        return (px, py)

    def _draw_hoops(self, court: Court):
        """Draw all hoops."""
        for hoop in court.hoops:
            self._draw_hoop(hoop)

    def _draw_hoop(self, hoop: Hoop):
        """
        Draw a single hoop as a crown shape.

        Args:
            hoop: The hoop to draw
        """
        self._ensure_font()
        px, py = hoop.get_pixel_position()

        # Hoop dimensions
        width = config.HOOP_WIDTH_PX
        height = config.HOOP_HEIGHT_PX

        # Draw hoop as two uprights and a crown
        upright_width = 3

        # Left upright
        pygame.draw.rect(
            self.screen,
            config.HOOP_COLOR,
            (px - width // 2, py - height // 2, upright_width, height)
        )

        # Right upright
        pygame.draw.rect(
            self.screen,
            config.HOOP_COLOR,
            (px + width // 2 - upright_width, py - height // 2, upright_width, height)
        )

        # Crown (top bar)
        pygame.draw.rect(
            self.screen,
            config.HOOP_COLOR,
            (px - width // 2, py - height // 2, width, upright_width)
        )

        # Draw direction indicator (small arrow showing which way to run)
        arrow_length = 8
        dir_x = hoop.direction.x * arrow_length
        dir_y = -hoop.direction.y * arrow_length  # Flip Y for screen coords

        # Arrow line
        pygame.draw.line(
            self.screen,
            config.WHITE,
            (px, py + height // 2 + 5),
            (px + dir_x, py + height // 2 + 5 + dir_y),
            2
        )

        # Hoop number label
        label = self.font.render(str(hoop.number), True, config.WHITE)
        label_rect = label.get_rect(center=(px, py + height // 2 + 18))
        self.screen.blit(label, label_rect)

    def _draw_peg(self, court: Court):
        """Draw the center peg."""
        px, py = court.get_peg_pixel_position()
        radius = config.PEG_RADIUS_PX

        # Draw striped peg (alternating red and white)
        stripe_height = 4
        num_stripes = (radius * 2) // stripe_height

        for i in range(num_stripes):
            color = config.PEG_RED if i % 2 == 0 else config.PEG_WHITE
            stripe_y = py - radius + i * stripe_height
            pygame.draw.rect(
                self.screen,
                color,
                (px - radius // 2, stripe_y, radius, stripe_height)
            )

        # Outline
        pygame.draw.rect(
            self.screen,
            config.BLACK,
            (px - radius // 2, py - radius, radius, radius * 2),
            1
        )
