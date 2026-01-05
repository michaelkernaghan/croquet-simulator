"""
Court renderer - draws the croquet court, hoops, and peg.
"""
import pygame
import math
from typing import Tuple, Dict, Optional
import config
from models.court import Court, Hoop
from models.ball import Ball


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

    def draw(self, court: Court, balls: Optional[Dict[str, Ball]] = None):
        """
        Draw the complete court.

        Args:
            court: The court to draw
            balls: Optional dictionary of balls (for drawing hoop clips)
        """
        self._draw_grass()
        self._draw_boundary_lines(court)
        self._draw_yard_lines(court)
        self._draw_hoops(court)
        self._draw_peg(court)

        # Draw clips on hoops showing which ball needs to run each hoop
        if balls:
            self._draw_hoop_clips(court, balls)

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

    def _draw_hoop_clips(self, court: Court, balls: Dict[str, Ball]):
        """
        Draw colored clips on hoops showing which ball needs to run each hoop.

        In AC, clips indicate which hoop each ball must run next:
        - Hoops 1-6: Clip on TOP of the hoop crown
        - Hoops 7-12 (1-back through rover): Clip on the SIDE of the hoop

        Args:
            court: The court with hoops
            balls: Dictionary of balls to check next hoops
        """
        # First, group balls by their target hoop to handle multiple clips on same hoop
        # Key: (physical_hoop_num, clip_on_top), Value: list of ball colors
        hoop_clips = {}

        for color, ball in balls.items():
            if ball.has_pegged_out:
                continue  # No clip for pegged out balls

            hoops_run = ball.hoops_run
            if hoops_run >= 12:
                continue  # Rover - no more hoops, just peg

            # Get the next hoop info
            next_hoop_num = hoops_run + 1  # 1-indexed hoop number (1-12)

            # Determine which physical hoop and clip position
            if next_hoop_num <= 6:
                # First circuit: physical hoop matches, clip on TOP
                physical_hoop_num = next_hoop_num
                clip_on_top = True
            else:
                # Second circuit: need to map to physical hoop, clip on SIDE
                circuit_info = config.AC_SECOND_CIRCUIT.get(hoops_run)
                if circuit_info:
                    physical_hoop_num, _ = circuit_info
                else:
                    physical_hoop_num = None
                clip_on_top = False

            if physical_hoop_num:
                key = (physical_hoop_num, clip_on_top)
                if key not in hoop_clips:
                    hoop_clips[key] = []
                hoop_clips[key].append(color)

        # Now draw clips with offsets for multiple balls on same hoop
        for (physical_hoop_num, clip_on_top), colors in hoop_clips.items():
            physical_hoop = court.get_hoop(physical_hoop_num)
            if physical_hoop:
                for idx, color in enumerate(colors):
                    self._draw_clip(physical_hoop, color, clip_on_top, idx, len(colors))

    def _draw_clip(self, hoop: Hoop, ball_color: str, on_top: bool, index: int = 0, total: int = 1):
        """
        Draw a colored clip on a hoop.

        Args:
            hoop: The hoop to draw the clip on
            ball_color: Color of the ball (blue, red, black, yellow)
            on_top: If True, draw on top bar; if False, draw on side
            index: Index of this clip (for offsetting multiple clips)
            total: Total number of clips on this hoop
        """
        px, py = hoop.get_pixel_position()
        clip_color = config.BALL_COLORS.get(ball_color, config.WHITE)

        # Clip dimensions
        clip_width = 8
        clip_height = 6

        width = config.HOOP_WIDTH_PX
        height = config.HOOP_HEIGHT_PX

        if on_top:
            # Clip on top of the crown (hoops 1-6)
            # Spread multiple clips horizontally across the top bar
            total_clips_width = total * clip_width + (total - 1) * 2  # 2px gap between clips
            start_x = px - total_clips_width // 2
            clip_x = start_x + index * (clip_width + 2)
            clip_y = py - height // 2 - clip_height  # Above the crown
        else:
            # Clip on the side (hoops 7-12)
            # Stack multiple clips vertically on the side
            total_clips_height = total * clip_height + (total - 1) * 2  # 2px gap
            start_y = py - total_clips_height // 2
            clip_x = px - width // 2 - clip_width  # Left of left upright
            clip_y = start_y + index * (clip_height + 2)

        # Draw the clip as a small rectangle
        pygame.draw.rect(
            self.screen,
            clip_color,
            (clip_x, clip_y, clip_width, clip_height)
        )

        # Add a small outline for visibility
        pygame.draw.rect(
            self.screen,
            config.BLACK,
            (clip_x, clip_y, clip_width, clip_height),
            1
        )
