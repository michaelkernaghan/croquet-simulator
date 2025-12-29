"""
Leave Pattern Bootstrap - Classic croquet leave patterns for learning initialization.

Implements three standard leave patterns that serve as starting points for
the AI's learning of good end-of-turn positions:

1. NSL (North-South Leave): Traditional defensive leave
   - Opponent balls on opposite yard lines (north and south)
   - Partner balls positioned to maintain control

2. OSL (Old Standard Leave): Classic partnership leave
   - Partner balls close together for easy pickup
   - Opponent balls separated and far from hoops

3. Diagonal Leave: Positional control leave
   - Balls positioned on court diagonals
   - Controls multiple hoops simultaneously
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto

from models.ball import Vector2


class LeaveType(Enum):
    """Types of standard leave patterns."""
    NSL = auto()      # North-South Leave
    OSL = auto()      # Old Standard Leave
    DIAGONAL = auto() # Diagonal Leave
    CUSTOM = auto()   # AI-learned custom pattern


@dataclass
class LeavePattern:
    """A leave pattern specifying ideal ball positions."""
    name: str
    leave_type: LeaveType
    description: str
    # Positions are relative to the striker's next hoop
    # Each ball's position is (x_yards, y_yards) from court origin
    ball_positions: Dict[str, Tuple[float, float]]
    # Which ball this leave is optimized for (the one taking the break)
    for_ball: str
    # Target hoop this leave sets up (1-12)
    target_hoop: int
    # Score multiplier - how good this pattern is (1.0 = baseline)
    quality_score: float = 1.0


class LeavePatternLibrary:
    """
    Library of standard croquet leave patterns.

    These patterns serve as:
    1. Bootstrap data for the learning system
    2. Evaluation targets for leave quality scoring
    3. Strategic templates for end-of-turn play
    """

    def __init__(self):
        """Initialize the pattern library with standard leaves."""
        self.patterns: List[LeavePattern] = []
        self._initialize_standard_patterns()

    def _initialize_standard_patterns(self):
        """Initialize the library with classic leave patterns."""
        # NSL patterns - opponent balls on yard lines, partners positioned
        self._add_nsl_patterns()

        # OSL patterns - partners together, opponents separated
        self._add_osl_patterns()

        # Diagonal patterns - court diagonal control
        self._add_diagonal_patterns()

    def _add_nsl_patterns(self):
        """Add North-South Leave patterns."""
        # NSL for hoop 1: Blue to run hoop 1
        # - Red on south yard line near corner IV (SE)
        # - Yellow on north yard line near corner II (NW)
        # - Black positioned as pilot ball near hoop 2
        self.patterns.append(LeavePattern(
            name="NSL Hoop 1 for Blue",
            leave_type=LeaveType.NSL,
            description="Classic NSL with opponents spread N/S, partner at hoop 2",
            ball_positions={
                "blue": (7, 4),      # Approaching hoop 1 from south
                "black": (8, 26),    # Pilot near hoop 2
                "red": (22, 1),      # South yard line, SE area
                "yellow": (6, 34),   # North yard line, NW area
            },
            for_ball="blue",
            target_hoop=1,
            quality_score=1.2  # Excellent defensive leave
        ))

        # NSL for hoop 2: Blue to run hoop 2
        self.patterns.append(LeavePattern(
            name="NSL Hoop 2 for Blue",
            leave_type=LeaveType.NSL,
            description="NSL setup for hoop 2 approach",
            ball_positions={
                "blue": (8, 25),     # Approaching hoop 2 from south
                "black": (20, 26),   # Pilot near hoop 3
                "red": (20, 1),      # South yard line
                "yellow": (8, 34),   # North yard line
            },
            for_ball="blue",
            target_hoop=2,
            quality_score=1.1
        ))

        # NSL for hoop 3: Blue to run hoop 3
        self.patterns.append(LeavePattern(
            name="NSL Hoop 3 for Blue",
            leave_type=LeaveType.NSL,
            description="NSL setup for hoop 3 approach",
            ball_positions={
                "blue": (20, 30),    # Approaching hoop 3 from north
                "black": (22, 9),    # Pilot near hoop 4
                "red": (6, 1),       # South yard line
                "yellow": (20, 34),  # North yard line
            },
            for_ball="blue",
            target_hoop=3,
            quality_score=1.1
        ))

    def _add_osl_patterns(self):
        """Add Old Standard Leave patterns."""
        # OSL: Partners close together, easy pickup
        self.patterns.append(LeavePattern(
            name="OSL Hoop 1 for Blue",
            leave_type=LeaveType.OSL,
            description="Old Standard Leave - partners together near hoop 1",
            ball_positions={
                "blue": (7, 4),      # Ready to run hoop 1
                "black": (9, 5),     # Close to blue for easy roquet
                "red": (22, 20),     # Away from hoops, center-east
                "yellow": (6, 28),   # Away from hoops, west side
            },
            for_ball="blue",
            target_hoop=1,
            quality_score=1.0  # Good offensive potential
        ))

        # OSL for hoop 4
        self.patterns.append(LeavePattern(
            name="OSL Hoop 4 for Blue",
            leave_type=LeaveType.OSL,
            description="Partners positioned for hoop 4 break",
            ball_positions={
                "blue": (21, 10),    # Approaching hoop 4 from north
                "black": (19, 8),    # Close to blue
                "red": (6, 15),      # West side, away from action
                "yellow": (14, 30),  # North, away from hoop 4
            },
            for_ball="blue",
            target_hoop=4,
            quality_score=1.0
        ))

    def _add_diagonal_patterns(self):
        """Add Diagonal Leave patterns."""
        # Diagonal: Balls on court diagonals for multi-hoop control
        self.patterns.append(LeavePattern(
            name="Diagonal Control for Blue",
            leave_type=LeaveType.DIAGONAL,
            description="Diagonal positioning for court control",
            ball_positions={
                "blue": (7, 5),      # Near hoop 1
                "black": (21, 26),   # Near hoop 3 (diagonal from blue)
                "red": (21, 5),      # Near hoop 4
                "yellow": (7, 26),   # Near hoop 2
            },
            for_ball="blue",
            target_hoop=1,
            quality_score=0.9  # Balanced but less focused
        ))

        # Cross-diagonal for center hoops
        self.patterns.append(LeavePattern(
            name="Center Cross Diagonal",
            leave_type=LeaveType.DIAGONAL,
            description="Diagonal control through center hoops",
            ball_positions={
                "blue": (14, 8),     # Near hoop 5
                "black": (14, 26),   # Near hoop 6 (diagonal)
                "red": (6, 17),      # West yard line, center
                "yellow": (22, 17),  # East yard line, center
            },
            for_ball="blue",
            target_hoop=5,
            quality_score=0.85
        ))

    def get_patterns_for_hoop(self, hoop_num: int, ball_color: str = None) -> List[LeavePattern]:
        """
        Get leave patterns suitable for a specific target hoop.

        Args:
            hoop_num: Target hoop number (1-12)
            ball_color: Optional - filter by ball color

        Returns:
            List of applicable leave patterns
        """
        patterns = [p for p in self.patterns if p.target_hoop == hoop_num]
        if ball_color:
            patterns = [p for p in patterns if p.for_ball == ball_color]
        return patterns

    def get_pattern_by_type(self, leave_type: LeaveType) -> List[LeavePattern]:
        """Get all patterns of a specific type."""
        return [p for p in self.patterns if p.leave_type == leave_type]

    def evaluate_leave_similarity(
        self,
        actual_positions: Dict[str, Vector2],
        pattern: LeavePattern
    ) -> float:
        """
        Evaluate how similar actual ball positions are to a pattern.

        Args:
            actual_positions: Current ball positions
            pattern: Leave pattern to compare against

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        total_distance = 0.0
        max_distance = 50.0  # Maximum meaningful distance (diagonal of court)

        for color, (px, py) in pattern.ball_positions.items():
            if color in actual_positions:
                actual = actual_positions[color]
                pattern_pos = Vector2(px, py)
                distance = (actual - pattern_pos).magnitude()
                total_distance += min(distance, max_distance)
            else:
                total_distance += max_distance  # Penalty for missing ball

        # Convert to similarity score (0-1)
        avg_distance = total_distance / len(pattern.ball_positions)
        similarity = 1.0 - (avg_distance / max_distance)

        return max(0, similarity)

    def find_best_matching_pattern(
        self,
        actual_positions: Dict[str, Vector2],
        target_hoop: int = None
    ) -> Tuple[Optional[LeavePattern], float]:
        """
        Find the pattern that best matches current positions.

        Args:
            actual_positions: Current ball positions
            target_hoop: Optional - only consider patterns for this hoop

        Returns:
            Tuple of (best_pattern, similarity_score)
        """
        patterns = self.patterns
        if target_hoop:
            patterns = [p for p in patterns if p.target_hoop == target_hoop]

        if not patterns:
            return None, 0.0

        best_pattern = None
        best_score = -1.0

        for pattern in patterns:
            score = self.evaluate_leave_similarity(actual_positions, pattern)
            # Weight by pattern quality
            weighted_score = score * pattern.quality_score

            if weighted_score > best_score:
                best_score = weighted_score
                best_pattern = pattern

        return best_pattern, best_score

    def get_bootstrap_position_values(self) -> Dict[str, Dict[str, float]]:
        """
        Extract position value bootstrapping data from patterns.

        Returns a dictionary mapping ball positions to value estimates,
        which can be used to initialize the learning system.

        Returns:
            Dict of {ball_color: {position_key: value}}
        """
        values = {}

        for pattern in self.patterns:
            for color, (x, y) in pattern.ball_positions.items():
                if color not in values:
                    values[color] = {}

                # Create a position key (quantized to 2-yard grid)
                pos_key = f"{int(x/2)*2},{int(y/2)*2}"

                # Accumulate value based on pattern quality
                if pos_key in values[color]:
                    values[color][pos_key] = max(
                        values[color][pos_key],
                        pattern.quality_score
                    )
                else:
                    values[color][pos_key] = pattern.quality_score

        return values

    def suggest_leave_positions(
        self,
        striker_color: str,
        striker_next_hoop: int,
        current_positions: Dict[str, Vector2]
    ) -> Dict[str, Vector2]:
        """
        Suggest target positions for a good leave based on patterns.

        Args:
            striker_color: Color of the ball making the leave
            striker_next_hoop: Next hoop for the striker
            current_positions: Current ball positions

        Returns:
            Dict of suggested positions for each ball
        """
        # Find best pattern for this situation
        patterns = self.get_patterns_for_hoop(striker_next_hoop, striker_color)

        if not patterns:
            # No specific pattern, try any pattern for this hoop
            patterns = self.get_patterns_for_hoop(striker_next_hoop)

        if not patterns:
            # Fallback to diagonal pattern
            patterns = self.get_pattern_by_type(LeaveType.DIAGONAL)[:1]

        if not patterns:
            return current_positions  # No suggestions available

        # Use the highest quality pattern
        best_pattern = max(patterns, key=lambda p: p.quality_score)

        # Convert pattern positions to Vector2
        suggested = {}
        for color, (x, y) in best_pattern.ball_positions.items():
            suggested[color] = Vector2(x, y)

        return suggested


# Singleton instance for easy access
_pattern_library = None

def get_pattern_library() -> LeavePatternLibrary:
    """Get the global pattern library instance."""
    global _pattern_library
    if _pattern_library is None:
        _pattern_library = LeavePatternLibrary()
    return _pattern_library
