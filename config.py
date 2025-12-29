"""
Configuration constants for the Association Croquet Simulator.

Based on standard court layout:
- Court: 28 yards wide (4 units) x 35 yards tall (5 units)
- 1 unit = 7 yards
- Corner hoops 1 unit (7 yards) from boundaries
- Center hoops 1 unit (7 yards) from peg
- Baulk lines: on yard-line level with hoops 1 and 4
"""

# Display settings
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 1000
FPS = 60
TITLE = "Association Croquet Simulator"

# Colors (RGB)
GRASS_GREEN = (34, 139, 34)  # Forest green
GRASS_DARK = (28, 110, 28)   # Darker green for contrast
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BOUNDARY_WHITE = (255, 255, 255)

# Ball colors - bright and distinct
BALL_COLORS = {
    "blue": (30, 80, 200),
    "black": (20, 20, 20),
    "red": (220, 40, 40),
    "yellow": (240, 220, 50),
}

# Hoop and peg colors
HOOP_COLOR = (255, 255, 255)  # White hoops
HOOP_BLUE = (50, 100, 200)    # Blue top for hoop 1
HOOP_RED = (200, 50, 50)      # Red top for rover hoop
PEG_RED = (200, 50, 50)
PEG_WHITE = (255, 255, 255)

# Court dimensions (in yards)
# Standard croquet court: 28 yards wide x 35 yards long
# This is 4 units x 5 units where 1 unit = 7 yards
COURT_WIDTH_YARDS = 28   # 4 units (East-West)
COURT_HEIGHT_YARDS = 35  # 5 units (North-South)
UNIT_YARDS = 7           # 1 unit = 7 yards
BOUNDARY_MARGIN_YARDS = 1  # Yard line from edge

# Scaling - fit court on screen with margins
MARGIN_PX = 50
SCALE_X = (SCREEN_WIDTH - 2 * MARGIN_PX) / COURT_WIDTH_YARDS
SCALE_Y = (SCREEN_HEIGHT - 2 * MARGIN_PX) / COURT_HEIGHT_YARDS
YARDS_TO_PIXELS = min(SCALE_X, SCALE_Y)  # Use smaller to fit

# Calculated court pixel dimensions
COURT_WIDTH_PX = int(COURT_WIDTH_YARDS * YARDS_TO_PIXELS)
COURT_HEIGHT_PX = int(COURT_HEIGHT_YARDS * YARDS_TO_PIXELS)

# Court offset (to center on screen)
COURT_OFFSET_X = (SCREEN_WIDTH - COURT_WIDTH_PX) // 2
COURT_OFFSET_Y = (SCREEN_HEIGHT - COURT_HEIGHT_PX) // 2

# Physics constants
BALL_RADIUS_YARDS = 0.1  # Ball radius for collision detection
BALL_RADIUS_PX = 10      # Display size
BALL_MASS = 0.454        # kg (16 oz)
FRICTION_COEFFICIENT = 0.4  # Grass friction
RESTITUTION = 0.8        # Collision bounce coefficient - higher for better rush transfers
GRAVITY = 9.81           # m/s^2
MIN_VELOCITY = 0.1       # Below this, ball stops (yards/s)
MAX_SHOT_POWER = 15.0    # Maximum initial velocity (yards/s) - allows longer shots

# Hoop dimensions
HOOP_WIDTH_YARDS = 0.15  # ~5 inches gap
HOOP_WIDTH_PX = 16       # Display width
HOOP_HEIGHT_PX = 24      # Visual height

# Hoop positions (in yards from bottom-left corner)
# Based on the standard layout image:
#
#   Court: 28 yards wide x 35 yards tall
#   Center: (14, 17.5)
#   Peg at center
#
#   Corner hoops: 7 yards from each edge
#   Center hoops: 7 yards north/south of peg
#
#       0         7        14        21       28
#   35  +------------------------------------+
#       |                                    |
#   28  |    [2]                      [3]    |  y=28 (7 from top)
#       |     ^                        |     |
#       |     |                        v     |  2=run N, 3=run S
#       |                                    |
#  24.5 |               [6]                  |  (7 north of center)
#       |                |                   |  6=run S (rover)
#       |                v                   |
#  17.5 |               PEG                  |  CENTER
#       |                ^                   |
#       |                |                   |
#  10.5 |               [5]                  |  (7 south of center), 5=run N
#       |                                    |
#       |     ^                        |     |
#       |     |                        v     |  1=run N, 4=run S
#    7  |    [1]                      [4]    |  y=7 (7 from bottom)
#       |                                    |
#    0  +------------------------------------+
#       South boundary
#
# Golf Croquet hoop order: 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 (twice around)
# First to 7 points wins
#
HOOP_POSITIONS = [
    # Hoop 1: SW corner, run NORTH (approach from south)
    {"num": 1, "pos": (7, 7), "direction": (0, 1), "color": "blue"},
    # Hoop 2: NW corner, run NORTH (approach from south, continue from hoop 1)
    {"num": 2, "pos": (7, 28), "direction": (0, 1), "color": "white"},
    # Hoop 3: NE corner, run SOUTH (approach from north after crossing from hoop 2)
    {"num": 3, "pos": (21, 28), "direction": (0, -1), "color": "white"},
    # Hoop 4: SE corner, run SOUTH (approach from north, continue from hoop 3)
    {"num": 4, "pos": (21, 7), "direction": (0, -1), "color": "white"},
    # Hoop 5: Center-South (penult), run NORTH (approach from south after crossing from hoop 4)
    {"num": 5, "pos": (14, 10.5), "direction": (0, 1), "color": "white"},
    # Hoop 6: Center-North (rover in AC), run SOUTH (approach from north, then to peg)
    {"num": 6, "pos": (14, 24.5), "direction": (0, -1), "color": "red"},
]

# Peg position (center of court)
PEG_POSITION = (14, 17.5)
PEG_RADIUS_PX = 6

# Baulk lines - where balls enter play from
# In Association Croquet, balls are played into the game from either baulk line
# Both baulk lines run HALF the width of the court (14 yards), 1 yard in from boundary
#
# A-baulk (South): On the SOUTH yard-line, from CENTER to WEST edge
#   - y = 1 (1 yard from south boundary)
#   - x = 0 to 14 (west half of court)
#
# B-baulk (North): On the NORTH yard-line, from CENTER to EAST edge
#   - y = 34 (1 yard from north boundary, since court is 35 yards)
#   - x = 14 to 28 (east half of court)
#
BAULK_A_START = (0, 1)       # SW corner
BAULK_A_END = (14, 1)        # Center of south yard-line
BAULK_B_START = (14, 34)     # Center of north yard-line
BAULK_B_END = (28, 34)       # NE corner

# Ball starting - balls start OFF the court and are played in
# None means ball hasn't entered play yet
STARTING_POSITIONS = {
    "blue": None,    # Will be played from baulk
    "red": None,
    "black": None,
    "yellow": None,
}

# Baulk line entry positions (where balls start when played in)
# In Association Croquet, balls can be played from EITHER baulk line
# A-baulk: y=1, x=0 to x=14 (south yard-line, west half)
# B-baulk: y=34, x=14 to x=28 (north yard-line, east half)
BAULK_A_POSITIONS = [(x, 1) for x in range(0, 15)]    # A-baulk spots (south, west half)
BAULK_B_POSITIONS = [(x, 34) for x in range(14, 29)]  # B-baulk spots (north, east half)

# Default starting positions - balls can choose either baulk
# Traditional: Blue/Black often start from A-baulk (south), Red/Yellow from B-baulk (north)
BAULK_ENTRY_POSITIONS = {
    "blue": (7, 1),      # A-baulk (south) - good angle to hoop 1
    "red": (21, 34),     # B-baulk (north) - can aim at hoop 2 or join
    "black": (10, 1),    # A-baulk (south) - near center
    "yellow": (18, 34),  # B-baulk (north) - near center
}

# Turn order (alternating sides: blue/black vs red/yellow)
# Blue and Black are partners, Red and Yellow are partners
TURN_ORDER = ["blue", "red", "black", "yellow"]

# Teams/Sides
TEAMS = {
    "blue_black": ["blue", "black"],
    "red_yellow": ["red", "yellow"],
}

# Which team each ball belongs to
BALL_TEAMS = {
    "blue": "blue_black",
    "black": "blue_black",
    "red": "red_yellow",
    "yellow": "red_yellow",
}

# AI settings
AI_THINK_DELAY = 0.5     # Seconds before AI shoots
SHOT_ANGLE_SAMPLES = 24  # Angles to try
SHOT_POWER_LEVELS = 5    # Power levels to try

# Golf Croquet settings
WINNING_SCORE = 7        # First to 7 wins
HOOP_SEQUENCE = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]  # Play hoops in order

# Association Croquet - Second circuit hoop mapping
# After running hoops 1-6, the second circuit uses the SAME physical hoops
# but in a DIFFERENT ORDER and run from the OPPOSITE direction.
#
# First circuit directions: 1=N, 2=N, 3=S, 4=S, 5=N, 6=S
# Second circuit sequence (reversed directions from first circuit):
#   Hoop 7 (1-back):      Physical hoop 2 (NW), run SOUTH (was North for hoop 2)
#   Hoop 8 (2-back):      Physical hoop 1 (SW), run SOUTH (was North for hoop 1)
#   Hoop 9 (3-back):      Physical hoop 4 (SE), run NORTH (was South for hoop 4)
#   Hoop 10 (4-back):     Physical hoop 3 (NE), run NORTH (was South for hoop 3)
#   Hoop 11 (penultimate): Physical hoop 6 (Center-N), run NORTH (was South for hoop 6)
#   Hoop 12 (rover):      Physical hoop 5 (Center-S), run SOUTH (was North for hoop 5)
#
# This maps hoops_run (0-11) to (physical_hoop_number, direction)
# Second circuit reverses the direction of each hoop from the first circuit
AC_SECOND_CIRCUIT = {
    6: (2, (0, -1)),   # Hoop 7 (1-back): physical hoop 2, run SOUTH (was NORTH in 1st circuit)
    7: (1, (0, -1)),   # Hoop 8 (2-back): physical hoop 1, run SOUTH (was NORTH in 1st circuit)
    8: (4, (0, 1)),    # Hoop 9 (3-back): physical hoop 4, run NORTH (was SOUTH in 1st circuit)
    9: (3, (0, 1)),    # Hoop 10 (4-back): physical hoop 3, run NORTH (was SOUTH in 1st circuit)
    10: (6, (0, 1)),   # Hoop 11 (penultimate): physical hoop 6, run NORTH (was SOUTH in 1st circuit)
    11: (5, (0, -1)),  # Hoop 12 (rover): physical hoop 5, run SOUTH (was NORTH in 1st circuit)
}

# Association Croquet hoop order (12 hoops total)
# First circuit: 1, 2, 3, 4, 5, 6 (standard positions and directions)
# Second circuit: 1-back, 2-back, 3-back, 4-back, penultimate, rover
#                 (physical hoops 2, 1, 4, 3, 6, 5 in opposite directions)
AC_HOOP_SEQUENCE = [1, 2, 3, 4, 5, 6, 2, 1, 4, 3, 6, 5]  # Physical hoop numbers
