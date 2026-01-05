"""
Association Croquet Simulator - Main Entry Point

A fully AI-driven croquet simulator that plays games automatically.
Watch as AI players compete following Association Croquet rules.
"""
import pygame
import sys
import time
from typing import Dict, List, Optional
from enum import Enum, auto

import config
from models.ball import Ball, Vector2
from models.court import Court
from physics.physics_engine import PhysicsEngine
from rules.rule_engine import RuleEngine, TurnState
from ai.ai_controller import AIController
from ai.basic_strategy import BasicStrategy
from ai.learning_strategy import LearningStrategy
from ai.learning.learner import CroquetLearner
from ai.learning.position_evaluator import PositionEvaluator
from view.renderer import Renderer


class GamePhase(Enum):
    """Phases of the game."""
    STARTING = auto()      # Initial setup
    THINKING = auto()      # AI is selecting a shot
    SHOOTING = auto()      # Ball is in motion
    TURN_END = auto()      # Processing turn results
    GAME_OVER = auto()     # Game finished


class CroquetSimulator:
    """
    Main game class that runs the AI-driven simulation.

    Follows Association Croquet rules:
    - Turn order: Blue, Red, Black, Yellow
    - One stroke per turn unless you earn extras via:
      - Running a hoop (1 continuation stroke)
      - Roqueting a ball (croquet stroke + continuation)
    - Deadness: Can't roquet same ball again until you run a hoop
    """

    def __init__(self):
        """Initialize the simulator."""
        pygame.init()
        pygame.display.set_caption(config.TITLE)

        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Game components
        self.court = Court()
        self.balls = self._create_balls()
        self.physics = PhysicsEngine(self.court)
        self.rules = RuleEngine()
        self.renderer = Renderer(self.screen)
        self.ai_controllers = self._create_ai_controllers()

        # Learning system
        self.learner = CroquetLearner()
        self.position_evaluator = PositionEvaluator()
        self.pre_shot_value = 0.0  # Track position value before shot

        # Set up strategic yard line placement callback
        self.physics.yard_line_placement_callback = self._strategic_yard_line_placement

        # Game state
        self.running = True
        self.phase = GamePhase.STARTING
        self.current_ball_index = 0
        self.turn_order = config.TURN_ORDER  # Only used for opening 4 turns

        # Track which balls are in play (started from baulk)
        self.balls_in_play: Dict[str, bool] = {
            "blue": False,
            "red": False,
            "black": False,
            "yellow": False,
        }

        # Innings tracking - which side has the turn
        # After opening, the side can choose either of their balls
        self.current_side = "blue_black"  # Starts with blue/black
        self.opening_complete = False  # True after all 4 balls played in
        self.current_ball_color_override: Optional[str] = None  # For side's choice

        # Double-tap prevention: Track which ball each side played last
        # Under AC Laws, same ball shouldn't play consecutively (except if only ball)
        self.last_ball_played: Dict[str, Optional[str]] = {
            "blue_black": None,
            "red_yellow": None
        }

        # Track collisions during shot
        self.shot_collisions: List[Dict] = []

        # Display
        self.message = ""
        self.message_timer = 0
        self.event_log: List[str] = []

        # Timing
        self.think_start_time = 0
        self.ai_think_delay = config.AI_THINK_DELAY

        # Statistics
        self.turn_count = 0
        self.shots_taken = 0

        # Track last shot info for learning
        self.last_shot_type = None
        self.last_rules_applied = []

    def _create_balls(self) -> Dict[str, Ball]:
        """Create all four croquet balls at randomized baulk entry positions.

        Randomizes which team starts from which baulk to eliminate positional bias.
        This addresses the issue where blue/black won all 6 recorded games.
        """
        import random
        balls = {}

        # Randomize baulk assignments each game to eliminate starting position bias
        # 50% chance: blue/black from A-baulk (south), red/yellow from B-baulk (north)
        # 50% chance: red/yellow from A-baulk (south), blue/black from B-baulk (north)
        if random.random() < 0.5:
            a_baulk_team = ["blue", "black"]
            b_baulk_team = ["red", "yellow"]
            print("  Starting positions: Blue/Black from A-baulk (south), Red/Yellow from B-baulk (north)")
        else:
            a_baulk_team = ["red", "yellow"]
            b_baulk_team = ["blue", "black"]
            print("  Starting positions: Red/Yellow from A-baulk (south), Blue/Black from B-baulk (north)")

        # Place balls on their assigned baulk lines
        for color in ["blue", "black", "red", "yellow"]:
            if color in a_baulk_team:
                # A-baulk: random x position in west half, y=1 (south yard-line)
                x = random.uniform(2, 12)  # Stay away from corners
                y = 1
            else:
                # B-baulk: random x position in east half, y=34 (north yard-line)
                x = random.uniform(16, 26)  # Stay away from corners
                y = 34

            balls[color] = Ball(color, (x, y))

        return balls

    def _create_ai_controllers(self) -> Dict[str, AIController]:
        """Create AI controllers with learning strategies and varied personalities."""
        import random
        controllers = {}

        # Base skill levels with random variation for each game
        base_skills = {
            "blue": 0.85,
            "black": 0.80,
            "red": 0.75,
            "yellow": 0.70,
        }

        for color, base_skill in base_skills.items():
            # Add random skill variation (±0.1) to make each game different
            skill = base_skill + random.uniform(-0.1, 0.1)
            skill = max(0.5, min(0.95, skill))  # Clamp to reasonable range

            # Use learning strategy for smarter play
            strategy = LearningStrategy(skill_level=skill)

            # Random aggression level for opening strategy
            aggression = random.uniform(0.3, 0.8)
            controllers[color] = AIController(strategy, aggression=aggression)

        return controllers

    @property
    def current_ball_color(self) -> str:
        """Get the current ball color based on game phase."""
        if self.current_ball_color_override:
            return self.current_ball_color_override
        return self.turn_order[self.current_ball_index]

    @property
    def current_ball(self) -> Ball:
        return self.balls[self.current_ball_color]

    def _select_ball_for_side(self, side: str) -> str:
        """
        AI selects which ball to play for the side with innings.

        In Association Croquet, after the opening, the side can choose
        either of their balls to play.

        DOUBLE-TAP PREVENTION: Under AC Laws, the same ball shouldn't play
        consecutively for the same side (unless it's the only ball available).

        Uses a comprehensive evaluation considering:
        1. Double-tap prevention (must alternate if both balls available)
        2. Immediate hoop-running opportunity
        3. Break potential (roquet options, not wired)
        4. Position quality from the position evaluator
        5. Strategic catch-up for trailing balls

        Args:
            side: "blue_black" or "red_yellow"

        Returns:
            Color of the ball to play
        """
        if side == "blue_black":
            balls = ["blue", "black"]
        else:
            balls = ["red", "yellow"]

        # Filter out pegged-out balls
        available = [c for c in balls if not self.balls[c].has_pegged_out]
        if not available:
            return balls[0]  # Shouldn't happen, but fallback

        if len(available) == 1:
            return available[0]

        # DOUBLE-TAP PREVENTION: Remove the ball that just played (if both available)
        last_played = self.last_ball_played.get(side)
        if last_played and last_played in available and len(available) > 1:
            # Must play the other ball - can't double-tap
            available = [c for c in available if c != last_played]
            if len(available) == 1:
                print(f"  [DOUBLE-TAP] {side} must play {available[0]} (can't repeat {last_played})")
                return available[0]

        # Get balls that are actually in play for evaluation
        balls_in_play = {c: b for c, b in self.balls.items() if self.balls_in_play.get(c, False)}

        best_ball = available[0]
        best_score = float('-inf')

        for color in available:
            ball = self.balls[color]
            score = 0.0

            # 1. Evaluate position using the position evaluator (comprehensive features)
            features = self.position_evaluator.extract_features(
                ball, self.balls, self.court, self.rules.deadness
            )

            # Base position score
            position_score = self.position_evaluator.evaluate(
                ball, self.balls, self.court, self.rules.deadness
            )
            score += position_score * 0.5  # Weight position evaluation

            # 2. Immediate hoop-running opportunity (high priority)
            target_hoop = self.court.get_hoop_for_ball(ball.hoops_run)
            if target_hoop:
                to_hoop = target_hoop.position - ball.position
                distance = to_hoop.magnitude()

                if features.threatens_hoop:
                    score += 100  # Strong incentive for immediate hoop run
                elif features.is_in_good_position:
                    score += 50  # Good position, can likely make the hoop
                elif distance < 5:
                    score += 20  # At least close to the hoop

            # 3. Break potential - roquet options accounting for wire detection
            # features.can_roquet_count already accounts for wiring
            score += features.can_roquet_count * 15  # Each live, non-wired target is valuable

            # 4. Penalty for being wired (limits options)
            if features.is_wired:
                # Check how severely wired
                wired_count = len(features.wired_from)
                score -= wired_count * 10  # Penalty for each ball we're wired from

            # 5. Strategic catch-up bonus for trailing ball
            partner_color = self._get_partner_color(color)
            if partner_color:
                partner = self.balls[partner_color]
                hoop_difference = partner.hoops_run - ball.hoops_run
                if hoop_difference > 0:
                    # Bonus for catching up, scaled by how far behind
                    score += min(hoop_difference * 5, 25)

            # 6. Consider if partner is threatening their hoop
            # If partner is about to score, maybe let them continue
            if features.partner_threatens_hoop:
                score -= 15  # Slight disincentive to switch if partner has opportunity

            # 7. Defensive consideration - if opponent threatens, prefer to block
            if features.opponent_threatens_hoop:
                # Check if this ball can roquet the threatening opponent
                opponent_colors = ["red", "yellow"] if side == "blue_black" else ["blue", "black"]
                for opp_color in opponent_colors:
                    if opp_color not in features.wired_from and opp_color not in self.rules.deadness.get(color, set()):
                        opp_hoop = self.court.get_hoop_for_ball(self.balls[opp_color].hoops_run)
                        if opp_hoop:
                            opp_dist = (self.balls[opp_color].position - opp_hoop.position).magnitude()
                            if opp_dist < 4:
                                score += 20  # Can potentially disrupt opponent's hoop run

            if score > best_score:
                best_score = score
                best_ball = color

        return best_ball

    def _strategic_yard_line_placement(self, ball: Ball, exit_position: Vector2, boundary_hit: str) -> Optional[Vector2]:
        """
        Choose optimal yard line placement for a ball that went out of bounds.

        Evaluates all valid yard line positions using the position evaluator
        and returns the strategically best position.

        Args:
            ball: The ball that went out
            exit_position: Where the ball exited the court
            boundary_hit: Which boundary was hit ('north', 'south', 'east', 'west')

        Returns:
            Best position on the yard line, or None to use default
        """
        # Get all valid yard line positions
        valid_positions = self.physics.get_valid_yard_line_positions(boundary_hit, exit_position)

        if not valid_positions:
            return None

        # Determine which side this ball is on
        side = config.BALL_TEAMS[ball.color]

        # Get balls that are in play (excluding the one being placed)
        balls_for_eval = {c: b for c, b in self.balls.items()
                        if c != ball.color and self.balls_in_play.get(c, False)}

        best_position = valid_positions[0]
        best_score = float('-inf')

        for pos in valid_positions:
            # Temporarily set ball position
            original_pos = ball.position.copy()
            ball.position = pos

            # Evaluate position
            # Include the ball being placed in evaluation
            eval_balls = dict(balls_for_eval)
            eval_balls[ball.color] = ball

            score = self.position_evaluator.evaluate_side(
                side, eval_balls, self.court, self.rules.deadness
            )

            # Additional factors for yard line placement:
            # 1. Prefer positions closer to partner ball
            partner_color = self._get_partner_color(ball.color)
            if partner_color and partner_color in balls_for_eval:
                partner = balls_for_eval[partner_color]
                dist_to_partner = (pos - partner.position).magnitude()
                score -= dist_to_partner * 0.5  # Prefer closer to partner

            # 2. Prefer positions further from opponents
            for opp_color, opp_ball in balls_for_eval.items():
                if config.BALL_TEAMS[opp_color] != side:
                    dist_to_opp = (pos - opp_ball.position).magnitude()
                    score += dist_to_opp * 0.3  # Prefer further from opponents

            # 3. Consider approach angle to next hoop
            target_hoop = self.court.get_hoop_for_ball(ball.hoops_run)
            if target_hoop:
                to_hoop = target_hoop.position - pos
                if to_hoop.magnitude() > 0.5:
                    approach_dir = to_hoop.normalize()
                    angle_quality = max(0, approach_dir.dot(target_hoop.direction))
                    score += angle_quality * 5  # Reward good approach angle

            # Restore position
            ball.position = original_pos

            if score > best_score:
                best_score = score
                best_position = pos

        return best_position

    def _get_partner_color(self, ball_color: str) -> Optional[str]:
        """Get the partner ball color for a given ball."""
        if ball_color == "blue":
            return "black"
        elif ball_color == "black":
            return "blue"
        elif ball_color == "red":
            return "yellow"
        elif ball_color == "yellow":
            return "red"
        return None

    def _handle_lift_entitlement(self):
        """
        Handle Advanced Play lift entitlement.

        When the opponent has run 1-back or 4-back, the side with the lift
        may place one of their balls on either baulk line.

        The AI chooses strategically:
        - Which ball to lift (usually the one in worse position)
        - Which baulk line (based on opponent ball positions)
        """
        side = self.current_side
        if side == "blue_black":
            balls = ["blue", "black"]
        else:
            balls = ["red", "yellow"]

        # Choose which ball to lift - pick the one in worse position
        ball_to_lift = None
        worst_score = float('inf')

        for color in balls:
            ball = self.balls[color]
            if ball.has_pegged_out:
                continue

            # Score position - lower is worse
            score = 0
            target_hoop = self.court.get_hoop_for_ball(ball.hoops_run)
            if target_hoop:
                dist = (ball.position - target_hoop.position).magnitude()
                score = -dist  # Further from hoop = worse

            # Bonus for being on boundary (already in good lift position)
            if ball.position.x < 2 or ball.position.x > self.court.width - 2:
                score += 5  # Already near boundary, less need to lift
            if ball.position.y < 2 or ball.position.y > self.court.height - 2:
                score += 5

            if score < worst_score:
                worst_score = score
                ball_to_lift = color

        if ball_to_lift is None:
            self.rules.clear_lift_entitlement()
            return

        # Choose baulk line - prefer the one further from opponents
        opponent_balls = ["red", "yellow"] if side == "blue_black" else ["blue", "black"]
        a_baulk_y = 1  # South
        b_baulk_y = self.court.height - 1  # North

        # Calculate average opponent Y position
        opp_avg_y = sum(self.balls[c].position.y for c in opponent_balls
                        if not self.balls[c].has_pegged_out) / 2

        # Choose baulk line further from opponents
        if abs(opp_avg_y - a_baulk_y) > abs(opp_avg_y - b_baulk_y):
            baulk = "A"
        else:
            baulk = "B"

        # Execute the lift
        new_position = self.rules.use_lift(ball_to_lift, baulk, self.court)
        self.balls[ball_to_lift].position = new_position
        self._add_event(f"[LIFT] {ball_to_lift.capitalize()} lifted to {baulk}-baulk")

    def _handle_wiring_lift(self, ball_color: str):
        """
        Handle wiring lift when a ball is wired from ALL other balls.

        Under AC Laws, if a ball is wired from all other balls at the
        start of its turn, it may be lifted to either baulk line.

        This is different from the Advanced Play lift (1-back/4-back) -
        it's a general fairness rule to prevent unfair wiring.

        Args:
            ball_color: Color of the ball that is wired
        """
        ball = self.balls[ball_color]

        # Determine which side this ball is on
        side = "blue_black" if ball_color in ["blue", "black"] else "red_yellow"

        # Choose baulk line - prefer the one further from opponents
        opponent_balls = ["red", "yellow"] if side == "blue_black" else ["blue", "black"]
        a_baulk_y = 1  # South
        b_baulk_y = self.court.height - 1  # North

        # Calculate average opponent Y position
        active_opps = [c for c in opponent_balls if not self.balls[c].has_pegged_out]
        if active_opps:
            opp_avg_y = sum(self.balls[c].position.y for c in active_opps) / len(active_opps)
        else:
            opp_avg_y = self.court.height / 2

        # Choose baulk line further from opponents
        if abs(opp_avg_y - a_baulk_y) > abs(opp_avg_y - b_baulk_y):
            baulk = "A"
            new_y = a_baulk_y
        else:
            baulk = "B"
            new_y = b_baulk_y

        # Place ball on chosen baulk line (center of baulk)
        new_position = Vector2(self.court.width / 2, new_y)
        ball.position = new_position
        self._add_event(f"[WIRING LIFT] {ball_color.capitalize()} lifted to {baulk}-baulk")
        print(f"  [WIRING LIFT] {ball_color} placed on {baulk}-baulk at {new_position}")

    def handle_events(self):
        """Handle Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.phase == GamePhase.THINKING:
                        self.think_start_time = 0
                elif event.key == pygame.K_r:
                    self._reset_game()
                elif event.key == pygame.K_f:
                    # Fast forward - reduce think delay
                    self.ai_think_delay = 0.05

    def _reset_game(self):
        """Reset the game with fresh randomization."""
        self.balls = self._create_balls()  # Randomized baulk positions
        self.ai_controllers = self._create_ai_controllers()  # Fresh random personalities
        self.rules = RuleEngine()
        self.current_ball_index = 0
        self.phase = GamePhase.STARTING
        self.turn_count = 0
        self.shots_taken = 0
        self.event_log = []
        self.shot_collisions = []
        self.balls_in_play = {c: False for c in self.turn_order}
        self.ai_think_delay = config.AI_THINK_DELAY
        # Reset innings tracking
        self.current_side = "blue_black"
        self.opening_complete = False
        self.current_ball_color_override = None
        # Reset double-tap tracking
        self.last_ball_played = {"blue_black": None, "red_yellow": None}
        self._add_event("Game restarted")

    def _add_event(self, text: str):
        """Add an event to the log."""
        self.event_log.append(text)
        if len(self.event_log) > 10:
            self.event_log.pop(0)
        self.message = text
        self.message_timer = 1.5

    def update(self, dt: float):
        """Update game state."""
        if self.message_timer > 0:
            self.message_timer -= dt

        if self.phase == GamePhase.STARTING:
            self._handle_starting()
        elif self.phase == GamePhase.THINKING:
            self._handle_thinking()
        elif self.phase == GamePhase.SHOOTING:
            self._handle_shooting(dt)
        elif self.phase == GamePhase.TURN_END:
            self._handle_turn_end()

    def _handle_starting(self):
        """Start the game."""
        self._add_event(f"Game Start - {self.current_ball_color.capitalize()}'s turn")
        self.rules.start_turn(self.current_ball_color)
        self.phase = GamePhase.THINKING
        self.think_start_time = time.time()

    def _handle_thinking(self):
        """AI thinks and shoots."""
        if time.time() - self.think_start_time < self.ai_think_delay:
            return

        ball = self.current_ball
        ai = self.ai_controllers[ball.color]

        # Check if this is ball's first shot (entering from baulk)
        is_entering_from_baulk = not self.balls_in_play[ball.color]

        # Get balls that are actually in play for AI to consider
        # Note: Current ball is NOT yet in play if this is its first shot
        balls_in_play = {c: b for c, b in self.balls.items() if self.balls_in_play[c]}

        # Check if we need to take a croquet stroke
        is_croquet_stroke = False
        if self.rules.turn_info and self.rules.turn_info.state == TurnState.CROQUET_REQUIRED:
            # Get the roqueted ball
            roqueted_color = self.rules.turn_info.just_roqueted
            if roqueted_color and roqueted_color in self.balls:
                roqueted_ball = self.balls[roqueted_color]

                # Place striker ball in contact with roqueted ball
                # AI chooses placement strategically
                placement_pos = ai.select_croquet_placement(
                    ball, roqueted_ball, balls_in_play, self.court
                )
                ball.position = placement_pos

                # Get croquet stroke - returns velocities for both balls
                striker_vel, croqueted_vel, description, stroke_type = ai.select_croquet_shot(
                    ball, roqueted_ball, balls_in_play, self.court, self.rules.deadness
                )

                is_croquet_stroke = True
                self.last_stroke_was_croquet = True

                # Check if this croquet shot placed a pioneer (based on strategy description)
                if "pioneer" in description.lower():
                    self.last_pioneer_placed = True
                else:
                    self.last_pioneer_placed = False

                # Update state - croquet stroke being taken
                self.rules.turn_info.state = TurnState.CROQUET_TAKEN
            else:
                # Fallback to normal shot if something went wrong
                velocity, description = ai.select_shot(
                    ball, balls_in_play, self.court,
                    balls_in_play=self.balls_in_play
                )
        else:
            # Normal shot - In Association Croquet, each ball aims for its own next hoop
            # Pass balls_in_play status for opening strategy detection
            # Also pass continuation context so AI can prioritize hoop running
            is_continuation = (
                self.rules.turn_info and
                self.rules.turn_info.state in [TurnState.CONTINUATION, TurnState.CROQUET_TAKEN]
            )
            strokes_left = self.rules.turn_info.strokes_remaining if self.rules.turn_info else 1

            velocity, description = ai.select_shot(
                ball, balls_in_play, self.court,
                balls_in_play=self.balls_in_play,
                deadness=self.rules.deadness,
                strokes_remaining=strokes_left,
                is_continuation=is_continuation
            )
            self.last_stroke_was_croquet = False
            self.last_pioneer_placed = False

        # Capture shot type info from strategy for learning
        strategy = ai.strategy
        if hasattr(strategy, 'last_shot_type') and strategy.last_shot_type:
            self.last_shot_type = strategy.last_shot_type.name
        else:
            self.last_shot_type = "unknown"

        if hasattr(strategy, 'tactical_rules') and hasattr(strategy.tactical_rules, 'rules_applied'):
            self.last_rules_applied = strategy.tactical_rules.rules_applied.copy()
        else:
            self.last_rules_applied = []

        # Debug: show shot info (can be removed later)
        # target_hoop = self.court.get_hoop_for_ball(ball.hoops_run)
        # if target_hoop:
        #     print(f"  {ball.color} at {ball.position} -> hoop {ball.hoops_run + 1}")

        # Store previous position for hoop detection
        ball.previous_position = ball.position.copy()

        # Record pre-shot position value for learning
        self.pre_shot_value = self.position_evaluator.evaluate_side(
            "blue_black" if ball.color in ["blue", "black"] else "red_yellow",
            balls_in_play, self.court, self.rules.deadness
        )

        # Execute shot
        if is_croquet_stroke:
            # Croquet stroke moves both balls
            self.physics.execute_croquet_stroke(ball, roqueted_ball, striker_vel, croqueted_vel)
        else:
            # Normal shot - only striker moves
            self.physics.shoot_ball(ball, velocity)

        self.shot_collisions = []
        self.shots_taken += 1

        # Mark ball as in play AFTER shot selection (for proper opening detection)
        if is_entering_from_baulk:
            self.balls_in_play[ball.color] = True
            self._add_event(f"{ball.color.capitalize()} enters from baulk")

        self._add_event(description)
        self.phase = GamePhase.SHOOTING

    def _handle_shooting(self, dt: float):
        """Update physics while balls are moving."""
        events = self.physics.update(self.balls, dt)
        self.shot_collisions.extend(events)

        # Display hoop collision events immediately
        for event in events:
            if event.get('type') == 'hoop_hit':
                self._add_event(f"{event['ball'].capitalize()} hit hoop {event['hoop']} upright!")
            elif event.get('type') == 'jaws':
                self._add_event(f"{event['ball'].capitalize()} stuck in jaws of hoop {event['hoop']}!")

        if self.physics.are_all_balls_stopped(self.balls):
            self.phase = GamePhase.TURN_END

    def _handle_turn_end(self):
        """Process end of stroke for Association Croquet."""
        ball = self.current_ball

        # Process the stroke result through rules engine
        turn_continues, events = self.rules.process_stroke_result(
            ball, self.balls, self.court, self.shot_collisions
        )

        # Calculate post-shot position value for learning
        balls_in_play = {c: b for c, b in self.balls.items() if self.balls_in_play[c]}
        post_shot_value = self.position_evaluator.evaluate_side(
            "blue_black" if ball.color in ["blue", "black"] else "red_yellow",
            balls_in_play, self.court, self.rules.deadness
        )

        # Record experience for learning
        hoop_run = any("runs hoop" in e for e in events)
        roqueted = None
        for e in events:
            if "roquets" in e:
                # Extract roqueted ball from event
                for c in ["blue", "black", "red", "yellow"]:
                    if c in e.lower() and c != ball.color:
                        roqueted = c
                        break

        # Determine stroke type for learning
        stroke_type = "standard"
        if hasattr(self, 'last_stroke_was_croquet') and self.last_stroke_was_croquet:
            stroke_type = "croquet"
        elif self.rules.turn_info and self.rules.turn_info.state in [TurnState.CONTINUATION, TurnState.CROQUET_TAKEN]:
            stroke_type = "continuation"

        # Calculate approach distance and angle for hoop attempts
        approach_distance = 0.0
        approach_angle = 0.0
        target_hoop = self.court.get_hoop_for_ball(ball.hoops_run)
        if target_hoop and hasattr(ball, 'previous_position'):
            # Distance from where ball was before shot
            approach_distance = (ball.previous_position - target_hoop.position).magnitude()
            # Angle quality (how aligned with hoop direction)
            to_hoop = target_hoop.position - ball.previous_position
            if to_hoop.magnitude() > 0.5:
                approach_dir = to_hoop.normalize()
                approach_angle = max(0, approach_dir.dot(target_hoop.direction))

        # Check if a pioneer was placed or rush achieved (from AI strategy)
        pioneer_placed = False
        rush_achieved = False
        if hasattr(self, 'last_pioneer_placed'):
            pioneer_placed = self.last_pioneer_placed
        if roqueted:
            # A roquet gives potential for rush on next stroke
            rush_achieved = True

        self.learner.record_experience(
            striker=ball,
            all_balls=balls_in_play,
            deadness=self.rules.deadness,
            shot_angle=0,  # We'd need to track this
            shot_power=0,
            hoop_run=hoop_run,
            roqueted=roqueted,
            turn_continued=turn_continues,
            old_position_value=self.pre_shot_value,
            new_position_value=post_shot_value,
            shot_type=self.last_shot_type,
            rules_applied=self.last_rules_applied,
            # Enhanced tracking
            stroke_type=stroke_type,
            approach_distance=approach_distance,
            approach_angle=approach_angle,
            pioneer_placed=pioneer_placed,
            rush_achieved=rush_achieved,
            court=self.court,
        )

        # Record leave quality when turn ends
        if not turn_continues:
            self.learner.record_leave_quality(post_shot_value, balls_in_play, self.court)

        for event in events:
            self._add_event(event)

        # MARK IN: Place balls in yard line area onto the yard line
        # Exception: striker's ball if they still have strokes remaining
        striker_has_strokes = turn_continues
        mark_in_events = self.physics.mark_in_all_balls(
            self.balls,
            striker_color=ball.color,
            striker_has_strokes=striker_has_strokes
        )
        for mark_event in mark_in_events:
            self._add_event(f"{mark_event['ball'].capitalize()} marked in to yard line")

        # Check for game over
        if self._check_game_over():
            self.phase = GamePhase.GAME_OVER
            self._show_winner()
            return

        if turn_continues:
            # Continue with same ball (earned extra stroke)
            self.phase = GamePhase.THINKING
            self.think_start_time = time.time()
        else:
            # Turn ended - record which ball played for double-tap tracking
            ball_side = config.BALL_TEAMS[ball.color]
            self.last_ball_played[ball_side] = ball.color

            # Next ball's turn
            self._next_turn()

    def _next_turn(self):
        """Move to next turn - handles both opening and normal play."""
        self.turn_count += 1

        # Check if opening is complete (all 4 balls have been played in)
        if not self.opening_complete:
            balls_played = sum(1 for v in self.balls_in_play.values() if v)
            if balls_played >= 4:
                self.opening_complete = True

        if not self.opening_complete:
            # During opening: strict B, R, Bk, Y sequence
            self.current_ball_index = (self.current_ball_index + 1) % len(self.turn_order)
            self.current_ball_color_override = None  # Use turn_order
            # Update current side based on ball
            ball_color = self.turn_order[self.current_ball_index]
            self.current_side = config.BALL_TEAMS[ball_color]
        else:
            # After opening: innings passes to other side, who chooses a ball
            # Toggle side
            if self.current_side == "blue_black":
                self.current_side = "red_yellow"
            else:
                self.current_side = "blue_black"

            # ADVANCED PLAY: Check for lift entitlement (from 1-back/4-back)
            if self.rules.check_lift_available(self.current_side):
                self._handle_lift_entitlement()

            # Side chooses which ball to play
            chosen_ball = self._select_ball_for_side(self.current_side)
            self.current_ball_color_override = chosen_ball

            # WIRING LIFT: Check if chosen ball is wired from ALL other balls
            # Under AC Laws, this grants a lift entitlement
            chosen_ball_obj = self.balls[chosen_ball]
            if self.rules.check_wiring_lift(chosen_ball_obj, self.balls, self.court):
                self._add_event(f"[WIRING LIFT] {chosen_ball.capitalize()} is wired from all balls!")
                self._handle_wiring_lift(chosen_ball)

        self._add_event(f"{self.current_ball_color.capitalize()}'s turn")
        self.rules.start_turn(self.current_ball_color)

        # Start a new break for learning - track all strokes in this turn
        self.learner.start_new_break()

        self.phase = GamePhase.THINKING
        self.think_start_time = time.time()

    def _check_game_over(self) -> bool:
        """Check if game is over (both balls of a side have pegged out)."""
        blue_black = self.balls["blue"].has_pegged_out and self.balls["black"].has_pegged_out
        red_yellow = self.balls["red"].has_pegged_out and self.balls["yellow"].has_pegged_out
        return blue_black or red_yellow

    def _show_winner(self):
        """Display winner for Association Croquet."""
        bb_score = sum(self.balls[c].hoops_run + (1 if self.balls[c].has_pegged_out else 0)
                       for c in ["blue", "black"])
        ry_score = sum(self.balls[c].hoops_run + (1 if self.balls[c].has_pegged_out else 0)
                       for c in ["red", "yellow"])

        if bb_score > ry_score:
            winner = "Blue/Black"
            winner_side = "blue_black"
        elif ry_score > bb_score:
            winner = "Red/Yellow"
            winner_side = "red_yellow"
        else:
            winner = "Draw"
            winner_side = "draw"

        # Record game outcome for learning
        self.learner.record_game_outcome(
            winner_side=winner_side,
            final_scores={"blue_black": bb_score, "red_yellow": ry_score}
        )

        # Display learning stats
        stats = self.learner.get_stats()
        print(f"\n=== Learning Stats ===")
        print(f"Games played: {stats['games_played']}")
        print(f"Avg hoops/game: {stats['avg_hoops_per_game']:.2f}")
        print(f"Avg roquets/game: {stats['avg_roquets_per_game']:.2f}")
        print(f"Experiences recorded: {stats['experiences_recorded']}")

        # Enhanced stats
        if 'break_stats' in stats and stats['break_stats'].get('avg_break_length', 0) > 0:
            print(f"Avg break length: {stats['break_stats']['avg_break_length']:.2f} hoops")
            print(f"Best break: {stats['break_stats']['best_break']} hoops")

        if 'croquet_shot_stats' in stats:
            cs = stats['croquet_shot_stats']
            if cs['pioneer_placements']['attempts'] > 0:
                useful_rate = cs['pioneer_placements']['useful'] / cs['pioneer_placements']['attempts'] * 100
                print(f"Pioneer effectiveness: {useful_rate:.0f}%")

        if 'optimal_approaches' in stats and stats['optimal_approaches']:
            print(f"Learned optimal approaches for {len(stats['optimal_approaches'])} hoops")

        self._add_event(f"Game Over! {winner} wins! ({bb_score}-{ry_score})")

    def render(self):
        """Render game state."""
        bb_score = sum(self.balls[c].hoops_run + (1 if self.balls[c].has_pegged_out else 0)
                       for c in ["blue", "black"])
        ry_score = sum(self.balls[c].hoops_run + (1 if self.balls[c].has_pegged_out else 0)
                       for c in ["red", "yellow"])

        strokes = self.rules.turn_info.strokes_remaining if self.rules.turn_info else 1

        turn_info = {
            "current_ball": self.current_ball_color,
            "strokes_remaining": strokes,
            "scores": {"Blue/Black": bb_score, "Red/Yellow": ry_score},
            "turn": self.turn_count,
            "shots": self.shots_taken,
            "phase": self.phase.name,
            "deadness": self.rules.get_deadness_display(),
        }

        self.renderer.render(self.court, self.balls, self.current_ball_color, turn_info)

        # Removed distracting black bar message overlay
        # Messages are still logged to the event log at the bottom

        # Draw event log
        self._draw_event_log()

        pygame.display.flip()

    def _draw_event_log(self):
        """Draw recent events."""
        font = pygame.font.Font(None, 20)
        y = config.SCREEN_HEIGHT - 20 * len(self.event_log) - 10

        for event in self.event_log:
            text = font.render(event, True, config.WHITE)
            self.screen.blit(text, (10, y))
            y += 20

    def run(self):
        """Main game loop."""
        print("=" * 50)
        print("Association Croquet Simulator")
        print("=" * 50)
        print()
        print("Fully AI-driven simulation following Association Croquet rules.")
        print()
        print("Controls:")
        print("  SPACE - Skip thinking time")
        print("  F     - Fast forward (reduce delays)")
        print("  R     - Restart game")
        print("  ESC   - Quit")
        print()

        while self.running:
            dt = self.clock.tick(config.FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.render()

        # Save learning state before exiting
        print("\nSaving learning data...")
        self.learner._save_state()
        stats = self.learner.get_stats()
        print(f"Saved: {stats['experiences_recorded']} experiences, {stats['total_hoops_run']} hoops run")

        pygame.quit()
        sys.exit()


def main():
    simulator = CroquetSimulator()
    simulator.run()


if __name__ == "__main__":
    main()
