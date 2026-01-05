"""
AI Controller - Manages AI decision making for shots.

Coordinates between:
- Opening strategy: First 4 turns with tice/supershot placements
- Break planning: 2/3/4-ball break building with pilot/pioneer/pivot
- Croquet strokes: Drive, roll, take-off, split shots
- Peel strategy: Triple peel planning and execution
- Aiton tactics: Standard leaves (NSL/MSL/Diagonal), approach assessment,
                 standard strokes, 3-to-4 ball transitions
- Tactical Decision Maker: Q-value based shot selection with learning

Based on Keith Aiton's "The Basics" teachings for approach quality,
leave positioning, and break building fundamentals.
"""
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import config
from models.ball import Ball, Vector2
from models.court import Court
from ai.basic_strategy import BasicStrategy

# Try to import neural network components
try:
    from ai.neural.croquet_net import CroquetNet, check_torch_available
    NEURAL_AVAILABLE = check_torch_available()
except ImportError:
    NEURAL_AVAILABLE = False
    CroquetNet = None
from ai.break_strategy import BreakPlanner, BreakPlan
from ai.opening_strategy import OpeningPlanner
from ai.peel_strategy import PeelPlanner, PeelState, PeelOpportunity
from ai.aiton_tactics import AitonTactics, LeaveType, ApproachQuality
from ai.expert_tactics import ExpertTactics, TacticalSituation, PositionVsShootDecision
from ai.tactical_decision_maker import TacticalDecisionMaker, ShotOption, ShotType
from physics.croquet_strokes import CroquetStrokeCalculator, StrokeType as CroquetStrokeType, StrokeResult
# Alias for backward compatibility in this file
StrokeType_Croquet = CroquetStrokeType


class AIController:
    """
    Controls AI decision-making for a ball/player.

    Uses opening strategy for early game, break planning for mid-game,
    croquet stroke selection for strategic play, and peel strategy
    for triple peel attempts.
    """

    # Default path for trained neural network model
    DEFAULT_NEURAL_MODEL_PATH = "ai_data/neural/model.pt"

    def __init__(
        self,
        strategy=None,
        aggression: float = 0.5,
        court: Court = None,
        learner=None,
        use_neural: bool = False,
        neural_model_path: str = None
    ):
        """
        Initialize AI controller.

        Args:
            strategy: Strategy to use (defaults to BasicStrategy)
            aggression: How aggressive to play openings (0-1)
            court: Court reference for Aiton tactics
            learner: CroquetLearner for tactical learning (optional)
            use_neural: Whether to use neural network for shot selection
            neural_model_path: Path to trained neural model (default: ai_data/neural/model.pt)
        """
        self.strategy = strategy or BasicStrategy(skill_level=0.75)
        self.break_planner = BreakPlanner()
        self.opening_planner = OpeningPlanner(aggression=aggression)
        self.stroke_calculator = CroquetStrokeCalculator()
        self.peel_planner = PeelPlanner(skill_level=0.75)
        self.current_break_plan: Optional[BreakPlan] = None
        self._court = court
        self._aiton_tactics: Optional[AitonTactics] = None
        self._expert_tactics: Optional[ExpertTactics] = None
        self._learner = learner
        self._tactical_dm: Optional[TacticalDecisionMaker] = None

        # Neural network settings
        self._use_neural = use_neural
        self._neural_model_path = neural_model_path or self.DEFAULT_NEURAL_MODEL_PATH
        self._neural_net = None

        # Load neural network if requested
        if use_neural:
            self._load_neural_net()

        # Track shooting performance for confidence calculations
        self._recent_shots: list = []  # (hit: bool) for last N shots
        self._max_shot_history = 10

        # Use tactical decision making by default
        self.use_tactical_dm = True

    def _get_aiton_tactics(self, court: Court) -> AitonTactics:
        """Get or create Aiton tactics instance."""
        if self._aiton_tactics is None or self._court != court:
            self._court = court
            self._aiton_tactics = AitonTactics(court)
        return self._aiton_tactics

    def _get_expert_tactics(self, court: Court) -> ExpertTactics:
        """Get or create Expert tactics instance."""
        if self._expert_tactics is None or self._court != court:
            self._court = court
            self._expert_tactics = ExpertTactics(court)
        return self._expert_tactics

    def _get_tactical_dm(self) -> TacticalDecisionMaker:
        """Get or create Tactical Decision Maker instance."""
        if self._tactical_dm is None:
            self._tactical_dm = TacticalDecisionMaker(
                learner=self._learner,
                neural_net=self._neural_net,
                use_neural=self._use_neural and self._neural_net is not None
            )
        return self._tactical_dm

    def _load_neural_net(self):
        """Load neural network model if available."""
        if not NEURAL_AVAILABLE:
            print("  [AI] Neural network not available (PyTorch not installed)")
            self._use_neural = False
            return

        model_path = Path(self._neural_model_path)
        if not model_path.exists():
            print(f"  [AI] Neural model not found at {model_path}")
            print("       Run 'python train_neural.py' to train a model")
            self._use_neural = False
            return

        try:
            self._neural_net = CroquetNet.load(str(model_path))
            self._neural_net.eval()
            print(f"  [AI] Loaded neural network from {model_path}")
        except Exception as e:
            print(f"  [AI] Failed to load neural network: {e}")
            self._use_neural = False

    def enable_neural(self, model_path: str = None):
        """Enable neural network for shot selection."""
        if model_path:
            self._neural_model_path = model_path
        self._use_neural = True
        self._load_neural_net()
        # Reset tactical DM to pick up the neural network
        self._tactical_dm = None

    def disable_neural(self):
        """Disable neural network, use hand-crafted Q-values."""
        self._use_neural = False
        self._tactical_dm = None

    def set_learner(self, learner):
        """Set the learner for tactical learning."""
        self._learner = learner
        # Reset tactical DM so it picks up new learner
        self._tactical_dm = None

    def record_shot_result(self, hit: bool):
        """
        Record the result of a shot for tracking shooting form.

        From Aiton-Maugham game: Keith went "0 hits and 10 misses"
        which affected his confidence and shooting quality.

        Args:
            hit: Whether the shot hit its target
        """
        self._recent_shots.append(hit)
        if len(self._recent_shots) > self._max_shot_history:
            self._recent_shots.pop(0)

    def get_shooting_form(self) -> float:
        """
        Get current shooting form based on recent results.

        Returns:
            Float 0-1 representing shooting confidence/form
        """
        if not self._recent_shots:
            return 0.5  # Neutral starting form

        hits = sum(1 for h in self._recent_shots if h)
        return hits / len(self._recent_shots)

    def get_recent_misses(self) -> int:
        """Get count of recent consecutive misses."""
        misses = 0
        for shot in reversed(self._recent_shots):
            if not shot:
                misses += 1
            else:
                break
        return misses

    def select_shot(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        court: Court,
        target_hoop_num: int = None,
        balls_in_play: Dict[str, bool] = None,
        deadness: Dict[str, set] = None,
        strokes_remaining: int = 1,
        is_continuation: bool = False
    ) -> Tuple[Vector2, str]:
        """
        Select a shot for the given ball.

        Uses TacticalDecisionMaker for Q-value based shot selection when enabled,
        falling back to strategy-based selection otherwise.

        Args:
            ball: The ball to shoot
            balls: All balls on the court
            court: The court
            target_hoop_num: Which hoop to aim for (Golf Croquet mode, None for Association)
            balls_in_play: Which balls have entered play (for opening detection)
            deadness: Which balls the striker is dead on
            strokes_remaining: Strokes remaining in this turn
            is_continuation: Whether this is a continuation stroke

        Returns:
            Tuple of (velocity Vector2, description string)
        """
        # Check if we're in opening phase
        if balls_in_play is not None and self.opening_planner.is_opening_phase(balls_in_play):
            return self._select_opening_shot(ball, balls, balls_in_play, court)

        # Initialize deadness if not provided
        if deadness is None:
            deadness = {c: set() for c in ["blue", "black", "red", "yellow"]}

        # Use TacticalDecisionMaker for intelligent shot selection
        if self.use_tactical_dm:
            return self._select_tactical_shot(
                ball, balls, court, deadness, strokes_remaining, is_continuation
            )

        # Fallback: Use legacy strategy-based selection
        return self._select_strategy_shot(
            ball, balls, court, target_hoop_num, deadness, strokes_remaining, is_continuation
        )

    def _select_tactical_shot(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int,
        is_continuation: bool
    ) -> Tuple[Vector2, str]:
        """
        Select shot using TacticalDecisionMaker (Q-value based).

        This method uses learned parameters and expert tactics to
        evaluate all shot options and select the best one.
        """
        dm = self._get_tactical_dm()

        # Get game state for context
        game_state = {
            'strokes_remaining': strokes_remaining,
            'is_continuation': is_continuation,
            'shooting_form': self.get_shooting_form(),
            'recent_misses': self.get_recent_misses(),
        }

        # Get best shot option
        best_option = dm.select_best_shot(
            striker=ball,
            balls=balls,
            court=court,
            deadness=deadness,
            strokes_remaining=strokes_remaining,
            game_state=game_state,
            use_expert_tactics=True
        )

        # Store last decision for learning
        self._last_shot_option = best_option

        # Convert shot option to velocity and description
        velocity, description = self._shot_option_to_velocity(ball, best_option, court)

        # Print tactical reasoning
        print(f"  [TACTICAL] {best_option.description}")
        print(f"    Expected value: {best_option.expected_value:.2f}, "
              f"Success: {best_option.success_probability:.0%}, "
              f"Risk: {best_option.risk_factor:.0%}")

        return (velocity, description)

    def _shot_option_to_velocity(
        self,
        ball: Ball,
        option: ShotOption,
        court: Court
    ) -> Tuple[Vector2, str]:
        """Convert a ShotOption to a velocity vector."""
        to_target = option.target - ball.position
        distance = to_target.magnitude()
        angle = math.atan2(to_target.y, to_target.x)

        # Calculate power using physics-based formula: v = sqrt(2 * friction_decel * distance)
        # This matches the physics engine's deceleration model
        friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY

        # Base velocity needed to travel the distance
        if distance > 0:
            base_power = math.sqrt(2 * friction_decel * distance)
        else:
            base_power = 0.0

        # Adjust power based on shot type
        # NOTE: base_power formula is accurate to ~1% in simulation.
        # Multipliers:
        # - 1.0x lands ~1% past target (acceptable for most shots)
        # - 1.01x lands ~3% past (good margin for contact shots)
        # OLD multipliers (1.1-1.15x) were overhitting by 20-34%!
        if option.shot_type == ShotType.HOOP_RUN:
            # Hoop running: need to pass through hoop, target is ~2 yards past
            power = base_power * 1.0  # Land on target
            power = min(power, config.MAX_SHOT_POWER * 0.6)
        elif option.shot_type == ShotType.ROQUET:
            # Roquet: need to reach and contact target ball
            power = base_power * 1.01  # Tiny margin for contact
            power = min(power, config.MAX_SHOT_POWER * 0.8)
        elif option.shot_type == ShotType.APPROACH:
            # Approach shot - stop at the target position
            power = base_power * 1.0
            power = min(power, config.MAX_SHOT_POWER * 0.5)
        elif option.shot_type == ShotType.PEG_OUT:
            # Peg out needs to hit the peg
            power = base_power * 1.01
            power = min(power, config.MAX_SHOT_POWER * 0.7)
        else:  # DEFENSIVE, LEAVE
            # Controlled shot to boundary
            power = base_power * 1.0
            power = min(power, config.MAX_SHOT_POWER * 0.6)

        # Apply learned power adjustment
        if self._learner and hasattr(self._learner, 'power_adjustment'):
            power *= self._learner.power_adjustment

        velocity = Vector2.from_angle(angle, power)
        return (velocity, option.description)

    def _select_strategy_shot(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        court: Court,
        target_hoop_num: int,
        deadness: Dict[str, set],
        strokes_remaining: int,
        is_continuation: bool
    ) -> Tuple[Vector2, str]:
        """Legacy strategy-based shot selection."""
        # Get the target hoop
        if target_hoop_num is not None:
            target_hoop = court.get_hoop(target_hoop_num)
            hoop_num = target_hoop_num
        else:
            # Association Croquet: ball aims for its own next hoop
            target_hoop = court.get_hoop_for_ball(ball.hoops_run)
            hoop_num = ball.get_next_hoop_number()

        # Determine if this is a run attempt or positioning shot
        is_running = False
        if target_hoop and hasattr(self.strategy, '_is_good_approach'):
            is_running = self.strategy._is_good_approach(ball, target_hoop)

        # Use strategy to get angle and power, passing continuation context
        if hasattr(self.strategy, 'select_shot'):
            # Learning strategy supports extra parameters
            try:
                angle, power = self.strategy.select_shot(
                    ball, balls, court, target_hoop_num,
                    deadness=deadness,
                    strokes_remaining=strokes_remaining,
                    is_continuation=is_continuation
                )
            except TypeError:
                # Fallback for strategies that don't support extra params
                angle, power = self.strategy.select_shot(ball, balls, court, target_hoop_num)
        else:
            angle, power = self.strategy.select_shot(ball, balls, court, target_hoop_num)

        # Convert to velocity vector
        velocity = Vector2.from_angle(angle, power)

        # Generate description based on shot type
        if target_hoop and hoop_num > 0:
            if is_running:
                description = f"{ball.color.capitalize()} runs at hoop {hoop_num}"
            else:
                description = f"{ball.color.capitalize()} sets up for hoop {hoop_num}"
        elif ball.is_rover:
            description = f"{ball.color.capitalize()} aims for peg"
        else:
            description = f"{ball.color.capitalize()} shoots"

        return (velocity, description)

    def _select_opening_shot(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        balls_in_play: Dict[str, bool],
        court: Court
    ) -> Tuple[Vector2, str]:
        """
        Select an opening shot using opening strategy.

        Args:
            ball: The ball to shoot
            balls: All balls on court
            balls_in_play: Which balls have entered play
            court: The court

        Returns:
            Tuple of (velocity, description)
        """
        # Count turn number based on balls in play
        turn_number = sum(1 for v in balls_in_play.values() if v) + 1

        # Get opening shot recommendation
        target, power, description = self.opening_planner.get_opening_shot(
            ball, balls, balls_in_play, court, turn_number
        )

        # Convert target position to velocity
        to_target = target - ball.position
        angle = math.atan2(to_target.y, to_target.x)
        velocity = Vector2.from_angle(angle, power)

        return (velocity, f"{ball.color.capitalize()}: {description}")

    def select_croquet_placement(
        self,
        striker: Ball,
        roqueted: Ball,
        balls: Dict[str, Ball],
        court: Court
    ) -> Vector2:
        """
        Select where to place the striker ball for a croquet stroke.

        In croquet, after roqueting a ball, you place your ball in contact
        with the roqueted ball, then hit your ball causing both to move.

        Strategic placement depends on where you want to send both balls.

        Args:
            striker: The striker ball
            roqueted: The ball that was roqueted
            balls: All balls on the court
            court: The court

        Returns:
            Position to place the striker ball
        """
        # Get the next hoop for the striker
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)

        if target_hoop:
            # Place ball so we can rush toward the hoop
            # Direction from roqueted ball to hoop
            to_hoop = target_hoop.position - roqueted.position
            to_hoop_dir = to_hoop.normalize()

            # Place striker ball behind roqueted ball (opposite direction to hoop)
            # This way we can send roqueted ball toward hoop and follow
            contact_distance = (striker.radius + roqueted.radius) * 2 + 0.05
            placement = roqueted.position - to_hoop_dir * contact_distance
        else:
            # Default: place to one side
            placement = roqueted.position + Vector2(0.15, 0)

        # Ensure placement is within court bounds
        margin = 0.5
        placement.x = max(margin, min(court.width - margin, placement.x))
        placement.y = max(margin, min(court.height - margin, placement.y))

        return placement

    def select_croquet_shot(
        self,
        striker: Ball,
        roqueted: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set] = None
    ) -> Tuple[Vector2, Vector2, str, CroquetStrokeType]:
        """
        Select the direction and power for a croquet stroke.

        KEY PRINCIPLE (from Wylie): The croquet stroke is the heart of break-building.
        It should achieve TWO objectives simultaneously:
        1. Send the croqueted ball to a STRATEGIC position (pioneer at next hoop, or useful location)
        2. Position the striker to get a RUSH on another LIVE ball toward the current hoop

        The pattern is:
        - Roquet ball A -> Croquet: send A as pioneer, get position for rush on B
        - Rush B to hoop -> Run hoop -> Rush to next ball -> Repeat

        Args:
            striker: The striker ball (placed in contact with roqueted)
            roqueted: The ball that was roqueted
            balls: All balls on the court
            court: The court
            deadness: Which balls striker is dead on

        Returns:
            Tuple of (striker_velocity, croqueted_velocity, description, stroke_type)
        """
        if deadness is None:
            deadness = {c: set() for c in ["blue", "black", "red", "yellow"]}

        # Get which balls we're dead on (including the one we just roqueted)
        dead_on = deadness.get(striker.color, set())
        # The roqueted ball is now dead - we can't roquet it again until we run a hoop
        dead_on = dead_on | {roqueted.color}

        # Find LIVE balls we can rush after this croquet stroke
        live_balls = []
        for color, ball in balls.items():
            if color != striker.color and color not in dead_on and not ball.has_pegged_out:
                live_balls.append((color, ball))

        # Update break plan
        self.current_break_plan = self.break_planner.analyze_position(
            striker, balls, court, deadness
        )

        # Get the target hoops
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)
        next_hoop = court.get_hoop_for_ball(striker.hoops_run + 1)
        hoop_after_next = court.get_hoop_for_ball(striker.hoops_run + 2)

        # Print detailed croquet planning info to show the strategic thinking
        print(f"\n  [CROQUET PLANNING] {striker.color} taking croquet on {roqueted.color}")
        print(f"    Striker's next hoop: {striker.hoops_run + 1}")
        print(f"    Live balls available: {[c for c, _ in live_balls] if live_balls else 'NONE'}")
        if target_hoop:
            dist_to_hoop = (target_hoop.position - striker.position).magnitude()
            print(f"    Distance to hoop {striker.hoops_run + 1}: {dist_to_hoop:.1f} yards")
        if next_hoop:
            print(f"    Pioneer target: hoop {striker.hoops_run + 2} at ({next_hoop.position.x:.1f}, {next_hoop.position.y:.1f})")

        # Check for PEEL OPPORTUNITY if croqueted ball is partner
        partner_color = {"blue": "black", "black": "blue", "red": "yellow", "yellow": "red"}.get(striker.color)
        peel_result = None

        if roqueted.color == partner_color:
            # Taking croquet on partner - check for peel opportunity
            peel_result = self._check_peel_opportunity(striker, roqueted, balls, court)

            if peel_result:
                # Peel opportunity found - execute it
                return peel_result

        # STRATEGIC CROQUET SHOT SELECTION (4-ball break pattern)
        # Priority 1: If we have live balls, position to RUSH one toward hoop
        # Priority 2: Send croqueted ball as PIONEER to next hoop
        # Priority 3: If no live balls, position for direct hoop approach

        if target_hoop and live_balls:
            # Find the best ball to rush and calculate ideal positions
            best_rush_ball, striker_target, rush_description = self._find_best_rush_setup(
                striker, live_balls, target_hoop, court
            )

            if best_rush_ball:
                # We have a live ball to rush - this is ideal break play!
                # Send croqueted ball as pioneer, position for rush
                pioneer_dest = "center (pivot)"
                if next_hoop:
                    # Pioneer position: 3-4 yards in front of next hoop
                    croqueted_target = next_hoop.position - next_hoop.direction * 4
                    pioneer_dest = f"hoop {striker.hoops_run + 2}"
                elif hoop_after_next:
                    croqueted_target = hoop_after_next.position - hoop_after_next.direction * 4
                    pioneer_dest = f"hoop {striker.hoops_run + 3}"
                else:
                    # Send to center as pivot
                    croqueted_target = Vector2(court.width / 2, court.height / 2)

                stroke_type, aim_dir, power, split_angle = self.stroke_calculator.select_best_stroke(
                    striker, roqueted, striker_target, croqueted_target, court
                )

                result = self.stroke_calculator.calculate_stroke(
                    stroke_type, striker, roqueted, aim_dir, power, split_angle
                )

                # Clear strategic description showing the 4-ball break plan
                print(f"    [BREAK PLAN] Split shot:")
                print(f"      - Send {roqueted.color} as PIONEER to {pioneer_dest}")
                print(f"      - Position striker to {rush_description}")
                print(f"      - Then: rush -> approach hoop -> run hoop -> continue break")
                description = f"{striker.color.capitalize()} {result.description}: send {roqueted.color} to {pioneer_dest}, {rush_description}"
                return (result.striker_velocity, result.croqueted_velocity, description, stroke_type)

        # No live balls to rush - simpler approach (not ideal break play)
        print(f"    [NO RUSH AVAILABLE] Must use direct approach - break not building optimally")
        if target_hoop:
            to_hoop = target_hoop.position - striker.position
            distance_to_hoop = to_hoop.magnitude()

            if distance_to_hoop < 5:
                # Already close to hoop - take-off to run it, send croqueted away
                striker_target = target_hoop.position - target_hoop.direction * 2
                pioneer_dest = "center"
                if next_hoop:
                    croqueted_target = next_hoop.position - next_hoop.direction * 4
                    pioneer_dest = f"hoop {striker.hoops_run + 2}"
                else:
                    croqueted_target = Vector2(court.width / 2, court.height / 2)

                stroke_type, aim_dir, power, split_angle = self.stroke_calculator.select_best_stroke(
                    striker, roqueted, striker_target, croqueted_target, court
                )

                result = self.stroke_calculator.calculate_stroke(
                    stroke_type, striker, roqueted, aim_dir, power, split_angle
                )

                print(f"    [PLAN] Take-off: approach hoop {striker.hoops_run + 1}, send {roqueted.color} to {pioneer_dest}")
                description = f"{striker.color.capitalize()} {result.description}: approach hoop, send {roqueted.color} to {pioneer_dest}"
            else:
                # Far from hoop, no rush available - drive toward hoop
                striker_target = target_hoop.position - target_hoop.direction * 4
                pioneer_dest = "center"
                if next_hoop:
                    croqueted_target = next_hoop.position - next_hoop.direction * 4
                    pioneer_dest = f"hoop {striker.hoops_run + 2}"
                else:
                    croqueted_target = Vector2(court.width / 2, court.height / 2)

                stroke_type, aim_dir, power, split_angle = self.stroke_calculator.select_best_stroke(
                    striker, roqueted, striker_target, croqueted_target, court
                )

                result = self.stroke_calculator.calculate_stroke(
                    stroke_type, striker, roqueted, aim_dir, power, split_angle
                )

                print(f"    [PLAN] Drive toward hoop {striker.hoops_run + 1}, send {roqueted.color} to {pioneer_dest}")
                description = f"{striker.color.capitalize()} {result.description}: drive to hoop, {roqueted.color} as pioneer"
        else:
            # Rover - aim toward peg
            peg_pos = court.peg_position
            aim_dir = (peg_pos - roqueted.position).normalize()
            result = self.stroke_calculator.calculate_stroke(
                CroquetStrokeType.DRIVE, striker, roqueted, aim_dir, 6.0
            )
            stroke_type = CroquetStrokeType.DRIVE
            description = f"{striker.color.capitalize()} drives toward peg"

        return (result.striker_velocity, result.croqueted_velocity, description, stroke_type)

    def _find_best_rush_setup(
        self,
        striker: Ball,
        live_balls: list,
        target_hoop,
        court: Court
    ) -> Tuple[Optional[Ball], Optional[Vector2], str]:
        """
        Find the best live ball to set up a rush toward the hoop.

        The ideal croquet shot positions the striker BEHIND a live ball,
        so the striker can then roquet (rush) that ball toward the hoop.

        Returns:
            Tuple of (ball to rush, striker target position, description)
            Returns (None, None, "") if no good rush available
        """
        best_ball = None
        best_position = None
        best_score = 0
        best_description = ""

        for color, ball in live_balls:
            # Calculate where striker should be to rush this ball toward hoop
            # Ideal: striker behind ball, with ball between striker and hoop
            ball_to_hoop = target_hoop.position - ball.position
            ball_to_hoop_dist = ball_to_hoop.magnitude()

            if ball_to_hoop_dist < 1:
                continue  # Ball too close to hoop

            rush_direction = ball_to_hoop.normalize()

            # Striker should be 2-4 yards behind the ball (opposite to rush direction)
            ideal_striker_pos = ball.position - rush_direction * 3

            # Check if this is a reasonable position (1 yard from boundary)
            if not court.is_in_bounds(ideal_striker_pos, radius=1):
                continue

            # Score this rush setup
            # Factors: distance to ball, alignment quality, ball distance to hoop
            striker_to_ball = (ball.position - ideal_striker_pos).magnitude()

            # Good rush: straight or slight cut (not too angled)
            # The alignment is already good by construction

            # Prefer balls that are closer to the hoop (shorter rush needed)
            hoop_proximity_score = max(0, 1 - ball_to_hoop_dist / 15)

            # Prefer balls that are easier to reach
            reach_score = max(0, 1 - striker_to_ball / 10)

            # Combined score
            score = hoop_proximity_score * 0.6 + reach_score * 0.4

            if score > best_score:
                best_score = score
                best_ball = ball
                best_position = ideal_striker_pos
                best_description = f"rush {color} to hoop"

        # Only return if we found a decent option
        if best_score > 0.3:
            return (best_ball, best_position, best_description)

        return (None, None, "")

    def _check_peel_opportunity(
        self,
        striker: Ball,
        partner: Ball,
        balls: Dict[str, Ball],
        court: Court
    ) -> Optional[Tuple[Vector2, Vector2, str, CroquetStrokeType]]:
        """
        Check for and execute a peel opportunity during croquet stroke.

        This is where triple peel magic happens! When taking croquet on
        partner ball, check if we can peel them through their next hoop.

        Args:
            striker: The striker ball
            partner: The partner ball (being croqueted)
            balls: All balls on the court
            court: The court

        Returns:
            Croquet shot tuple if peel should be attempted, None otherwise
        """
        # Initialize TP tracking if not already
        if not self.peel_planner.peel_state:
            should_tp, reason = self.peel_planner.should_attempt_tp(
                striker, partner, balls, court
            )
            if should_tp:
                self.peel_planner.initialize_tp(striker, partner)
                print(f"  [TRIPLE PEEL] Initiating TP - {reason}")

        # Check for peel opportunity
        peel_opp = self.peel_planner.find_peel_opportunity(
            striker, partner, court, is_croquet_stroke=True
        )

        if not peel_opp:
            return None

        # Decide if we should prioritize the peel
        should_peel, reason = self.peel_planner.should_prioritize_peel(
            striker, partner, peel_opp, court
        )

        if not should_peel:
            print(f"  [PEEL] Skipping peel: {reason}")
            return None

        # Execute the peel!
        print(f"  [PEEL] Attempting {peel_opp.description} ({peel_opp.success_probability:.0%} success)")

        # Calculate the peel shot
        # For a peel, we want to send partner through their hoop
        # Striker position depends on where we need to be afterwards
        partner_hoop = court.get_hoop_for_ball(partner.hoops_run)
        if not partner_hoop:
            return None

        # Target for partner: through the hoop
        peel_target = partner_hoop.position + partner_hoop.direction * 3

        # Striker target: position for continuing break
        # After peel, we want to be in position to continue
        striker_target_hoop = court.get_hoop_for_ball(striker.hoops_run)
        if striker_target_hoop:
            striker_target = striker_target_hoop.position - striker_target_hoop.direction * 4
        else:
            striker_target = Vector2(court.width / 2, court.height / 2)

        # Select best stroke type for the peel
        stroke_type, aim_dir, power, split_angle = self.stroke_calculator.select_best_stroke(
            striker, partner, striker_target, peel_target, court
        )

        result = self.stroke_calculator.calculate_stroke(
            stroke_type, striker, partner, aim_dir, power * 1.1, split_angle  # Slightly more power for peel
        )

        description = f"{striker.color.capitalize()} PEEL: {peel_opp.description}"

        return (result.striker_velocity, result.croqueted_velocity, description, stroke_type)

    def check_tp_status(self) -> Optional[Dict]:
        """
        Get the current triple peel status.

        Returns:
            Dict with TP status info, or None if no TP in progress
        """
        if not self.peel_planner.peel_state:
            return None

        state = self.peel_planner.peel_state
        return {
            "partner": state.partner_color,
            "peels_needed": [state.get_peel_name(h) for h in state.peels_needed],
            "peels_completed": [state.get_peel_name(h) for h in state.peels_completed],
            "remaining": state.peels_remaining,
            "is_complete": state.is_complete
        }

    def set_strategy(self, strategy):
        """Change the AI strategy."""
        self.strategy = strategy

    def get_strategy_name(self) -> str:
        """Get the name of the current strategy."""
        return self.strategy.name

    def assess_approach(
        self,
        ball: Ball,
        court: Court
    ) -> Optional[ApproachQuality]:
        """
        Assess the quality of ball's current approach to its target hoop.

        Uses Aiton's principles:
        - 1 yard is ideal distance
        - Right side approaches easier than left
        - Angle significantly affects difficulty

        Args:
            ball: Ball to assess
            court: The court

        Returns:
            ApproachQuality assessment, or None if no target hoop
        """
        target_hoop = court.get_hoop_for_ball(ball.hoops_run)
        if not target_hoop:
            return None

        aiton = self._get_aiton_tactics(court)
        return aiton.assess_approach(ball, target_hoop)

    def plan_leave(
        self,
        striker_color: str,
        balls: Dict[str, Ball],
        court: Court,
        leave_type: LeaveType = LeaveType.NSL
    ) -> Dict:
        """
        Plan a standard leave position.

        AITON LEAVES (Section 2.5):
        - NSL: New Standard Leave - partner at hoop 2, opponents separated
        - MSL: Maugham Standard Leave - variation on NSL
        - Diagonal Spread: Maximum separation for defensive leave

        Args:
            striker_color: Color of striker's ball
            balls: All balls on court
            court: The court
            leave_type: Which standard leave to set up

        Returns:
            Dict with leave plan including target positions
        """
        aiton = self._get_aiton_tactics(court)
        leave = aiton.get_leave_positions(striker_color, balls, leave_type)

        return {
            'type': leave_type.name,
            'striker_target': leave.striker_pos,
            'partner_target': leave.partner_pos,
            'opponent1_target': leave.opponent1_pos,
            'opponent2_target': leave.opponent2_pos,
            'description': leave.description
        }

    def check_standard_stroke(
        self,
        ball: Ball,
        court: Court
    ) -> Optional[Dict]:
        """
        Check if ball is positioned for a standard Aiton stroke pattern.

        From Section 2.7: "The roll approach from corner II to hoop 2
        is another standard stroke"

        Args:
            ball: Ball to check
            court: The court

        Returns:
            Dict with stroke info if standard pattern found, None otherwise
        """
        aiton = self._get_aiton_tactics(court)
        stroke = aiton.find_standard_stroke(ball)

        if stroke:
            return {
                'name': stroke.name,
                'target': stroke.target_position,
                'stroke_type': stroke.stroke_type,
                'description': stroke.description
            }
        return None

    def plan_break_transition(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set]
    ) -> Optional[Dict]:
        """
        Plan transition from 3-ball to 4-ball break.

        AITON PRINCIPLE (Section 2.17-2.18):
        "The general best way to transition to a 3-ball break is to a 4-ball break"

        Args:
            striker: Ball playing the break
            balls: All balls
            court: The court
            deadness: Deadness information

        Returns:
            Transition plan dict, or None if not applicable
        """
        return self.break_planner.plan_3_to_4_ball_transition(
            striker, balls, court, deadness
        )

    def evaluate_break_pickup(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set]
    ) -> Tuple[float, str]:
        """
        Evaluate the quality of a potential break pickup.

        Uses Aiton's Section 2.6 principles for 4th turn pickup
        and break establishment.

        Args:
            striker: Ball attempting pickup
            balls: All balls
            court: The court
            deadness: Which balls striker is dead on

        Returns:
            Tuple of (quality 0-1, description)
        """
        aiton = self._get_aiton_tactics(court)
        return aiton.evaluate_break_pickup(striker, balls, deadness)

    # ========== EXPERT TACTICS (from Aiton-Maugham match) ==========

    def analyze_lift_situation(
        self,
        striker: Ball,
        opponent_balls: list,
        court: Court
    ) -> Dict:
        """
        Analyze lift implications for current position.

        From Aiton-Maugham game:
        - Turn 14: "mindful of the lift, goes to E boundary, maximum length position"
        - Turn 29: "guard the W side of the lawn in view of the impending lift"

        Args:
            striker: Striker's ball
            opponent_balls: List of opponent ball objects
            court: The court

        Returns:
            Dict with lift analysis and recommendations
        """
        expert = self._get_expert_tactics(court)
        lift_info = expert.analyze_lift_situation(striker, opponent_balls)

        return {
            'lift_pending': lift_info.lift_available,
            'lift_corners': lift_info.lift_corners,
            'baulk_available': lift_info.baulk_available,
            'recommended_leave': lift_info.recommended_leave,
            'danger_zones': [(v.x, v.y) for v in lift_info.danger_zones]
        }

    def analyze_impasse(
        self,
        striker: Ball,
        opponent: Ball,
        court: Court
    ) -> Dict:
        """
        Analyze if current situation is an impasse.

        From Aiton-Maugham game (turns 45-49):
        - Both players near 4-back, neither wanting to shoot first
        - Dave: "Hopeful of an impasse..."
        - Keith: "...or a 9 yarder"

        Args:
            striker: Striker's ball
            opponent: Opponent's ball nearest to contested hoop
            court: The court

        Returns:
            Dict with impasse analysis
        """
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)
        expert = self._get_expert_tactics(court)
        analysis = expert.analyze_impasse(striker, opponent, target_hoop)

        return {
            'is_impasse': analysis.is_impasse,
            'hoop_contested': analysis.hoop_contested,
            'distance_advantage': analysis.distance_advantage,
            'should_shoot': analysis.shoot_recommendation,
            'reasoning': analysis.reasoning
        }

    def should_shoot_or_position(
        self,
        striker: Ball,
        target_ball: Ball,
        court: Court,
        can_take_wired_position: bool = False,
        opponent_shooting_form: float = None
    ) -> Dict:
        """
        Decide whether to shoot at target or take position.

        From Aiton-Maugham game:
        - Turn 23: "take position wired from Y on the basis that Y won't
                   risk shot from B-baulk"
        - Turn 38: "the little voice in my head said 'What do you do after
                   you've run this then eh?'"

        Args:
            striker: Striker's ball
            target_ball: Ball to potentially shoot at
            court: The court
            can_take_wired_position: Is a wired position available?
            opponent_shooting_form: Opponent's current form (uses own if None)

        Returns:
            Dict with decision and reasoning
        """
        shot_distance = (target_ball.position - striker.position).magnitude()
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)

        # Use opponent form if provided, otherwise assume neutral
        opp_form = opponent_shooting_form if opponent_shooting_form is not None else 0.5

        expert = self._get_expert_tactics(court)
        decision = expert.position_vs_shoot_decision(
            striker=striker,
            target_ball=target_ball,
            shot_distance=shot_distance,
            can_take_wired_position=can_take_wired_position,
            target_hoop=target_hoop,
            opponent_shooting_form=opp_form
        )

        return {
            'should_shoot': decision.should_shoot,
            'shot_confidence': decision.shoot_confidence,
            'position_value': decision.position_value,
            'next_turn_consideration': decision.next_turn_consideration,
            'reasoning': decision.reasoning
        }

    def get_maximum_length_position(
        self,
        avoid_corners: list,
        court: Court
    ) -> Tuple[float, float]:
        """
        Calculate maximum length position avoiding specified lift corners.

        From turn 14: "goes to E boundary, maximum length position"

        Args:
            avoid_corners: List of corners to avoid ("I", "III")
            court: The court

        Returns:
            Tuple (x, y) of best position
        """
        expert = self._get_expert_tactics(court)
        pos = expert.calculate_maximum_length_position(avoid_corners)
        return (pos.x, pos.y)

    def evaluate_tpo_opportunity(
        self,
        striker: Ball,
        partner: Ball,
        opponent_balls: list,
        court: Court
    ) -> Dict:
        """
        Evaluate whether TPO (Triple Peel Out) is advisable.

        From turn 8 commentary:
        "I left 3 balls on in the Southerns final, and Keith finished
        the turn after contact. I thought I'd try something different."

        Args:
            striker: Striker's ball
            partner: Partner ball (to potentially peg out)
            opponent_balls: Opponent's balls
            court: The court

        Returns:
            Dict with TPO evaluation
        """
        # Get opponent's shooting form (if we have data)
        # In a real implementation, this would track opponent's shots
        opponent_form = 0.5  # Default to neutral

        expert = self._get_expert_tactics(court)
        should_tpo, reasoning = expert.evaluate_tpo_opportunity(
            striker, partner, opponent_balls, opponent_form
        )

        return {
            'should_attempt_tpo': should_tpo,
            'reasoning': reasoning
        }

    def get_guarding_position(
        self,
        side_to_guard: str,
        opponent_position: Vector2,
        court: Court
    ) -> Tuple[float, float]:
        """
        Calculate position to guard a side of lawn.

        From turn 29: "guard the W side of the lawn in view of
        the impending lift"

        Args:
            side_to_guard: "W", "E", "N", or "S"
            opponent_position: Where opponent is
            court: The court

        Returns:
            Tuple (x, y) of guarding position
        """
        target_hoop = None  # Could pass actual hoop if needed
        expert = self._get_expert_tactics(court)
        pos = expert.get_guarding_position(side_to_guard, target_hoop, opponent_position)
        return (pos.x, pos.y)

    def get_stroke_quality_modifier(
        self,
        is_rushed: bool = False,
        is_pressured: bool = False
    ) -> float:
        """
        Get modifier for stroke quality based on conditions.

        From Aiton-Maugham commentary:
        - "wild thrash" vs running hoop "smoothly"
        - Confidence degrades with consecutive misses

        Args:
            is_rushed: Is the player rushing the shot?
            is_pressured: Is there time/tactical pressure?

        Returns:
            Multiplier for success probability (0.5-1.0)
        """
        recent_misses = self.get_recent_misses()
        expert = self._get_expert_tactics(self._court) if self._court else None

        if expert:
            return expert.assess_stroke_quality_effect(
                is_rushed, is_pressured, recent_misses
            )
        else:
            # Fallback calculation
            modifier = 1.0
            if is_rushed:
                modifier *= 0.75
            if is_pressured:
                modifier *= 0.9
            if recent_misses > 0:
                modifier *= max(0.75, 1.0 - recent_misses * 0.03)
            return max(0.5, modifier)
