"""
Tactical Decision Maker - Connects learning to tactical decisions.

This module bridges the gap between:
- Learned parameters (weights, success rates, pattern recognition)
- Expert tactics (Aiton fundamentals, match play insights)
- Neural network Q-value estimation (when available)
- Actual shot selection

The key insight is that learning should INFORM tactical decisions,
not replace them. We use learned values to:
1. Adjust confidence thresholds
2. Weight different tactical options
3. Calibrate risk/reward calculations

Supports two modes:
- Hand-crafted Q-values (default, no PyTorch required)
- Neural network Q-values (requires PyTorch, better after training)
"""
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto

from models.ball import Ball, Vector2
from models.court import Court

# Try to import neural network components
try:
    from ai.neural.croquet_net import CroquetNet, StateEncoder, check_torch_available
    NEURAL_AVAILABLE = check_torch_available()
except ImportError:
    NEURAL_AVAILABLE = False


class ShotType(Enum):
    """Types of shots the AI can select."""
    ROQUET = auto()          # Hit another ball
    HOOP_RUN = auto()        # Run a hoop
    APPROACH = auto()        # Position for hoop
    RUSH = auto()            # Controlled roquet toward target
    LEAVE = auto()           # End-of-turn positioning
    DEFENSIVE = auto()       # Safety shot
    PEG_OUT = auto()         # Hit peg to finish


@dataclass
class ShotOption:
    """A potential shot with its evaluation."""
    shot_type: ShotType
    target: Vector2
    target_ball: Optional[Ball]  # For roquet/rush
    expected_value: float        # Q-value estimate
    success_probability: float
    risk_factor: float           # 0-1, how risky
    reward_if_success: float     # Expected reward on success
    description: str


class TacticalDecisionMaker:
    """
    Makes tactical decisions using learned parameters.

    Implements a Q-value estimation approach where:
    Q(shot) = P(success) * reward_success + P(fail) * reward_fail

    Learned parameters adjust:
    - Success probability estimates
    - Reward values for different outcomes
    - Risk tolerance based on game state
    """

    # Base probabilities (adjusted by learning)
    BASE_ROQUET_PROB = {
        'short': 0.90,   # < 5 yards
        'medium': 0.70,  # 5-10 yards
        'long': 0.45,    # 10-15 yards
        'very_long': 0.25  # > 15 yards
    }

    # Base hoop running success rates by approach quality
    BASE_HOOP_PROB = {
        'excellent': 0.85,  # 1 yard, good angle
        'good': 0.70,       # 2-3 yards, reasonable angle
        'fair': 0.50,       # 4-5 yards or poor angle
        'poor': 0.25        # > 5 yards or bad angle
    }

    # Reward values (adjusted by learning and game state)
    # Higher hoop run reward to prioritize scoring
    BASE_REWARDS = {
        'hoop_run': 15.0,       # High reward for scoring
        'roquet': 5.0,          # Good for building break
        'good_position': 2.0,   # Less valuable than making progress
        'safe_leave': 1.5,      # Safety is okay but not the goal
        'peg_out': 20.0,        # Winning is best
        'miss_cost': -2.0,      # Penalty for missing
        'bad_leave_cost': -4.0,
    }

    # Mapping from neural network actions to shot types
    NEURAL_ACTION_MAP = {
        0: ShotType.HOOP_RUN,
        1: ShotType.ROQUET,      # Nearest
        2: ShotType.ROQUET,      # Partner
        3: ShotType.ROQUET,      # Opponent 1
        4: ShotType.ROQUET,      # Opponent 2
        5: ShotType.APPROACH,
        6: ShotType.DEFENSIVE,
        7: ShotType.PEG_OUT,
    }

    def __init__(self, learner=None, neural_net=None, use_neural: bool = False):
        """
        Initialize with optional learner and neural network.

        Args:
            learner: CroquetLearner instance (optional)
            neural_net: Trained CroquetNet for Q-value estimation (optional)
            use_neural: Whether to use neural network for decisions
        """
        self.learner = learner
        self.neural_net = neural_net
        self.use_neural = use_neural and NEURAL_AVAILABLE and neural_net is not None

        # State encoder for neural network
        self.encoder = StateEncoder() if NEURAL_AVAILABLE else None

        # Learned adjustments (start neutral)
        self.probability_adjustments = {}
        self.reward_adjustments = {}
        self.risk_tolerance = 0.5  # 0=conservative, 1=aggressive

        # Load learned parameters if available
        if learner:
            self._load_learned_parameters()

    def set_neural_net(self, net, use_neural: bool = True):
        """Set neural network for Q-value estimation."""
        self.neural_net = net
        self.use_neural = use_neural and NEURAL_AVAILABLE and net is not None

    def _load_learned_parameters(self):
        """Load parameters from learner."""
        if not self.learner:
            return

        stats = self.learner.get_stats()

        # Adjust based on learned power calibration
        power_adj = stats.get('power_adjustment', 1.0)
        # If we've been hitting too hard/soft, affects all shots
        self.probability_adjustments['power_factor'] = power_adj

        # Load hoop-specific success rates
        if hasattr(self.learner, 'hoop_success_rates'):
            for hoop_num, rate in self.learner.hoop_success_rates.items():
                if self.learner.hoop_attempts.get(hoop_num, 0) > 5:
                    self.probability_adjustments[f'hoop_{hoop_num}'] = rate

        # Load approach pattern success (approach_patterns is dict of lists)
        if hasattr(self.learner, 'approach_patterns'):
            for hoop_num, patterns in self.learner.approach_patterns.items():
                if len(patterns) > 5:  # Only use if enough data
                    successes = sum(1 for p in patterns if p.get('success', False))
                    self.probability_adjustments[f'approach_{hoop_num}'] = successes / len(patterns)

    def evaluate_shot_options(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int = 1,
        game_state: Dict = None
    ) -> List[ShotOption]:
        """
        Evaluate all available shot options and return ranked list.

        Args:
            striker: Ball making the shot
            balls: All balls on court
            court: The court
            deadness: Which balls striker is dead on
            strokes_remaining: Strokes left in turn
            game_state: Optional game context (score, time, etc.)

        Returns:
            List of ShotOptions sorted by expected value (best first)
        """
        options = []

        dead_on = deadness.get(striker.color, set())
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)

        # Option 1: Run hoop if positioned
        if target_hoop:
            hoop_option = self._evaluate_hoop_run(striker, target_hoop, game_state)
            if hoop_option:
                options.append(hoop_option)

        # Option 2: Roquet available balls
        for color, ball in balls.items():
            if color == striker.color:
                continue
            if color in dead_on:
                continue
            if ball.has_pegged_out:
                continue

            roquet_option = self._evaluate_roquet(
                striker, ball, target_hoop, strokes_remaining, game_state
            )
            if roquet_option:
                options.append(roquet_option)

        # Option 3: Approach hoop (if not close enough to run)
        if target_hoop:
            approach_option = self._evaluate_approach(striker, target_hoop, balls, game_state)
            if approach_option:
                options.append(approach_option)

        # Option 4: Defensive/safe shot
        defensive_option = self._evaluate_defensive(striker, balls, court, game_state)
        if defensive_option:
            options.append(defensive_option)

        # Option 5: Peg out (if rover)
        if striker.is_rover:
            peg_option = self._evaluate_peg_out(striker, court, game_state)
            if peg_option:
                options.append(peg_option)

        # Sort by expected value
        options.sort(key=lambda x: x.expected_value, reverse=True)

        return options

    def _evaluate_hoop_run(
        self,
        striker: Ball,
        target_hoop,
        game_state: Dict = None
    ) -> Optional[ShotOption]:
        """Evaluate running the target hoop."""
        to_hoop = target_hoop.position - striker.position
        distance = to_hoop.magnitude()

        # Check angle to hoop
        approach_angle = abs(math.atan2(to_hoop.y, to_hoop.x) -
                           math.atan2(target_hoop.direction.y, target_hoop.direction.x))
        if approach_angle > math.pi:
            approach_angle = 2 * math.pi - approach_angle

        # Determine quality category
        # Aiton: 1 yard is ideal, angle matters but closer means more forgiving
        if distance < 1.5 and approach_angle < 0.5:  # Very close, decent angle
            quality = 'excellent'
        elif distance < 2.5 and approach_angle < 0.4:  # Close with good angle
            quality = 'excellent'
        elif distance < 3 and approach_angle < 0.6:
            quality = 'good'
        elif distance < 4 and approach_angle < 0.8:
            quality = 'good'
        elif distance < 5 and approach_angle < 1.0:
            quality = 'fair'
        elif distance < 3:  # Very close but bad angle - still attempt
            quality = 'fair'
        else:
            quality = 'poor'

        # Get base probability
        base_prob = self.BASE_HOOP_PROB[quality]

        # Apply learned adjustments
        hoop_num = target_hoop.number if hasattr(target_hoop, 'number') else striker.hoops_run + 1
        if f'hoop_{hoop_num}' in self.probability_adjustments:
            learned_rate = self.probability_adjustments[f'hoop_{hoop_num}']
            # Blend base with learned (more weight to learned with more data)
            base_prob = 0.5 * base_prob + 0.5 * learned_rate

        # Calculate expected value
        reward = self.BASE_REWARDS['hoop_run']
        miss_cost = self.BASE_REWARDS['miss_cost']

        expected_value = base_prob * reward + (1 - base_prob) * miss_cost

        # Only suggest if above threshold
        if quality in ['excellent', 'good'] or (quality == 'fair' and distance < 3):
            return ShotOption(
                shot_type=ShotType.HOOP_RUN,
                target=target_hoop.position,
                target_ball=None,
                expected_value=expected_value,
                success_probability=base_prob,
                risk_factor=1.0 - base_prob,
                reward_if_success=reward,
                description=f"Run hoop {hoop_num} ({quality} approach, {base_prob:.0%})"
            )

        return None

    def _evaluate_roquet(
        self,
        striker: Ball,
        target: Ball,
        target_hoop,
        strokes_remaining: int,
        game_state: Dict = None
    ) -> Optional[ShotOption]:
        """Evaluate roqueting a ball."""
        to_target = target.position - striker.position
        distance = to_target.magnitude()

        # Categorize distance
        if distance < 5:
            dist_cat = 'short'
        elif distance < 10:
            dist_cat = 'medium'
        elif distance < 15:
            dist_cat = 'long'
        else:
            dist_cat = 'very_long'

        base_prob = self.BASE_ROQUET_PROB[dist_cat]

        # Adjust for learned power calibration
        if 'power_factor' in self.probability_adjustments:
            pf = self.probability_adjustments['power_factor']
            # If power is off, accuracy suffers
            if abs(pf - 1.0) > 0.1:
                base_prob *= 0.9

        # Calculate reward based on what we gain
        reward = self.BASE_REWARDS['roquet']

        # Bonus if target is well-positioned relative to our hoop
        if target_hoop:
            target_to_hoop = (target_hoop.position - target.position).magnitude()
            if target_to_hoop < 8:
                reward += 2  # Ball is near our hoop - good for rush

        # Bonus for partner ball (can peel)
        partner_color = {"blue": "black", "black": "blue",
                        "red": "yellow", "yellow": "red"}.get(striker.color)
        if target.color == partner_color:
            reward += 1  # Slight bonus for partner

        miss_cost = self.BASE_REWARDS['miss_cost']
        expected_value = base_prob * reward + (1 - base_prob) * miss_cost

        return ShotOption(
            shot_type=ShotType.ROQUET,
            target=target.position,
            target_ball=target,
            expected_value=expected_value,
            success_probability=base_prob,
            risk_factor=1.0 - base_prob,
            reward_if_success=reward,
            description=f"Roquet {target.color} ({distance:.1f}yd, {base_prob:.0%})"
        )

    def _evaluate_approach(
        self,
        striker: Ball,
        target_hoop,
        balls: Dict[str, Ball],
        game_state: Dict = None
    ) -> Optional[ShotOption]:
        """Evaluate approaching the hoop."""
        to_hoop = target_hoop.position - striker.position
        distance = to_hoop.magnitude()

        # Approach is positioning - usually high success, moderate reward
        # Ideal approach position: 1 yard from hoop (Aiton)
        ideal_pos = target_hoop.position - target_hoop.direction * 1.0

        success_prob = 0.85  # Usually can position well
        reward = self.BASE_REWARDS['good_position']

        # Reward higher if we're far and need to get closer
        if distance > 5:
            reward += 1.5

        expected_value = success_prob * reward

        return ShotOption(
            shot_type=ShotType.APPROACH,
            target=ideal_pos,
            target_ball=None,
            expected_value=expected_value,
            success_probability=success_prob,
            risk_factor=0.15,
            reward_if_success=reward,
            description=f"Approach hoop ({distance:.1f}yd away)"
        )

    def _evaluate_defensive(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        game_state: Dict = None
    ) -> Optional[ShotOption]:
        """Evaluate a defensive/safe shot."""
        # Find a safe corner or boundary position
        # Away from opponents, maybe near partner

        corners = [
            Vector2(2, 2),       # Corner I
            Vector2(26, 2),      # Corner II
            Vector2(26, 33),     # Corner III
            Vector2(2, 33),      # Corner IV
        ]

        # Find corner furthest from opponents
        opponent_colors = []
        for color in balls:
            if color != striker.color:
                team = {"blue": "black", "black": "blue", "red": "yellow", "yellow": "red"}
                if team.get(color) != striker.color:
                    opponent_colors.append(color)

        best_corner = corners[0]
        best_dist = 0

        for corner in corners:
            min_opp_dist = float('inf')
            for opp_color in opponent_colors:
                if opp_color in balls:
                    opp_dist = (corner - balls[opp_color].position).magnitude()
                    min_opp_dist = min(min_opp_dist, opp_dist)
            if min_opp_dist > best_dist:
                best_dist = min_opp_dist
                best_corner = corner

        success_prob = 0.90  # Can usually get to safe position
        reward = self.BASE_REWARDS['safe_leave']

        expected_value = success_prob * reward

        return ShotOption(
            shot_type=ShotType.DEFENSIVE,
            target=best_corner,
            target_ball=None,
            expected_value=expected_value,
            success_probability=success_prob,
            risk_factor=0.1,
            reward_if_success=reward,
            description=f"Defensive to corner ({best_dist:.1f}yd from opponents)"
        )

    def _evaluate_peg_out(
        self,
        striker: Ball,
        court: Court,
        game_state: Dict = None
    ) -> Optional[ShotOption]:
        """Evaluate pegging out."""
        if not striker.is_rover:
            return None

        to_peg = court.peg_position - striker.position
        distance = to_peg.magnitude()

        # Similar to roquet probability
        if distance < 3:
            success_prob = 0.85
        elif distance < 6:
            success_prob = 0.70
        elif distance < 10:
            success_prob = 0.50
        else:
            success_prob = 0.30

        reward = self.BASE_REWARDS['peg_out']
        miss_cost = self.BASE_REWARDS['miss_cost']

        expected_value = success_prob * reward + (1 - success_prob) * miss_cost

        return ShotOption(
            shot_type=ShotType.PEG_OUT,
            target=court.peg_position,
            target_ball=None,
            expected_value=expected_value,
            success_probability=success_prob,
            risk_factor=1.0 - success_prob,
            reward_if_success=reward,
            description=f"Peg out ({distance:.1f}yd, {success_prob:.0%})"
        )

    def select_best_shot(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int = 1,
        game_state: Dict = None,
        use_expert_tactics: bool = True
    ) -> ShotOption:
        """
        Select the best shot considering all factors.

        Args:
            striker: Ball making the shot
            balls: All balls
            court: The court
            deadness: Deadness info
            strokes_remaining: Strokes left
            game_state: Game context
            use_expert_tactics: Whether to apply expert tactical adjustments

        Returns:
            Best ShotOption
        """
        # Use neural network if available and enabled
        if self.use_neural and self.neural_net is not None:
            return self._select_neural_shot(
                striker, balls, court, deadness, strokes_remaining,
                game_state, use_expert_tactics
            )

        options = self.evaluate_shot_options(
            striker, balls, court, deadness, strokes_remaining, game_state
        )

        if not options:
            # Fallback: defensive shot
            return ShotOption(
                shot_type=ShotType.DEFENSIVE,
                target=Vector2(court.width / 2, court.height / 2),
                target_ball=None,
                expected_value=0,
                success_probability=0.5,
                risk_factor=0.5,
                reward_if_success=1,
                description="No good options - defensive"
            )

        if use_expert_tactics:
            options = self._apply_expert_adjustments(
                options, striker, balls, court, deadness, strokes_remaining
            )

        # Apply risk tolerance
        if self.risk_tolerance < 0.3:
            # Conservative: penalize risky shots
            for opt in options:
                opt.expected_value *= (1 - opt.risk_factor * 0.3)
        elif self.risk_tolerance > 0.7:
            # Aggressive: bonus for high-reward shots
            for opt in options:
                if opt.reward_if_success > 5:
                    opt.expected_value *= 1.2

        # Re-sort after adjustments
        options.sort(key=lambda x: x.expected_value, reverse=True)

        return options[0]

    def _apply_expert_adjustments(
        self,
        options: List[ShotOption],
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int
    ) -> List[ShotOption]:
        """
        Apply expert tactical adjustments from Aiton/Maugham insights.
        """
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)

        for opt in options:
            # Aiton: "What do you do after you've run this?"
            # Penalize shots that leave poor position
            if opt.shot_type == ShotType.HOOP_RUN:
                # Check what position we'd be in after running
                if target_hoop:
                    next_hoop = court.get_hoop_for_ball(striker.hoops_run + 1)
                    if next_hoop:
                        # Will we have a ball to rush?
                        has_rush_available = False
                        dead_after = deadness.get(striker.color, set())

                        for color, ball in balls.items():
                            if color == striker.color:
                                continue
                            if color in dead_after:
                                continue
                            dist = (ball.position - target_hoop.position).magnitude()
                            if dist < 10:
                                has_rush_available = True
                                break

                        if not has_rush_available:
                            # No rush after hoop - less valuable
                            opt.expected_value *= 0.85
                            opt.description += " (no rush after)"

            # Aiton: Position value for wired positions
            if opt.shot_type == ShotType.DEFENSIVE:
                # Check if we'd be wired from opponents
                # (simplified check)
                opt.expected_value *= 1.1  # Slight boost for defensive awareness

            # Maugham: Consider impasse situations
            if opt.shot_type == ShotType.ROQUET:
                # If we're in an impasse near a hoop, roquet might not be best
                if target_hoop:
                    dist_to_hoop = (striker.position - target_hoop.position).magnitude()
                    if dist_to_hoop < 10:
                        # Near our hoop - be more careful
                        if opt.success_probability < 0.6:
                            opt.expected_value *= 0.9
                            opt.description += " (impasse caution)"

        return options

    def update_from_outcome(
        self,
        shot_taken: ShotOption,
        success: bool,
        actual_reward: float
    ):
        """
        Update learned parameters based on shot outcome.

        This is simplified temporal-difference learning:
        Q(s,a) <- Q(s,a) + alpha * (reward - Q(s,a))

        Args:
            shot_taken: The shot that was taken
            success: Whether it succeeded
            actual_reward: The actual reward received
        """
        if not self.learner:
            return

        # Update appropriate success rate based on shot type
        if shot_taken.shot_type == ShotType.HOOP_RUN:
            # Could update hoop-specific rates
            pass
        elif shot_taken.shot_type == ShotType.ROQUET:
            # Could update distance-based roquet rates
            pass

        # The learner handles most updates through its own interface
        # This method is for direct tactical learning

    def _select_neural_shot(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int,
        game_state: Dict,
        use_expert_tactics: bool
    ) -> ShotOption:
        """
        Select shot using neural network Q-values.

        The neural network outputs Q-values for action types:
        0: HOOP_RUN, 1-4: ROQUET (various targets), 5: APPROACH, 6: DEFENSIVE, 7: PEG_OUT

        We then use hand-crafted logic to determine the specific target
        based on the selected action type.
        """
        import torch

        # Encode state
        is_cont = game_state.get('is_continuation', False) if game_state else False
        state = self.encoder.encode(
            striker, balls, court, deadness, strokes_remaining, is_cont
        )

        # Get valid actions based on game state
        valid_actions = self._get_valid_neural_actions(striker, balls, court, deadness)

        # Get neural network prediction
        self.neural_net.eval()
        with torch.no_grad():
            action, q_value = self.neural_net.get_action(
                state, epsilon=0.0, valid_actions=valid_actions
            )

        # Convert neural action to shot option
        shot_option = self._neural_action_to_shot(
            action, q_value, striker, balls, court, deadness, game_state
        )

        # Apply expert adjustments if enabled
        if use_expert_tactics and shot_option:
            options = [shot_option]
            options = self._apply_expert_adjustments(
                options, striker, balls, court, deadness, strokes_remaining
            )
            shot_option = options[0]

        return shot_option

    def _get_valid_neural_actions(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set]
    ) -> List[int]:
        """Get list of valid neural network actions for current state."""
        valid = []
        dead_on = deadness.get(striker.color, set())

        # Action 0: HOOP_RUN - only valid if ball is on correct approach side
        if not striker.is_rover:
            target_hoop = court.get_hoop_for_ball(striker.hoops_run)
            if target_hoop:
                # Check if ball is on approach side (can run through in correct direction)
                ball_to_hoop = target_hoop.position - striker.position
                if ball_to_hoop.magnitude() > 0.1:
                    # Dot product: positive means ball is on approach side
                    approach_dot = ball_to_hoop.normalize().dot(target_hoop.direction)
                    # Also check we're close enough to reasonably attempt
                    dist_to_hoop = ball_to_hoop.magnitude()
                    if approach_dot > 0.2 and dist_to_hoop < 10:  # On approach side and within range
                        valid.append(0)

        # Actions 1-4: ROQUET variants
        # Check if we have live balls to roquet
        partner_color = {"blue": "black", "black": "blue",
                        "red": "yellow", "yellow": "red"}.get(striker.color)

        for color, ball in balls.items():
            if color == striker.color:
                continue
            if color in dead_on:
                continue
            if ball.has_pegged_out:
                continue

            # At least one roquet is valid
            valid.append(1)  # Nearest
            if color == partner_color:
                valid.append(2)  # Partner
            else:
                if 3 not in valid:
                    valid.append(3)  # Opponent 1
                elif 4 not in valid:
                    valid.append(4)  # Opponent 2
            break  # Just need to confirm at least one valid

        # Action 5: APPROACH - always valid
        valid.append(5)

        # Action 6: DEFENSIVE - always valid
        valid.append(6)

        # Action 7: PEG_OUT - only if rover
        if striker.is_rover:
            valid.append(7)

        return list(set(valid))  # Remove duplicates

    def _neural_action_to_shot(
        self,
        action: int,
        q_value: float,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        game_state: Dict
    ) -> ShotOption:
        """Convert neural network action to concrete ShotOption."""
        shot_type = self.NEURAL_ACTION_MAP.get(action, ShotType.DEFENSIVE)
        dead_on = deadness.get(striker.color, set())
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)

        if shot_type == ShotType.HOOP_RUN:
            if target_hoop:
                # Calculate target point BEYOND the hoop in the required direction
                # so ball travels through the hoop correctly
                # Target is 2 yards past the hoop center in the run direction
                run_target = target_hoop.position + target_hoop.direction * 2.0
                return ShotOption(
                    shot_type=ShotType.HOOP_RUN,
                    target=run_target,
                    target_ball=None,
                    expected_value=q_value,
                    success_probability=0.5,  # Neural net handles this
                    risk_factor=0.3,
                    reward_if_success=15.0,
                    description=f"[NN] Run hoop {striker.hoops_run + 1} (Q={q_value:.2f})"
                )

        elif shot_type == ShotType.ROQUET:
            # Find best roquet target based on action variant
            partner_color = {"blue": "black", "black": "blue",
                           "red": "yellow", "yellow": "red"}.get(striker.color)

            target_ball = None
            if action == 2 and partner_color in balls:  # Partner
                target_ball = balls[partner_color]
            elif action in [3, 4]:  # Opponent
                for color, ball in balls.items():
                    if color != striker.color and color != partner_color:
                        if color not in dead_on and not ball.has_pegged_out:
                            target_ball = ball
                            break
            else:  # Nearest (action 1)
                min_dist = float('inf')
                for color, ball in balls.items():
                    if color == striker.color:
                        continue
                    if color in dead_on or ball.has_pegged_out:
                        continue
                    dist = (ball.position - striker.position).magnitude()
                    if dist < min_dist:
                        min_dist = dist
                        target_ball = ball

            if target_ball:
                dist = (target_ball.position - striker.position).magnitude()
                return ShotOption(
                    shot_type=ShotType.ROQUET,
                    target=target_ball.position,
                    target_ball=target_ball,
                    expected_value=q_value,
                    success_probability=0.7,
                    risk_factor=0.2,
                    reward_if_success=5.0,
                    description=f"[NN] Roquet {target_ball.color} ({dist:.1f}yd, Q={q_value:.2f})"
                )

        elif shot_type == ShotType.APPROACH:
            if target_hoop:
                ideal_pos = target_hoop.position - target_hoop.direction * 1.0
                return ShotOption(
                    shot_type=ShotType.APPROACH,
                    target=ideal_pos,
                    target_ball=None,
                    expected_value=q_value,
                    success_probability=0.85,
                    risk_factor=0.15,
                    reward_if_success=2.0,
                    description=f"[NN] Approach hoop (Q={q_value:.2f})"
                )

        elif shot_type == ShotType.PEG_OUT:
            return ShotOption(
                shot_type=ShotType.PEG_OUT,
                target=court.peg_position,
                target_ball=None,
                expected_value=q_value,
                success_probability=0.5,
                risk_factor=0.4,
                reward_if_success=20.0,
                description=f"[NN] Peg out (Q={q_value:.2f})"
            )

        # Default: Defensive - go to boundary near next hoop (not center!)
        # Being at boundaries is defensive, being in center is BAD
        if target_hoop:
            # Go to approach side of next hoop, near boundary
            approach_dir = target_hoop.direction * -1  # Opposite of run direction
            defensive_target = target_hoop.position + approach_dir * 3  # 3 yards back from hoop
        else:
            # Fallback to a boundary position
            defensive_target = Vector2(1, 1)  # Near south-west corner
        return ShotOption(
            shot_type=ShotType.DEFENSIVE,
            target=defensive_target,
            target_ball=None,
            expected_value=q_value,
            success_probability=0.9,
            risk_factor=0.1,
            reward_if_success=1.5,
            description=f"[NN] Defensive (Q={q_value:.2f})"
        )

    def encode_state(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int = 1,
        is_continuation: bool = False
    ):
        """
        Encode game state for neural network.

        Returns tensor if encoder available, None otherwise.
        """
        if self.encoder is None:
            return None
        return self.encoder.encode(
            striker, balls, court, deadness, strokes_remaining, is_continuation
        )
