"""
Learning Strategy - AI strategy that uses the hybrid learning system.

Combines tactical rules, position evaluation, and shot simulation
to make informed decisions that improve over time.

BREAK-BUILDING PHILOSOPHY (from Wylie's Expert Croquet Tactics):
The goal in Association Croquet is to run all 12 hoops in one turn.
This requires building and maintaining a 4-ball break:

4-Ball Break Pattern:
1. ROQUET a ball near you (the pilot/reception ball)
2. CROQUET: Send pilot as PIONEER to next-but-one hoop, position for RUSH on another ball
3. RUSH that ball toward your target hoop
4. CROQUET: Send that ball as new pioneer, position in front of hoop
5. RUN THE HOOP with control (exit toward next ball)
6. RUSH to next ball, repeat from step 3

Key ball roles:
- PIONEER: Ball pre-positioned at next hoop (or hoop after next)
- PILOT/RECEPTION: Ball you roquet to start a break
- PIVOT: Ball kept near center for flexibility
- ESCAPE BALL: Ball used after running hoop to get to pioneer
"""
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum, auto

import config
from models.ball import Ball, Vector2
from models.court import Court, Hoop
from ai.learning.position_evaluator import PositionEvaluator
from ai.learning.shot_simulator import ShotSimulator
from ai.learning.tactical_rules import TacticalRules, ShotType
from ai.learning.leave_patterns import get_pattern_library, LeaveType


class BreakPhase(Enum):
    """Current phase in the break-building cycle."""
    NO_BREAK = auto()          # Not in a break (need to roquet to start)
    JUST_ROQUETED = auto()     # Just roqueted, need to take croquet stroke
    CROQUET_TAKEN = auto()     # Croquet stroke done, have continuation
    APPROACHING_HOOP = auto()  # Positioned for hoop run
    HOOP_RUN = auto()          # Just ran hoop, continuation stroke
    SEEKING_RUSH = auto()      # Looking for ball to rush


@dataclass
class BreakState:
    """Tracks the current state of a break in progress."""
    phase: BreakPhase = BreakPhase.NO_BREAK
    hoops_in_break: int = 0

    # Ball roles (colors)
    pilot_ball: Optional[str] = None      # Ball at/near current hoop
    pioneer_ball: Optional[str] = None    # Ball at next hoop
    pivot_ball: Optional[str] = None      # Ball near center
    escape_ball: Optional[str] = None     # Ball to rush after running hoop

    # Targets for this turn
    current_target: Optional[Vector2] = None
    next_pioneer_target: Optional[Vector2] = None


class LearningStrategy:
    """
    AI strategy that learns from experience.

    Uses:
    - Tactical rules for high-level decisions
    - Position evaluation for assessing board states
    - Shot simulation for evaluating shot options
    - Learned power adjustment from experience
    - BREAK-BUILDING: Strategic 4-ball break planning
    """

    def __init__(self, skill_level: float = 0.8):
        """
        Initialize learning strategy.

        Args:
            skill_level: Base skill level (affects shot accuracy)
        """
        self.skill_level = skill_level
        self.name = "Learning"

        # Learning components
        self.position_evaluator = PositionEvaluator()
        self.shot_simulator = ShotSimulator(num_simulations=20)
        self.tactical_rules = TacticalRules()

        # Leave pattern library for deliberate leave setup
        self.leave_patterns = get_pattern_library()

        # Break state tracking - THE KEY TO PROPER CROQUET TACTICS
        self.break_state = BreakState()

        # Track decisions for learning
        self.last_advice = None
        self.last_shot_type = None

        # Load learned parameters from saved state
        self.power_adjustment = self._load_power_adjustment()

    def _load_power_adjustment(self) -> float:
        """Load the learned power adjustment from saved learning state."""
        state_file = Path("ai_data/learning_state.json")
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    return state.get("power_adjustment", 1.0)
            except Exception:
                pass
        return 1.0

    def reset_break(self):
        """Reset break state (called at start of new turn)."""
        self.break_state = BreakState()

    def _analyze_break_potential(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set]
    ) -> None:
        """
        Analyze the current position and assign ball roles for break building.

        This is the heart of croquet strategy - identifying which balls can
        serve as pioneers, pivots, and pilots for the break.
        """
        dead_on = deadness.get(ball.color, set())

        # Get available (live) balls
        available = []
        for color, b in balls.items():
            if color != ball.color and color not in dead_on and not b.has_pegged_out:
                available.append((color, b))

        if not available:
            return

        target_hoop = court.get_hoop_for_ball(ball.hoops_run)
        next_hoop = court.get_hoop_for_ball(ball.hoops_run + 1)

        if not target_hoop:
            return

        # Score each ball for each role
        pilot_scores = {}
        pioneer_scores = {}
        pivot_scores = {}

        for color, b in available:
            # PILOT score: good for getting to current hoop
            # Ideal: 2-6 yards from hoop, in front, reachable
            to_hoop = target_hoop.position - b.position
            dist_to_hoop = to_hoop.magnitude()

            if dist_to_hoop > 0.5:
                approach_dot = to_hoop.normalize().dot(target_hoop.direction)
            else:
                approach_dot = 1.0

            # Pilot should be close to hoop, on approach side
            if 1 < dist_to_hoop < 8 and approach_dot > 0:
                pilot_score = (1 - dist_to_hoop / 10) * (0.5 + approach_dot * 0.5)
            else:
                pilot_score = 0.1

            # Bonus if ball is between striker and hoop (good rush alignment)
            striker_to_hoop = target_hoop.position - ball.position
            striker_to_ball = b.position - ball.position
            if striker_to_ball.magnitude() > 0.5 and striker_to_hoop.magnitude() > 0.5:
                rush_alignment = striker_to_ball.normalize().dot(striker_to_hoop.normalize())
                if rush_alignment > 0.3:
                    pilot_score += 0.3

            pilot_scores[color] = pilot_score

            # PIONEER score: good for next hoop
            if next_hoop:
                to_next = next_hoop.position - b.position
                dist_to_next = to_next.magnitude()

                if dist_to_next > 0.5:
                    next_approach = to_next.normalize().dot(next_hoop.direction)
                else:
                    next_approach = 1.0

                # Pioneer should be 3-6 yards from next hoop, in front
                if 2 < dist_to_next < 8 and next_approach > 0:
                    pioneer_score = (1 - dist_to_next / 10) * (0.5 + next_approach * 0.5)
                else:
                    pioneer_score = 0.1
            else:
                pioneer_score = 0.1

            pioneer_scores[color] = pioneer_score

            # PIVOT score: near center, flexible position
            center = Vector2(court.width / 2, court.height / 2)
            dist_to_center = (b.position - center).magnitude()
            pivot_scores[color] = max(0.1, 1 - dist_to_center / 15)

        # Assign roles based on scores
        assigned = set()

        # Best pilot first (most important for immediate play)
        if pilot_scores:
            best_pilot = max(pilot_scores, key=pilot_scores.get)
            if pilot_scores[best_pilot] > 0.25:
                self.break_state.pilot_ball = best_pilot
                assigned.add(best_pilot)

        # Best pioneer from remaining
        remaining_pioneer = {c: s for c, s in pioneer_scores.items() if c not in assigned}
        if remaining_pioneer:
            best_pioneer = max(remaining_pioneer, key=remaining_pioneer.get)
            if remaining_pioneer[best_pioneer] > 0.2:
                self.break_state.pioneer_ball = best_pioneer
                assigned.add(best_pioneer)

        # Best pivot from remaining
        remaining_pivot = {c: s for c, s in pivot_scores.items() if c not in assigned}
        if remaining_pivot:
            best_pivot = max(remaining_pivot, key=remaining_pivot.get)
            self.break_state.pivot_ball = best_pivot

        # Set targets
        if next_hoop:
            self.break_state.next_pioneer_target = next_hoop.position - next_hoop.direction * 4

    def _get_break_aware_shot(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int,
        is_continuation: bool
    ) -> Optional[Tuple[float, float, str]]:
        """
        Get a shot recommendation based on break-building strategy.

        This implements the core 4-ball break logic:
        - If in position for hoop: RUN IT
        - If not in position: ROQUET to build/continue break
        - After roquet: Position for rush to hoop (croquet stroke handles this)
        - After running hoop: Rush to next ball to continue break

        WIRING ENFORCEMENT: Balls that are wired (blocked by hoops/peg) are
        not valid roquet targets. We must check for wiring before selecting.

        Returns:
            Tuple of (angle, power, reason) or None if no break-aware shot
        """
        dead_on = deadness.get(ball.color, set())
        target_hoop = court.get_hoop_for_ball(ball.hoops_run)

        if not target_hoop:
            return None

        # Check if in position to run hoop
        to_hoop = target_hoop.position - ball.position
        dist_to_hoop = to_hoop.magnitude()

        if dist_to_hoop > 0.5:
            approach_dir = to_hoop.normalize()
            approach_quality = approach_dir.dot(target_hoop.direction)
        else:
            approach_quality = 1.0

        # PRIORITY 1: If in EXCELLENT hoop position, RUN THE HOOP
        # TIGHTENED: Must be within 3 yards (not 5) with very good approach angle (0.7 not 0.5)
        # This prevents the AI from "setting up in front of hoops" when it should be building breaks
        # Only run hoop if we're actually in runnable position, not just nearby
        if dist_to_hoop < 3 and approach_quality > 0.7:
            # Excellent position - run the hoop!
            aim_point = target_hoop.position + target_hoop.direction * 2
            to_aim = aim_point - ball.position
            angle = math.atan2(to_aim.y, to_aim.x)
            power = self._calculate_power_for_distance(to_aim.magnitude() + 2)
            return (angle, power, f"RUN HOOP {ball.hoops_run + 1} (excellent position)")

        # PRIORITY 2: If NOT in hoop position, need to ROQUET to continue break
        # Find the best ball to roquet based on break roles
        # WIRING CHECK: Skip balls we're wired from (can't legally hit them)
        best_roquet = None
        best_roquet_score = 0

        for color, other_ball in balls.items():
            if color == ball.color or color in dead_on or other_ball.has_pegged_out:
                continue

            # WIRING ENFORCEMENT: Check if we're wired from this ball
            is_wired, obstruction = court.is_wired(
                ball.position, other_ball.position, ball.radius
            )
            if is_wired:
                # Can't legally roquet this ball - skip it
                continue

            to_ball = other_ball.position - ball.position
            dist = to_ball.magnitude()

            if dist > 20:
                continue  # Too far

            # Base score on distance (closer = better)
            score = max(0, 1 - dist / 20)

            # HUGE bonus for pilot ball - this is THE ball to roquet for break
            if color == self.break_state.pilot_ball:
                score += 1.0

            # Bonus for balls that give good rush to hoop
            ball_to_hoop = target_hoop.position - other_ball.position
            if ball_to_hoop.magnitude() > 0.5 and to_ball.magnitude() > 0.5:
                # Check if we can rush this ball toward hoop
                rush_alignment = to_ball.normalize().dot(ball_to_hoop.normalize())
                if rush_alignment > 0.3:
                    score += 0.5 * rush_alignment

            # Bonus for pivot ball (reliable center position)
            if color == self.break_state.pivot_ball:
                score += 0.3

            if score > best_roquet_score:
                best_roquet_score = score
                best_roquet = (color, other_ball, dist)

        # LOWERED threshold from 0.3 to 0.15 - we WANT to roquet to build breaks
        # The key insight: roquet is almost always better than approaching hoop directly
        # because it earns extra strokes and allows croquet positioning
        if best_roquet and best_roquet_score > 0.15:
            color, other_ball, dist = best_roquet
            to_ball = other_ball.position - ball.position
            angle = math.atan2(to_ball.y, to_ball.x)
            # Hit ball with follow-through
            power = self._calculate_power_for_distance(dist + 1)

            role = ""
            if color == self.break_state.pilot_ball:
                role = " (pilot - for rush to hoop)"
            elif color == self.break_state.pioneer_ball:
                role = " (pioneer)"
            elif color == self.break_state.pivot_ball:
                role = " (pivot)"

            return (angle, power, f"ROQUET {color}{role} to build break")

        # Only return None if truly no roquet available - this triggers hoop approach fallback
        return None

    def select_shot(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        court: Court,
        target_hoop_num: int = None,
        deadness: Dict[str, set] = None,
        strokes_remaining: int = 1,
        is_continuation: bool = False
    ) -> Tuple[float, float]:
        """
        Select a shot using tactical analysis and simulation.

        BREAK-BUILDING STRATEGY (from Wylie):
        1. First, analyze the position and assign ball roles (pilot, pioneer, pivot)
        2. If in position to run hoop: RUN IT (highest priority)
        3. If not in position: ROQUET the pilot ball to build/continue break
        4. Croquet strokes position both balls strategically
        5. After running hoop, rush to next ball to continue break

        Args:
            ball: The ball to shoot
            balls: All balls on the court
            court: The court
            target_hoop_num: Ignored (we use ball's next hoop)
            deadness: Which balls we're dead on
            strokes_remaining: Number of strokes remaining this turn
            is_continuation: Whether this is a continuation stroke

        Returns:
            Tuple of (angle in radians, power in yards/second)
        """
        if deadness is None:
            deadness = {c: set() for c in ["blue", "black", "red", "yellow"]}

        # Store deadness for use in outcome evaluation
        self._current_deadness = deadness
        self._current_striker_color = ball.color

        # CHECK FOR LEAVE SETUP - When break is ending, position for good leave
        if self._should_set_leave(ball, balls, court, deadness, strokes_remaining, is_continuation):
            leave_shot = self._get_leave_setup_shot(ball, balls, court, deadness)
            if leave_shot:
                angle, power, reason = leave_shot
                print(f"  [LEAVE] {ball.color}: {reason}")

                # Apply learned power adjustment
                adjusted_power = power * self.power_adjustment

                # Add skill-based inaccuracy
                angle_error = random.gauss(0, (1 - self.skill_level) * 0.15)
                power_error = random.gauss(0, (1 - self.skill_level) * 0.08)

                final_angle = angle + angle_error
                final_power = adjusted_power * (1 + power_error)
                final_power = max(1.5, min(config.MAX_SHOT_POWER, final_power))

                self.last_shot_type = ShotType.DEFENSIVE
                return (final_angle, final_power)

        # BREAK-BUILDING: Analyze position and assign ball roles
        self._analyze_break_potential(ball, balls, court, deadness)

        # Try break-aware shot selection first
        # This implements proper croquet tactics: roquet -> croquet -> rush -> hoop
        break_shot = self._get_break_aware_shot(
            ball, balls, court, deadness, strokes_remaining, is_continuation
        )

        if break_shot:
            angle, power, reason = break_shot

            # Store for learning
            self.last_shot_type = ShotType.ROQUET if "ROQUET" in reason else ShotType.HOOP_RUN

            # Apply learned power adjustment
            adjusted_power = power * self.power_adjustment

            # Add skill-based inaccuracy
            angle_error = random.gauss(0, (1 - self.skill_level) * 0.15)
            power_error = random.gauss(0, (1 - self.skill_level) * 0.08)

            final_angle = angle + angle_error
            final_power = adjusted_power * (1 + power_error)
            final_power = max(1.5, min(config.MAX_SHOT_POWER, final_power))

            # Print break strategy for visibility
            roles = []
            if self.break_state.pilot_ball:
                roles.append(f"pilot={self.break_state.pilot_ball}")
            if self.break_state.pioneer_ball:
                roles.append(f"pioneer={self.break_state.pioneer_ball}")
            if self.break_state.pivot_ball:
                roles.append(f"pivot={self.break_state.pivot_ball}")
            if roles:
                print(f"  [BREAK] {ball.color}: {reason} | Roles: {', '.join(roles)}")

            return (final_angle, final_power)

        # Fallback to tactical rules if break-aware shot not available
        advice_list = self.tactical_rules.get_advice(
            ball, balls, court, deadness,
            strokes_remaining=strokes_remaining,
            is_continuation=is_continuation
        )

        if not advice_list:
            # Fallback to basic shot
            return self._fallback_shot(ball, court)

        # Evaluate top advice options with simulation
        # Use composite scores for tie-breaking
        best_angle = 0
        best_power = 5.0
        best_composite = None  # (primary_value, hoop_score, roquet_score, position_score)

        for advice in advice_list[:3]:
            angle, power = self._advice_to_shot(ball, advice, court)

            # Simulate the shot
            outcomes = self.shot_simulator.simulate_shot(
                ball, angle, power, self.skill_level, balls, court
            )

            # Calculate composite scores for tie-breaking
            composite = self._evaluate_outcomes_composite(outcomes, ball, court)

            # Weight by tactical priority
            primary_value = composite[0] + advice.priority * 10
            composite = (primary_value,) + composite[1:]

            # Compare using tuple ordering for automatic tie-breaking
            # Ties on primary value break by hoop score, then roquet score, then position
            if best_composite is None or composite > best_composite:
                best_composite = composite
                best_angle = angle
                best_power = power
                self.last_advice = advice
                self.last_shot_type = advice.recommended_shot

        # Apply learned power adjustment (calibrated from experience)
        adjusted_power = best_power * self.power_adjustment

        # Add skill-based inaccuracy
        angle_error = random.gauss(0, (1 - self.skill_level) * 0.2)
        power_error = random.gauss(0, (1 - self.skill_level) * 0.1)

        final_angle = best_angle + angle_error
        final_power = adjusted_power * (1 + power_error)
        final_power = max(1.5, min(config.MAX_SHOT_POWER, final_power))

        return (final_angle, final_power)

    def select_shot_for_hoop(
        self,
        ball: Ball,
        target_hoop: Hoop,
        balls: Dict[str, Ball],
        court: Court
    ) -> Tuple[float, float]:
        """Select a shot targeting a specific hoop."""
        # Check approach quality
        if self._is_good_approach(ball, target_hoop):
            return self._shoot_for_hoop(ball, target_hoop)
        else:
            return self._shoot_for_position(ball, target_hoop, court)

    def _advice_to_shot(
        self,
        ball: Ball,
        advice,
        court: Court
    ) -> Tuple[float, float]:
        """Convert tactical advice to shot parameters."""
        if advice.target_position:
            to_target = advice.target_position - ball.position
            angle = math.atan2(to_target.y, to_target.x)
            distance = to_target.magnitude()

            # Adjust power based on shot type
            if advice.recommended_shot == ShotType.HOOP_RUN:
                # Need to go through the hoop
                power = self._calculate_power_for_distance(distance + 3)
            elif advice.recommended_shot == ShotType.HOOP_APPROACH:
                # Want to stop at position
                power = self._calculate_power_for_distance(distance)
            elif advice.recommended_shot in [ShotType.ROQUET, ShotType.RUSH]:
                # Hit the ball with some follow-through
                power = self._calculate_power_for_distance(distance + 1)
            else:
                power = self._calculate_power_for_distance(distance + 2)

            return (angle, power)

        return (0, 5.0)

    def _evaluate_outcomes_composite(
        self,
        outcomes,
        ball: Ball,
        court: Court
    ) -> Tuple[float, float, float, float, float]:
        """
        Evaluate simulated shot outcomes with composite scoring for tie-breaking.

        Returns a tuple of scores for tie-breaking:
        (primary_value, hoop_score, roquet_score, position_score, boundary_penalty)

        Using a tuple allows Python's natural comparison to break ties:
        - First compare primary value
        - If tied, prefer higher hoop score (running hoops)
        - If still tied, prefer higher roquet score (gaining strokes)
        - If still tied, prefer better position
        - If still tied, prefer avoiding boundaries
        """
        total_primary = 0.0
        total_hoop = 0.0
        total_roquet = 0.0
        total_position = 0.0
        total_boundary = 0.0

        # Get which balls we're dead on
        dead_on = set()
        if hasattr(self, '_current_deadness') and hasattr(self, '_current_striker_color'):
            dead_on = self._current_deadness.get(self._current_striker_color, set())

        for outcome in outcomes:
            primary = 0.0
            hoop = 0.0
            roquet = 0.0
            position = 0.0
            boundary = 0.0

            # Immediate game rewards - hoops are the goal!
            if outcome.hoop_run:
                primary += 50.0  # Running hoops is VERY valuable - this is the objective!
                hoop += 100.0   # Maximum hoop score for tie-breaking

            # Roquets only count if the ball is LIVE (not dead on)
            if outcome.roqueted_ball:
                if outcome.roqueted_ball not in dead_on:
                    primary += 15.0  # Roquets earn extra strokes - but only if live!
                    roquet += 50.0   # Roquet score for tie-breaking
                else:
                    # PENALTY for hitting a dead ball - this wastes the shot!
                    primary -= 20.0  # Strong penalty - hitting dead ball ends turn with no benefit
                    roquet -= 100.0  # Heavily penalize in tie-breaking too

            # Position quality after the shot
            target_hoop = court.get_hoop_for_ball(ball.hoops_run)
            if target_hoop:
                dist = (outcome.striker_position - target_hoop.position).magnitude()

                # Prefer being in front of hoop with good approach angle
                to_hoop = target_hoop.position - outcome.striker_position
                if to_hoop.magnitude() > 0.5:
                    approach_dot = to_hoop.normalize().dot(target_hoop.direction)
                    if approach_dot > 0.5:
                        # Good approach - closer is better
                        pos_value = max(0, (8 - dist)) * 1.5  # Up to 12 points for being close
                        primary += pos_value
                        # Position score: weighted by approach angle quality
                        position += (10 - dist) * approach_dot * 5  # Refined position score
                    elif approach_dot > 0:
                        # OK approach
                        pos_value = max(0, (6 - dist)) * 0.5
                        primary += pos_value
                        position += (8 - dist) * approach_dot * 2
                    else:
                        # Wrong side of hoop - penalize
                        primary -= 5.0
                        position -= dist * 2  # Distance penalty when on wrong side

            # Avoid boundaries
            if outcome.hit_boundary:
                primary -= 2.0
                boundary -= 10.0  # Boundary penalty for tie-breaking

            total_primary += primary
            total_hoop += hoop
            total_roquet += roquet
            total_position += position
            total_boundary += boundary

        n = len(outcomes) if outcomes else 1
        return (
            total_primary / n,
            total_hoop / n,
            total_roquet / n,
            total_position / n,
            total_boundary / n
        )

    def _evaluate_outcomes(
        self,
        outcomes,
        ball: Ball,
        court: Court
    ) -> float:
        """Evaluate simulated shot outcomes (legacy single-value version)."""
        composite = self._evaluate_outcomes_composite(outcomes, ball, court)
        return composite[0]  # Return just the primary value

    def _fallback_shot(self, ball: Ball, court: Court) -> Tuple[float, float]:
        """Fallback shot when no good options."""
        target_hoop = court.get_hoop_for_ball(ball.hoops_run)

        if target_hoop:
            to_hoop = target_hoop.position - ball.position
            angle = math.atan2(to_hoop.y, to_hoop.x)
            distance = to_hoop.magnitude()
            power = self._calculate_power_for_distance(distance + 2)
        elif ball.is_rover:
            # Rover: aim for the peg to finish the game!
            to_peg = court.peg_position - ball.position
            angle = math.atan2(to_peg.y, to_peg.x)
            distance = to_peg.magnitude()
            # Need to hit the peg with enough power to reach it
            power = self._calculate_power_for_distance(distance + 1)
        else:
            # Aim for center
            center = Vector2(court.width / 2, court.height / 2)
            to_center = center - ball.position
            angle = math.atan2(to_center.y, to_center.x)
            power = 5.0

        return (angle, power)

    def _is_good_approach(self, ball: Ball, hoop: Hoop) -> bool:
        """Check if ball has good approach to hoop."""
        to_hoop = hoop.position - ball.position
        distance = to_hoop.magnitude()

        if distance < 0.5 or distance > 6:
            return False

        approach_dir = to_hoop.normalize()
        dot = approach_dir.dot(hoop.direction)
        return dot > 0.5

    def _shoot_for_hoop(self, ball: Ball, target_hoop: Hoop) -> Tuple[float, float]:
        """Shoot to run the hoop."""
        # Aim past the hoop
        aim_point = target_hoop.position + target_hoop.direction * 2

        delta = aim_point - ball.position
        angle = math.atan2(delta.y, delta.x)
        distance = delta.magnitude()
        power = self._calculate_power_for_distance(distance + 3)

        return (angle, power)

    def _shoot_for_position(
        self,
        ball: Ball,
        target_hoop: Hoop,
        court: Court
    ) -> Tuple[float, float]:
        """Shoot to set up for hoop."""
        # Position 4 yards in front of hoop
        setup_pos = target_hoop.position - target_hoop.direction * 4

        # Add slight randomness
        offset = Vector2(
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        )
        setup_pos = setup_pos + offset

        # Clamp to court
        setup_pos.x = max(1, min(court.width - 1, setup_pos.x))
        setup_pos.y = max(1, min(court.height - 1, setup_pos.y))

        delta = setup_pos - ball.position
        angle = math.atan2(delta.y, delta.x)
        distance = delta.magnitude()
        power = self._calculate_power_for_distance(distance)

        return (angle, power)

    def _get_leave_setup_shot(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set]
    ) -> Optional[Tuple[float, float, str]]:
        """
        Get a shot to deliberately set up a good leave.

        Called when the break is ending (no continuation possible) to
        position balls according to standard leave patterns.

        Args:
            ball: Current striker ball
            balls: All balls on court
            court: The court
            deadness: Which balls are dead

        Returns:
            Tuple of (angle, power, reason) or None if no leave setup needed
        """
        # Get suggested leave positions for our next hoop
        suggested = self.leave_patterns.suggest_leave_positions(
            ball.color, ball.hoops_run + 1, balls
        )

        if not suggested or ball.color not in suggested:
            return None

        # Target position from leave pattern
        target_pos = suggested[ball.color]

        # Calculate shot to get there
        delta = target_pos - ball.position
        distance = delta.magnitude()

        if distance < 1:
            return None  # Already in position

        angle = math.atan2(delta.y, delta.x)
        power = self._calculate_power_for_distance(distance)

        # Identify which pattern we're setting up
        actual_positions = {c: b.position for c, b in balls.items()}
        best_pattern, similarity = self.leave_patterns.find_best_matching_pattern(
            actual_positions, ball.hoops_run + 1
        )

        pattern_name = best_pattern.name if best_pattern else "defensive"

        return (angle, power, f"LEAVE SETUP: Position for {pattern_name} (dist={distance:.1f})")

    def _should_set_leave(
        self,
        ball: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int,
        is_continuation: bool
    ) -> bool:
        """
        Determine if we should set up a leave instead of continuing break.

        CONSERVATIVE: Only set up leaves when dead on ALL balls.
        This prevents over-defensive play that stalls games.

        Returns True if:
        - This is the last stroke (strokes_remaining == 1)
        - We're not in continuation (no extra strokes earned)
        - Dead on ALL other balls
        - Not in position to run hoop

        Args:
            ball: Current striker ball
            balls: All balls on court
            court: The court
            deadness: Which balls are dead
            strokes_remaining: Strokes remaining
            is_continuation: Whether we have continuation

        Returns:
            True if should set up leave
        """
        # Only on last stroke without continuation
        if strokes_remaining > 1 or is_continuation:
            return False

        # Check if we have ANY live balls to roquet (not dead)
        dead_on = deadness.get(ball.color, set())
        live_ball_exists = False

        for color, other_ball in balls.items():
            if color == ball.color or other_ball.has_pegged_out:
                continue
            if color not in dead_on:
                # There's a ball we're NOT dead on - try to roquet it
                live_ball_exists = True
                break

        # If there's ANY live ball, don't set up leave - try to roquet instead
        if live_ball_exists:
            return False

        # Check if in position to run hoop (even if dead on all balls)
        target_hoop = court.get_hoop_for_ball(ball.hoops_run)
        if target_hoop:
            to_hoop = target_hoop.position - ball.position
            dist_to_hoop = to_hoop.magnitude()
            if dist_to_hoop > 0.5:
                approach_dir = to_hoop.normalize()
                approach_quality = approach_dir.dot(target_hoop.direction)
            else:
                approach_quality = 1.0

            # If close to hoop at all, try to approach/run it
            # Even a poor approach is better than going defensive
            if dist_to_hoop < 8:
                return False

        # TRULY stuck: dead on all balls AND far from hoop
        # Only NOW set up a defensive leave
        return True

    def _calculate_power_for_distance(self, distance: float) -> float:
        """Calculate power needed for distance."""
        friction_decel = config.FRICTION_COEFFICIENT * config.GRAVITY
        velocity = math.sqrt(2 * friction_decel * distance)
        return min(velocity * 1.1, config.MAX_SHOT_POWER)

    def get_description(self) -> str:
        """Return description of this strategy."""
        return f"Learning (skill: {self.skill_level:.0%})"
