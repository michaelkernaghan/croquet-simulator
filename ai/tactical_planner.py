#!/usr/bin/env python3
"""
LLM-based Tactical Planner for Association Croquet.

Provides strategic guidance by calling an LLM with game state context.
Can be used as:
- Decision-time advisor (Phase 1)
- Data collection for behavior cloning (Phase 2)
- Training-time guidance (Phase 3)

The planner respects valid_intents constraints and provides fallback
behavior when the LLM is unavailable or returns invalid responses.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import os

# Try to import LLM client (optional dependency)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from models.ball import Ball, Vector2
from models.court import Court


# Intent action mapping (matches neural network action space)
INTENT_NAMES = [
    "HOOP_RUN",        # 0
    "ROQUET_NEAREST",  # 1
    "ROQUET_PARTNER",  # 2
    "ROQUET_OPPONENT1", # 3
    "ROQUET_OPPONENT2", # 4
    "APPROACH",        # 5
    "DEFENSIVE",       # 6
    "PEG_OUT",         # 7
]

INTENT_TO_IDX = {name: idx for idx, name in enumerate(INTENT_NAMES)}


@dataclass
class RankedIntent:
    """A single ranked intent from the planner."""
    intent: int  # 0-7
    intent_name: str
    reason: str
    expected: str
    risk: str
    fallback: Optional[int] = None


@dataclass
class PlannerOutput:
    """Output from the tactical planner."""
    ranked_intents: List[RankedIntent]
    one_sentence_plan: str
    risks_and_counters: List[Dict]
    global_phase: str = ""
    raw_response: Optional[Dict] = None
    cached: bool = False
    latency_ms: float = 0.0


@dataclass
class PlannerConfig:
    """Configuration for the tactical planner."""
    # LLM settings
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1500
    temperature: float = 0.0  # Deterministic for repeatability
    timeout_ms: int = 10000

    # Caching
    enable_cache: bool = True
    cache_dir: str = "ai_data/planner_cache"

    # Fallback behavior
    fallback_on_error: bool = True
    max_retries: int = 2

    # Rate limiting
    min_interval_ms: int = 100  # Minimum time between calls

    # Prompt version (for cache invalidation)
    prompt_version: str = "v1.0"


class TacticalPlanner:
    """
    LLM-based tactical planner for croquet.

    Converts game state to JSON, calls LLM with croquet-specific prompt,
    and returns validated intent recommendations.
    """

    def __init__(self, config: PlannerConfig = None):
        self.config = config or PlannerConfig()
        self.cache: Dict[str, PlannerOutput] = {}
        self.last_call_time = 0.0

        # Initialize LLM client if available
        self.client = None
        if ANTHROPIC_AVAILABLE:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)

        # Ensure cache directory exists
        if self.config.enable_cache:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.client is not None

    def state_to_json(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int,
        valid_intents: List[int],
        turn_number: int = 0,
        game_score: Optional[Dict] = None
    ) -> Dict:
        """
        Convert game state to JSON for the LLM.

        Keeps it minimal but includes all tactical information needed.
        """
        # Determine sides
        striker_color = striker.color
        if striker_color in ["blue", "black"]:
            striker_side = "A"
            partner_color = "black" if striker_color == "blue" else "blue"
            opponent_colors = ["red", "yellow"]
        else:
            striker_side = "B"
            partner_color = "yellow" if striker_color == "red" else "red"
            opponent_colors = ["blue", "black"]

        # Build ball states
        ball_states = {}
        for color, ball in balls.items():
            if ball.has_pegged_out:
                continue

            target_hoop = court.get_hoop_for_ball(ball.hoops_run)
            hoop_info = None
            if target_hoop:
                dist_to_hoop = (target_hoop.position - ball.position).magnitude()
                hoop_info = {
                    "number": target_hoop.number,
                    "distance": round(dist_to_hoop, 1),
                    "position": [round(target_hoop.position.x, 1), round(target_hoop.position.y, 1)]
                }

            ball_states[color] = {
                "position": [round(ball.position.x, 1), round(ball.position.y, 1)],
                "hoops_run": ball.hoops_run,
                "is_rover": ball.hoops_run >= 12,
                "next_hoop": hoop_info,
            }

        # Compute distances between balls
        distances = {}
        for c1, b1 in balls.items():
            if b1.has_pegged_out:
                continue
            for c2, b2 in balls.items():
                if c2 <= c1 or b2.has_pegged_out:
                    continue
                dist = (b1.position - b2.position).magnitude()
                distances[f"{c1}-{c2}"] = round(dist, 1)

        # Deadness info for striker
        dead_on = list(deadness.get(striker_color, set()))

        # Valid intents as names
        valid_intent_names = [INTENT_NAMES[i] for i in valid_intents]

        # Game score
        if game_score is None:
            # Compute from ball states
            side_a_hoops = sum(
                balls[c].hoops_run for c in ["blue", "black"]
                if c in balls and not balls[c].has_pegged_out
            )
            side_b_hoops = sum(
                balls[c].hoops_run for c in ["red", "yellow"]
                if c in balls and not balls[c].has_pegged_out
            )
            game_score = {"side_A": side_a_hoops, "side_B": side_b_hoops}

        return {
            "striker": {
                "color": striker_color,
                "side": striker_side,
                "position": [round(striker.position.x, 1), round(striker.position.y, 1)],
                "hoops_run": striker.hoops_run,
                "is_rover": striker.hoops_run >= 12,
                "dead_on": dead_on,
            },
            "partner": partner_color,
            "opponents": opponent_colors,
            "balls": ball_states,
            "distances": distances,
            "strokes_remaining": strokes_remaining,
            "valid_intents": valid_intent_names,
            "valid_intent_indices": valid_intents,
            "turn_number": turn_number,
            "game_score": game_score,
            "court_size": [court.width, court.height],
        }

    def _build_prompt(self, state_json: Dict) -> str:
        """Build the full prompt for the LLM."""
        state_str = json.dumps(state_json, indent=2)

        return f"""You are an Association Croquet tactical planner. Analyze the game state and recommend actions.

RULES CONTEXT:
- Turn continues if striker runs next hoop or roquets a LIVE ball
- After roquet: take croquet stroke then continuation stroke
- Deadness prevents re-roqueting same ball for bonuses in same turn
- Win by completing all hoops + peg with both balls
- Only recommend intents from valid_intents list

ACTION SPACE (8 intents):
0 HOOP_RUN - Attempt to run the next hoop
1 ROQUET_NEAREST - Roquet the nearest available ball
2 ROQUET_PARTNER - Roquet partner ball
3 ROQUET_OPPONENT1 - Roquet first opponent
4 ROQUET_OPPONENT2 - Roquet second opponent
5 APPROACH - Position for next hoop without running it
6 DEFENSIVE - Safe shot to boundary or away from opponents
7 PEG_OUT - Hit the peg to finish (rover only)

CURRENT STATE:
{state_str}

Return a JSON object with exactly these keys:
{{
  "global_phase": "opening|middlegame|endgame|pegging_out",
  "current_stroke": {{
    "recommended_intents_ranked": [
      {{ "intent": <0-7>, "reason": "...", "expected": "...", "risk": "...", "fallback": <0-7 or null> }}
    ],
    "one_sentence_plan": "..."
  }},
  "risks_and_counters": [
    {{ "risk": "...", "counter_intent": <0-7>, "how": "..." }}
  ]
}}

RULES:
- Only use intents from valid_intents list (indices: {state_json['valid_intent_indices']})
- Max 5 ranked intents, max 5 risks
- If PEG_OUT is valid and striker is rover near peg, include it
- Prefer actions that keep turn alive (secure roquet or hoop)
- Be concise"""

    def _cache_key(self, state_json: Dict) -> str:
        """Generate cache key from state."""
        # Include prompt version for cache invalidation
        key_data = {
            "state": state_json,
            "prompt_version": self.config.prompt_version,
            "model": self.config.model,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _load_from_cache(self, cache_key: str) -> Optional[PlannerOutput]:
        """Try to load from memory or disk cache."""
        # Memory cache first
        if cache_key in self.cache:
            output = self.cache[cache_key]
            output.cached = True
            return output

        # Disk cache
        if self.config.enable_cache:
            cache_path = Path(self.config.cache_dir) / f"{cache_key}.json"
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        data = json.load(f)
                    output = self._parse_response(data)
                    if output:
                        output.cached = True
                        self.cache[cache_key] = output
                        return output
                except Exception:
                    pass

        return None

    def _save_to_cache(self, cache_key: str, response: Dict):
        """Save response to disk cache."""
        if self.config.enable_cache:
            cache_path = Path(self.config.cache_dir) / f"{cache_key}.json"
            try:
                with open(cache_path, 'w') as f:
                    json.dump(response, f)
            except Exception:
                pass

    def _parse_response(self, response: Dict) -> Optional[PlannerOutput]:
        """Parse LLM response into PlannerOutput."""
        try:
            current = response.get("current_stroke", {})
            ranked_raw = current.get("recommended_intents_ranked", [])

            ranked_intents = []
            for item in ranked_raw:
                intent_idx = item.get("intent")
                if isinstance(intent_idx, int) and 0 <= intent_idx < 8:
                    ranked_intents.append(RankedIntent(
                        intent=intent_idx,
                        intent_name=INTENT_NAMES[intent_idx],
                        reason=item.get("reason", ""),
                        expected=item.get("expected", ""),
                        risk=item.get("risk", ""),
                        fallback=item.get("fallback"),
                    ))

            return PlannerOutput(
                ranked_intents=ranked_intents,
                one_sentence_plan=current.get("one_sentence_plan", ""),
                risks_and_counters=response.get("risks_and_counters", []),
                global_phase=response.get("global_phase", ""),
                raw_response=response,
            )
        except Exception as e:
            print(f"  [PLANNER] Failed to parse response: {e}")
            return None

    def _validate_output(
        self,
        output: PlannerOutput,
        valid_intents: List[int]
    ) -> PlannerOutput:
        """Filter output to only include valid intents."""
        valid_ranked = [
            ri for ri in output.ranked_intents
            if ri.intent in valid_intents
        ]

        # Also validate fallbacks
        for ri in valid_ranked:
            if ri.fallback is not None and ri.fallback not in valid_intents:
                ri.fallback = None

        output.ranked_intents = valid_ranked
        return output

    def plan(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int,
        valid_intents: List[int],
        turn_number: int = 0,
        game_score: Optional[Dict] = None,
        force_refresh: bool = False
    ) -> Optional[PlannerOutput]:
        """
        Get tactical plan for current game state.

        Args:
            striker: Current striker ball
            balls: All balls in play
            court: Court object
            deadness: Deadness dictionary
            strokes_remaining: Strokes left in turn
            valid_intents: List of valid action indices
            turn_number: Current turn number
            game_score: Optional game score dict
            force_refresh: Bypass cache

        Returns:
            PlannerOutput with ranked intents, or None on failure
        """
        if not valid_intents:
            return None

        # Convert state to JSON
        state_json = self.state_to_json(
            striker, balls, court, deadness,
            strokes_remaining, valid_intents,
            turn_number, game_score
        )

        # Check cache
        cache_key = self._cache_key(state_json)
        if not force_refresh:
            cached = self._load_from_cache(cache_key)
            if cached:
                return self._validate_output(cached, valid_intents)

        # Check if LLM is available
        if not self.is_available():
            return None

        # Rate limiting
        now = time.time() * 1000
        elapsed = now - self.last_call_time
        if elapsed < self.config.min_interval_ms:
            time.sleep((self.config.min_interval_ms - elapsed) / 1000)

        # Build prompt
        prompt = self._build_prompt(state_json)

        # Call LLM
        start_time = time.time()
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )

                self.last_call_time = time.time() * 1000

                # Extract JSON from response
                content = response.content[0].text

                # Try to parse JSON (handle markdown code blocks)
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                response_json = json.loads(content.strip())

                # Parse and validate
                output = self._parse_response(response_json)
                if output:
                    output.latency_ms = (time.time() - start_time) * 1000
                    output = self._validate_output(output, valid_intents)

                    # Cache valid response
                    if output.ranked_intents:
                        self._save_to_cache(cache_key, response_json)
                        self.cache[cache_key] = output

                    return output

            except json.JSONDecodeError as e:
                print(f"  [PLANNER] JSON parse error (attempt {attempt+1}): {e}")
            except Exception as e:
                print(f"  [PLANNER] LLM call failed (attempt {attempt+1}): {e}")

        return None

    def get_top_intent(
        self,
        output: Optional[PlannerOutput],
        default: int = 5  # APPROACH
    ) -> int:
        """Get the top recommended intent, or default on failure."""
        if output and output.ranked_intents:
            return output.ranked_intents[0].intent
        return default

    def format_plan(self, output: Optional[PlannerOutput]) -> str:
        """Format plan for display/logging."""
        if not output:
            return "[No plan available]"

        lines = []
        lines.append(f"Phase: {output.global_phase}")
        lines.append(f"Plan: {output.one_sentence_plan}")
        lines.append("Ranked intents:")
        for i, ri in enumerate(output.ranked_intents[:3]):
            lines.append(f"  {i+1}. {ri.intent_name}: {ri.reason}")

        if output.cached:
            lines.append("(cached)")
        else:
            lines.append(f"(latency: {output.latency_ms:.0f}ms)")

        return "\n".join(lines)


# Convenience function for quick testing
def test_planner():
    """Quick test of the planner with a sample state."""
    from models.court import Court
    from models.ball import Ball, Vector2

    court = Court()
    balls = {
        "blue": Ball("blue", (7, 5)),
        "black": Ball("black", (15, 20)),
        "red": Ball("red", (10, 10)),
        "yellow": Ball("yellow", (20, 30)),
    }
    balls["blue"].hoops_run = 2
    balls["black"].hoops_run = 1
    balls["red"].hoops_run = 3
    balls["yellow"].hoops_run = 0

    striker = balls["blue"]
    deadness = {"blue": set(), "black": set(), "red": set(), "yellow": set()}
    valid_intents = [0, 1, 2, 3, 5, 6]  # No ROQUET_OPP2 or PEG_OUT

    planner = TacticalPlanner()

    if not planner.is_available():
        print("LLM not available (no ANTHROPIC_API_KEY)")
        print("\nState JSON that would be sent:")
        state = planner.state_to_json(
            striker, balls, court, deadness,
            strokes_remaining=1,
            valid_intents=valid_intents
        )
        print(json.dumps(state, indent=2))
        return

    print("Calling planner...")
    output = planner.plan(
        striker, balls, court, deadness,
        strokes_remaining=1,
        valid_intents=valid_intents
    )

    print(planner.format_plan(output))


if __name__ == "__main__":
    test_planner()
