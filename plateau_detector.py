#!/usr/bin/env python3
"""
Plateau and Collapse Detector for DQN Training.

Monitors greedy evaluation metrics to detect:
1. Plateau: No meaningful improvement over patience window
2. Collapse: Performance degradation from best
3. Divergence: Loss/Q-value blow-up

Usage:
    python plateau_detector.py                    # Check current training status
    python plateau_detector.py --watch            # Continuous monitoring
    python plateau_detector.py --best             # Just print best checkpoint
"""
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Configuration for plateau/collapse detection."""
    # Smoothing window
    window_size: int = 5  # Rolling mean over last N evals

    # Plateau detection
    improvement_threshold: float = 0.05  # Min improvement in hoops to reset patience
    patience: int = 10  # Evals without improvement before plateau stop

    # Collapse detection
    collapse_drop: float = 0.30  # 30% drop from best triggers collapse
    collapse_consecutive: int = 3  # Must see drop for K consecutive evals
    collapse_min_best: float = 0.5  # Only trigger collapse if best_H >= this (avoid early stops)

    # Divergence detection
    loss_threshold: float = 20.0  # Loss above this is concerning
    loss_increase_count: int = 3  # Consecutive loss increases to trigger
    warmup_steps: int = 5000  # Don't trigger divergence until replay warmed up

    # Turn cap handling
    turn_cap: int = 200  # Max turns in eval games (games hitting this are "timeout")
    timeout_threshold: float = 0.9  # If >90% games timeout, ignore turns as metric

    # Paths
    stats_dir: str = "ai_data/neural"


@dataclass
class DetectorState:
    """Current state of the detector."""
    best_hoops: float = 0.0
    best_score: float = 0.0  # Composite score for comparison
    best_checkpoint: str = ""
    best_step: int = 0
    patience_counter: int = 0
    collapse_counter: int = 0
    loss_increase_counter: int = 0
    prev_loss: float = 0.0

    # History for smoothing
    hoops_history: List[float] = None
    win_rate_history: List[float] = None
    turns_history: List[float] = None
    loss_history: List[float] = None
    q_value_history: List[float] = None

    def __post_init__(self):
        if self.hoops_history is None:
            self.hoops_history = []
        if self.win_rate_history is None:
            self.win_rate_history = []
        if self.turns_history is None:
            self.turns_history = []
        if self.loss_history is None:
            self.loss_history = []
        if self.q_value_history is None:
            self.q_value_history = []


def load_checkpoint_stats(stats_path: Path) -> Optional[Dict]:
    """Load stats from a checkpoint JSON file."""
    try:
        with open(stats_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def get_all_checkpoint_stats(stats_dir: Path) -> List[Tuple[int, Dict]]:
    """Load all checkpoint stats, sorted by episode number."""
    stats = []

    for stats_file in stats_dir.glob("stats_ep*.json"):
        # Extract episode number from filename
        try:
            ep_str = stats_file.stem.replace("stats_ep", "")
            episode = int(ep_str)
        except ValueError:
            continue

        data = load_checkpoint_stats(stats_file)
        if data:
            stats.append((episode, data))

    # Sort by episode number
    stats.sort(key=lambda x: x[0])
    return stats


def rolling_mean(values: List[float], window: int) -> float:
    """Compute rolling mean of last `window` values."""
    if not values:
        return 0.0
    recent = values[-window:]
    return sum(recent) / len(recent)


def compute_score(hoops: float, win_rate: float, turns: float = None,
                  turn_cap: int = 200, timeout_threshold: float = 0.9) -> float:
    """
    Compute composite score for checkpoint comparison.

    Primary: hoops (dominates)
    Secondary: win_rate (tie-breaker, scaled down)
    Tertiary: turns (only if not hitting turn cap)

    Args:
        hoops: Average hoops per game
        win_rate: Win rate percentage (0-100)
        turns: Average turns per game (None to ignore)
        turn_cap: Max turns in eval (games at this value are "timeout")
        timeout_threshold: If turns > cap * threshold, ignore turns metric
    """
    score = hoops + 0.01 * win_rate

    # Only factor in turns if games aren't timing out
    if turns is not None and turns < turn_cap * timeout_threshold:
        # Lower turns is better (more efficient wins)
        # Small bonus for efficiency, capped to not dominate
        efficiency_bonus = max(0, (turn_cap - turns) / turn_cap) * 0.1
        score += efficiency_bonus

    return score


class PlateauDetector:
    """
    Detects training plateau and collapse conditions.

    Monitors greedy evaluation metrics to determine when to:
    - Stop training (plateau or collapse detected)
    - Save best checkpoint
    - Alert on divergence
    """

    def __init__(self, config: DetectorConfig = None):
        self.config = config or DetectorConfig()
        self.state = DetectorState()
        self.stats_dir = Path(self.config.stats_dir)

    def update(self, stats: Dict, checkpoint_name: str) -> Optional[Dict]:
        """
        Update detector with new checkpoint stats.

        Args:
            stats: Checkpoint statistics dict
            checkpoint_name: Name/path of checkpoint

        Returns:
            Dict with detection results, or None if checkpoint lacks greedy metrics
        """
        # Skip checkpoints without greedy eval metrics (old format)
        if 'greedy_avg_hoops' not in stats:
            return None

        # Extract metrics
        hoops = stats.get('greedy_avg_hoops', 0.0)
        win_rate = stats.get('greedy_win_rate', 0.0)
        turns = stats.get('greedy_avg_turns', self.config.turn_cap)
        loss = stats.get('avg_loss', 0.0)
        q_value = stats.get('avg_q_value', 0.0)
        step = stats.get('total_steps', 0)

        # Update history
        self.state.hoops_history.append(hoops)
        self.state.win_rate_history.append(win_rate)
        self.state.turns_history.append(turns)
        self.state.loss_history.append(loss)
        self.state.q_value_history.append(q_value)

        # Compute smoothed metrics
        smoothed_hoops = rolling_mean(
            self.state.hoops_history, self.config.window_size
        )
        smoothed_win_rate = rolling_mean(
            self.state.win_rate_history, self.config.window_size
        )
        smoothed_turns = rolling_mean(
            self.state.turns_history, self.config.window_size
        )

        # Compute composite score
        current_score = compute_score(
            smoothed_hoops, smoothed_win_rate, smoothed_turns,
            self.config.turn_cap, self.config.timeout_threshold
        )

        # Check for improvement (use hoops as primary metric)
        improved = smoothed_hoops > self.state.best_hoops + self.config.improvement_threshold

        if improved:
            self.state.best_hoops = smoothed_hoops
            self.state.best_score = current_score
            self.state.best_checkpoint = checkpoint_name
            self.state.best_step = step
            self.state.patience_counter = 0
            self.state.collapse_counter = 0
        else:
            self.state.patience_counter += 1

        # Check for collapse (significant drop from best)
        # Only check collapse if best_hoops meets minimum threshold (avoid early false positives)
        if (self.state.best_hoops >= self.config.collapse_min_best and
            self.state.best_hoops > 0):
            drop_ratio = (self.state.best_hoops - smoothed_hoops) / self.state.best_hoops
            if drop_ratio >= self.config.collapse_drop:
                self.state.collapse_counter += 1
            else:
                self.state.collapse_counter = 0
        else:
            self.state.collapse_counter = 0  # Reset if best not yet meaningful

        # Check for loss divergence (only after warmup to avoid early noise)
        if step > self.config.warmup_steps:
            if loss > self.config.loss_threshold and loss > self.state.prev_loss:
                self.state.loss_increase_counter += 1
            else:
                self.state.loss_increase_counter = 0
        self.state.prev_loss = loss

        # Determine stop conditions
        # Apply warmup gating to ALL stop conditions (not just divergence)
        # Early training is noisy - don't stop until we have enough data
        warmup_complete = step > self.config.warmup_steps

        plateau_stop = (warmup_complete and
                       self.state.patience_counter >= self.config.patience)
        collapse_stop = (warmup_complete and
                        self.state.collapse_counter >= self.config.collapse_consecutive)
        divergence_stop = (warmup_complete and
                          self.state.loss_increase_counter >= self.config.loss_increase_count)

        should_stop = plateau_stop or collapse_stop or divergence_stop
        stop_reason = None
        if plateau_stop:
            stop_reason = "plateau"
        elif collapse_stop:
            stop_reason = "collapse"
        elif divergence_stop:
            stop_reason = "divergence"

        return {
            'checkpoint': checkpoint_name,
            'step': step,
            'hoops': hoops,
            'smoothed_hoops': smoothed_hoops,
            'win_rate': win_rate,
            'smoothed_win_rate': smoothed_win_rate,
            'turns': turns,
            'smoothed_turns': smoothed_turns,
            'score': current_score,
            'loss': loss,
            'q_value': q_value,
            'best_hoops': self.state.best_hoops,
            'best_score': self.state.best_score,
            'best_checkpoint': self.state.best_checkpoint,
            'patience': self.state.patience_counter,
            'collapse_count': self.state.collapse_counter,
            'should_stop': should_stop,
            'stop_reason': stop_reason,
            'improved': improved,
        }

    def analyze_all(self) -> List[Dict]:
        """Analyze all checkpoints in order."""
        all_stats = get_all_checkpoint_stats(self.stats_dir)
        results = []

        for episode, stats in all_stats:
            checkpoint_name = f"ep{episode}"
            result = self.update(stats, checkpoint_name)
            # Skip checkpoints without greedy metrics (old format)
            if result is not None:
                results.append(result)

        return results

    def get_summary(self) -> Dict:
        """Get current detector summary."""
        return {
            'best_checkpoint': self.state.best_checkpoint,
            'best_hoops': self.state.best_hoops,
            'best_score': self.state.best_score,
            'best_step': self.state.best_step,
            'current_patience': self.state.patience_counter,
            'max_patience': self.config.patience,
            'collapse_count': self.state.collapse_counter,
            'collapse_threshold': self.config.collapse_consecutive,
        }


def print_status(result: Dict, verbose: bool = False):
    """Print status for a single checkpoint."""
    # Use ASCII-safe characters for Windows compatibility
    status_char = "+" if result['improved'] else "."
    if result['should_stop']:
        status_char = "X"

    print(f"{status_char} {result['checkpoint']:>8} | "
          f"H={result['hoops']:.2f} (smooth={result['smoothed_hoops']:.2f}) | "
          f"WR={result['win_rate']:.0f}% | "
          f"Loss={result['loss']:.2f} | "
          f"Patience={result['patience']}/{10}")

    if result['should_stop']:
        print(f"  *** STOP: {result['stop_reason'].upper()} detected ***")
        print(f"  *** Best checkpoint: {result['best_checkpoint']} "
              f"(H={result['best_hoops']:.2f}) ***")


def main():
    parser = argparse.ArgumentParser(
        description="Plateau and collapse detector for DQN training"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuous monitoring mode"
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Just print best checkpoint"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--dir",
        default="ai_data/neural",
        help="Directory containing checkpoint stats"
    )
    args = parser.parse_args()

    config = DetectorConfig(stats_dir=args.dir)
    detector = PlateauDetector(config)

    if args.watch:
        print("Watching for new checkpoints... (Ctrl+C to stop)")
        seen_checkpoints = set()
        stop_triggered = False

        while True:
            all_stats = get_all_checkpoint_stats(Path(args.dir))

            for episode, stats in all_stats:
                checkpoint_name = f"ep{episode}"
                if checkpoint_name not in seen_checkpoints:
                    result = detector.update(stats, checkpoint_name)
                    seen_checkpoints.add(checkpoint_name)

                    # Skip checkpoints without greedy metrics (old format)
                    if result is None:
                        continue

                    print_status(result, args.verbose)

                    if result['should_stop'] and not stop_triggered:
                        print("\n" + "=" * 60)
                        print("RECOMMENDATION: Stop training and use best checkpoint")
                        summary = detector.get_summary()
                        print(f"Best: {summary['best_checkpoint']} (H={summary['best_hoops']:.2f})")
                        print("=" * 60)
                        stop_triggered = True

            time.sleep(10)  # Check every 10 seconds

    else:
        # One-shot analysis
        results = detector.analyze_all()

        if not results:
            print("No checkpoint stats found.")
            return

        if args.best:
            summary = detector.get_summary()
            print(f"Best checkpoint: {summary['best_checkpoint']}")
            print(f"Best greedy hoops: {summary['best_hoops']:.2f}")
            return

        print("=" * 70)
        print("PLATEAU DETECTOR ANALYSIS")
        print("=" * 70)
        print(f"{'Checkpoint':>10} | {'Hoops':>6} {'(smooth)':>8} | "
              f"{'WR':>4} | {'Loss':>6} | {'Patience':>10}")
        print("-" * 70)

        for result in results:
            print_status(result, args.verbose)

        print("-" * 70)
        summary = detector.get_summary()
        print(f"\nBest checkpoint: {summary['best_checkpoint']} "
              f"(H={summary['best_hoops']:.2f})")
        print(f"Current patience: {summary['current_patience']}/{summary['max_patience']}")

        if results[-1]['should_stop']:
            print(f"\n*** RECOMMENDATION: Stop training ({results[-1]['stop_reason']}) ***")
            print(f"*** Use checkpoint: {summary['best_checkpoint']} ***")


if __name__ == "__main__":
    main()
