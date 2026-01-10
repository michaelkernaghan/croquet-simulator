#!/usr/bin/env python3
"""
AlphaZero-style training for croquet.

This training mode combines:
1. PolicyValueNet: Dual-head network outputting policy + value
2. MCTS: Monte Carlo Tree Search for action selection
3. Self-play: Both sides use neural network
4. Policy gradient: Train on MCTS visit distributions + game outcomes

Key differences from DQN training:
- Actions selected via MCTS (multi-step lookahead) not single-step Q-values
- Training targets are MCTS policies + game outcomes, not TD targets
- No epsilon-greedy exploration (MCTS + Dirichlet noise handles exploration)
- Sparse rewards only (win/loss), no intermediate shaping

Usage:
    python train_alphazero.py --episodes 1000 --simulations 50
    python train_alphazero.py --checkpoint models/alphazero_latest.pt --episodes 500
"""
import argparse
import copy
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required for AlphaZero training")

from models.ball import Ball, Vector2
from models.court import Court
from rules.rule_engine import RuleEngine, TurnState
from physics.physics_engine import PhysicsEngine
from ai.tactical_decision_maker import TacticalDecisionMaker
from ai.neural.croquet_net import PolicyValueNet, StateEncoder, get_device
from ai.neural.mcts import MCTS, MCTSConfig, CroquetSimulator, create_mcts_for_training
import config


@dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero training."""

    # Network
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.2

    # MCTS
    num_simulations: int = 50       # MCTS simulations per move
    c_puct: float = 1.5             # Exploration constant
    dirichlet_alpha: float = 0.3    # Root noise parameter
    root_noise_frac: float = 0.25   # Noise fraction at root

    # Temperature
    temperature_start: float = 1.0   # Initial temperature for action selection
    temperature_end: float = 0.1     # Final temperature
    temperature_decay_steps: int = 30  # Steps per game before temperature drops

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    replay_buffer_size: int = 100000
    min_buffer_size: int = 1000     # Minimum samples before training
    train_steps_per_game: int = 10  # Training steps after each game

    # Self-play
    opponent_update_freq: int = 100  # Games between opponent updates
    use_past_opponent: bool = True   # Use past checkpoint as opponent

    # Checkpointing
    checkpoint_freq: int = 50       # Games between checkpoints
    eval_freq: int = 100            # Games between evaluation
    save_dir: str = "models/alphazero"


@dataclass
class TrainingExample:
    """Single training example from self-play."""
    state: torch.Tensor       # Encoded state
    policy: np.ndarray        # MCTS visit count distribution
    value: float              # Game outcome from this player's perspective


class ReplayBuffer:
    """Circular buffer for training examples."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[TrainingExample] = []
        self.position = 0

    def push(self, example: TrainingExample):
        """Add example to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(example)
        else:
            self.buffer[self.position] = example
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[TrainingExample]:
        """Sample random batch from buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class AlphaZeroTrainer:
    """
    AlphaZero training loop.

    Implements the core AlphaZero algorithm:
    1. Self-play games using MCTS to generate training data
    2. Train network on (state, MCTS policy, game outcome) tuples
    3. Periodically update opponent checkpoint
    """

    def __init__(self, config: AlphaZeroConfig = None, verbose: bool = False):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            verbose: Print detailed progress
        """
        self.config = config or AlphaZeroConfig()
        self.verbose = verbose
        self.device = get_device()

        # Create network
        self.network = PolicyValueNet(
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # State encoder
        self.encoder = StateEncoder()

        # MCTS configuration
        self.mcts_config = MCTSConfig(
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            root_noise_frac=self.config.root_noise_frac
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)

        # Opponent network (for self-play diversity)
        self.opponent_net = None
        self.games_since_opponent_update = 0

        if self.config.use_past_opponent:
            self.opponent_net = copy.deepcopy(self.network)
            self.opponent_net.eval()

        # Game components
        self.court = Court()

        # Training stats
        self.total_games = 0
        self.total_train_steps = 0
        self.game_results = []

        # Create save directory
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def self_play_game(self) -> List[TrainingExample]:
        """
        Play one game using MCTS and collect training examples.

        Returns:
            List of TrainingExample tuples from the game
        """
        # Initialize game - position balls near hoop 1
        balls = {
            "blue": Ball("blue", (random.uniform(5, 9), random.uniform(2, 5))),
            "red": Ball("red", (random.uniform(5, 9), random.uniform(2, 5))),
            "black": Ball("black", (random.uniform(5, 9), random.uniform(2, 5))),
            "yellow": Ball("yellow", (random.uniform(5, 9), random.uniform(2, 5)))
        }

        # Ensure balls are at rest
        for ball in balls.values():
            ball.velocity = Vector2(0, 0)

        rules = RuleEngine()
        physics = PhysicsEngine(self.court)

        # Create MCTS for each side
        main_mcts = MCTS(self.network, self.mcts_config, str(self.device))
        opponent_mcts = MCTS(
            self.opponent_net if self.opponent_net else self.network,
            self.mcts_config,
            str(self.device)
        )

        # Create simulator
        simulator = CroquetSimulator(self.encoder, TacticalDecisionMaker)

        # Game history: (state, policy, player_side)
        game_history = []

        # Play turn order
        turn_order = ["blue", "red", "black", "yellow"]
        turn_idx = 0
        max_turns = 200
        step_count = 0

        for turn in range(max_turns):
            current_color = turn_order[turn_idx % 4]
            ball = balls[current_color]
            is_main_player = current_color in ["blue", "black"]

            # Skip if ball is pegged out
            if ball.has_pegged_out:
                turn_idx += 1
                continue

            # Initialize turn
            rules.start_turn(current_color)

            while rules.turn_info and rules.turn_info.strokes_remaining > 0:
                # Encode state
                deadness = rules.deadness
                strokes_remaining = rules.turn_info.strokes_remaining
                is_continuation = rules.turn_info.state in [TurnState.CONTINUATION, TurnState.CROQUET_TAKEN]

                state = self.encoder.encode(
                    ball, balls, self.court, deadness, strokes_remaining, is_continuation
                )

                # Get valid actions
                dm = TacticalDecisionMaker()
                valid_actions = dm._get_valid_neural_actions(ball, balls, self.court, deadness)

                if not valid_actions:
                    break

                # Create game state dict for simulator
                game_state = {
                    'striker_color': current_color,
                    'balls': balls,
                    'court': self.court,
                    'deadness': deadness,
                    'strokes_remaining': strokes_remaining,
                    'is_continuation': is_continuation
                }

                # Use MCTS to get policy
                mcts = main_mcts if is_main_player else opponent_mcts
                add_noise = is_main_player  # Only add exploration noise for main player

                mcts_policy = mcts.search(
                    state, valid_actions, simulator, game_state, add_noise=add_noise
                )

                # Select action with temperature
                temperature = self._get_temperature(step_count)
                action = self._select_action_with_temperature(mcts_policy, valid_actions, temperature)

                # Store training example (only for main player)
                if is_main_player:
                    game_history.append((state.clone(), mcts_policy.copy(), current_color))

                # Execute action
                game_state_dict = {
                    'strokes_remaining': strokes_remaining,
                    'is_continuation': is_continuation
                }
                shot = dm._neural_action_to_shot(action, 0.0, ball, balls, self.court, deadness, game_state_dict)
                if shot:
                    # Calculate velocity from shot target
                    to_target = shot.target - ball.position
                    distance = to_target.magnitude()
                    angle = to_target.normalize() if distance > 0.1 else Vector2(1, 0)

                    # Calculate power based on distance
                    power = min(distance * 0.8 + 3.0, 10.0)
                    velocity = angle * power

                    physics.shoot_ball(ball, velocity)

                    # Simulate physics until balls stop
                    dt = 1.0 / 60.0
                    for _ in range(300):  # Max 5 seconds
                        physics.update(balls, dt)
                        if all(b.velocity.magnitude() < 0.1 for b in balls.values()):
                            break

                # Process result
                shot_collisions = []
                turn_continues, events = rules.process_stroke_result(
                    ball, balls, self.court, shot_collisions
                )

                step_count += 1

                if not turn_continues:
                    break

            turn_idx += 1

            # Check for game end
            bb_out = sum(1 for c in ["blue", "black"] if balls[c].has_pegged_out)
            ry_out = sum(1 for c in ["red", "yellow"] if balls[c].has_pegged_out)

            if bb_out == 2 or ry_out == 2:
                break

        # Determine winner and assign values
        bb_score = sum(balls[c].hoops_run for c in ["blue", "black"])
        ry_score = sum(balls[c].hoops_run for c in ["red", "yellow"])

        bb_out = sum(1 for c in ["blue", "black"] if balls[c].has_pegged_out)
        ry_out = sum(1 for c in ["red", "yellow"] if balls[c].has_pegged_out)

        if bb_out == 2:
            winner = "blue_black"
        elif ry_out == 2:
            winner = "red_yellow"
        elif bb_score > ry_score:
            winner = "blue_black"
        elif ry_score > bb_score:
            winner = "red_yellow"
        else:
            winner = "draw"

        # Convert game history to training examples
        examples = []
        for state, policy, player_color in game_history:
            player_side = "blue_black" if player_color in ["blue", "black"] else "red_yellow"

            # Assign value based on game outcome
            if winner == "draw":
                value = 0.0
            elif winner == player_side:
                value = 1.0
            else:
                value = -1.0

            examples.append(TrainingExample(state=state, policy=policy, value=value))

        # Record game result
        self.game_results.append({
            'winner': winner,
            'bb_score': bb_score,
            'ry_score': ry_score,
            'steps': step_count,
            'examples': len(examples)
        })

        return examples

    def _get_temperature(self, step: int) -> float:
        """Get temperature for action selection based on step count."""
        if step < self.config.temperature_decay_steps:
            return self.config.temperature_start
        else:
            return self.config.temperature_end

    def _select_action_with_temperature(
        self,
        policy: np.ndarray,
        valid_actions: List[int],
        temperature: float
    ) -> int:
        """Select action from policy with temperature."""
        if temperature == 0:
            # Greedy
            valid_probs = [(a, policy[a]) for a in valid_actions]
            return max(valid_probs, key=lambda x: x[1])[0]
        else:
            # Sample with temperature
            probs = np.array([policy[a] for a in valid_actions])
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
            return np.random.choice(valid_actions, p=probs)

    def train_step(self) -> float:
        """
        Perform one training step on replay buffer.

        Returns:
            Total loss value
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return 0.0

        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Prepare tensors
        states = torch.stack([ex.state for ex in batch]).to(self.device)
        target_policies = torch.tensor(
            np.stack([ex.policy for ex in batch]),
            dtype=torch.float32
        ).to(self.device)
        target_values = torch.tensor(
            [ex.value for ex in batch],
            dtype=torch.float32
        ).unsqueeze(1).to(self.device)

        # Forward pass
        policy_logits, values = self.network(states)

        # Policy loss: cross-entropy with MCTS policy
        # Note: Using soft targets (MCTS visit distribution)
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.sum(target_policies * policy_log_probs, dim=-1).mean()

        # Value loss: MSE with game outcome
        value_loss = F.mse_loss(values, target_values)

        # Combined loss (AlphaZero uses equal weighting)
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)

        self.optimizer.step()
        self.total_train_steps += 1

        return total_loss.item()

    def train(self, num_games: int, start_game: int = 0):
        """
        Main training loop.

        Args:
            num_games: Number of self-play games to run
            start_game: Starting game number (for resuming)
        """
        print("=" * 60)
        print("ALPHAZERO CROQUET TRAINING")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"MCTS simulations: {self.config.num_simulations}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Replay buffer: {self.config.replay_buffer_size}")
        print()

        start_time = time.time()

        for game in range(start_game, start_game + num_games):
            game_start = time.time()

            # Self-play game
            examples = self.self_play_game()

            # Add examples to replay buffer
            for ex in examples:
                self.replay_buffer.push(ex)

            self.total_games += 1

            # Training steps
            train_losses = []
            for _ in range(self.config.train_steps_per_game):
                loss = self.train_step()
                if loss > 0:
                    train_losses.append(loss)

            # Update opponent network periodically
            if self.config.use_past_opponent:
                self.games_since_opponent_update += 1
                if self.games_since_opponent_update >= self.config.opponent_update_freq:
                    self.opponent_net = copy.deepcopy(self.network)
                    self.opponent_net.eval()
                    self.games_since_opponent_update = 0
                    print(f"  [SELF-PLAY] Updated opponent network at game {game + 1}")

            # Progress report
            if (game + 1) % 10 == 0 or game == start_game:
                result = self.game_results[-1]
                avg_loss = np.mean(train_losses) if train_losses else 0.0
                game_time = time.time() - game_start

                print(f"Game {game + 1}/{start_game + num_games} | "
                      f"Winner: {result['winner'][:3]} | "
                      f"Score: {result['bb_score']}-{result['ry_score']} | "
                      f"Examples: {result['examples']} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Buffer: {len(self.replay_buffer)} | "
                      f"Time: {game_time:.1f}s")

            # Checkpoint
            if (game + 1) % self.config.checkpoint_freq == 0:
                self.save_checkpoint(f"alphazero_game_{game + 1}.pt")

            # Evaluation
            if (game + 1) % self.config.eval_freq == 0:
                self._evaluate()

        # Final save
        self.save_checkpoint("alphazero_final.pt")

        total_time = time.time() - start_time
        print()
        print("=" * 60)
        print(f"Training complete!")
        print(f"Total games: {self.total_games}")
        print(f"Total training steps: {self.total_train_steps}")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print("=" * 60)

    def _evaluate(self):
        """Run evaluation games against random opponent."""
        wins = 0
        total_score = 0

        self.network.eval()

        for _ in range(10):
            # Simplified evaluation - just check network predictions
            # Full evaluation would play games against baseline
            pass

        self.network.train()

        # For now, just report recent game stats
        recent = self.game_results[-min(100, len(self.game_results)):]
        bb_wins = sum(1 for r in recent if r['winner'] == 'blue_black')
        avg_score = np.mean([r['bb_score'] for r in recent])

        print(f"  [EVAL] Last {len(recent)} games: {bb_wins} BB wins ({100*bb_wins/len(recent):.0f}%), "
              f"avg score: {avg_score:.1f}")

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.save_dir / filename
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'total_games': self.total_games,
            'total_train_steps': self.total_train_steps,
            'game_results': self.game_results[-1000:],  # Keep last 1000
        }, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.total_games = checkpoint.get('total_games', 0)
        self.total_train_steps = checkpoint.get('total_train_steps', 0)
        self.game_results = checkpoint.get('game_results', [])

        # Update opponent network
        if self.config.use_past_opponent:
            self.opponent_net = copy.deepcopy(self.network)
            self.opponent_net.eval()

        print(f"Loaded checkpoint from {path}")
        print(f"  Games: {self.total_games}, Train steps: {self.total_train_steps}")


def main():
    parser = argparse.ArgumentParser(description="AlphaZero-style croquet training")

    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of self-play games (default: 1000)")
    parser.add_argument("--simulations", type=int, default=50,
                       help="MCTS simulations per move (default: 50)")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Training batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate (default: 0.001)")

    # Self-play
    parser.add_argument("--opponent-update-freq", type=int, default=100,
                       help="Games between opponent updates (default: 100)")
    parser.add_argument("--no-past-opponent", action="store_true",
                       help="Use current network as opponent instead of past checkpoint")

    # Checkpointing
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--save-dir", type=str, default="models/alphazero",
                       help="Directory for saving checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=50,
                       help="Games between checkpoints (default: 50)")

    # Misc
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress")

    args = parser.parse_args()

    # Create config
    config = AlphaZeroConfig(
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        opponent_update_freq=args.opponent_update_freq,
        use_past_opponent=not args.no_past_opponent,
        checkpoint_freq=args.checkpoint_freq,
        save_dir=args.save_dir
    )

    # Create trainer
    trainer = AlphaZeroTrainer(config, verbose=args.verbose)

    # Load checkpoint if specified
    start_game = 0
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        start_game = trainer.total_games

    # Train
    trainer.train(args.episodes, start_game=start_game)


if __name__ == "__main__":
    main()
