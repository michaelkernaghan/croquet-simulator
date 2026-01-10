"""
DQN Trainer - Deep Q-Network training with target network.

Implements DQN training with:
- Target network for stable Q-value targets
- Experience replay for breaking correlation
- Epsilon-greedy exploration
- Gradient clipping for stability
- Learning rate scheduling

Reference: Mnih et al., "Human-level control through deep reinforcement learning" (2015)
"""
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .croquet_net import CroquetNet, DuelingCroquetNet, StateEncoder, get_device
from .replay_buffer import (
    ReplayBuffer, PrioritizedReplayBuffer,
    NStepReplayBuffer, NStepPrioritizedReplayBuffer,
    DemoMixingReplayBuffer
)


@dataclass
class TrainingConfig:
    """Configuration for DQN training."""
    # Network
    hidden_sizes: List[int] = None
    dropout: float = 0.2

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update rate for target network
    use_huber_loss: bool = True  # Use Huber loss instead of MSE (more stable)

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 200000  # Steps to decay epsilon (reach floor ~ep1000)

    # Replay buffer
    buffer_size: int = 100000
    min_buffer_size: int = 10000  # Min samples before training (warmup)
    use_prioritized: bool = False
    td_error_clip: float = 10.0  # Clip TD errors for PER stability

    # Advanced DQN options
    use_dueling: bool = False  # Use dueling network architecture
    n_step: int = 1  # N-step returns (1 = standard, 3-5 recommended)
    use_demo_mixing: bool = False  # Use DemoMixingReplayBuffer for DQfD-style training
    demo_fraction: float = 0.25  # Minimum fraction of demos per batch (if use_demo_mixing)

    # Updates
    target_update_freq: int = 100  # Steps between target updates
    train_freq: int = 4  # Steps between training updates
    gradient_clip: float = 10.0  # Increased from 1.0 per Peter's recommendation

    # Learning rate scheduling (two-stage decay per Peter's recommendation)
    lr_decay_after_epsilon: bool = True  # Decay LR as epsilon approaches floor
    lr_decay_factor: float = 0.5  # Factor to reduce LR by at each stage
    lr_decay_epsilon_threshold_1: float = 0.10  # First decay when ε ≤ this
    lr_decay_epsilon_threshold_2: float = 0.05  # Second decay when ε ≤ this

    # Checkpointing
    checkpoint_freq: int = 1000  # Steps between checkpoints
    save_dir: str = "ai_data/neural"

    # Reward shaping annealing (AlphaZero-style transition to sparse rewards)
    # When enabled, tactical shaping weight decays over training, shifting
    # from expert-guided learning toward pure outcome-based learning
    shaping_anneal: bool = False          # Enable annealing
    shaping_start: float = 1.0            # Initial shaping weight (full expert guidance)
    shaping_end: float = 0.1              # Final shaping weight (mostly sparse)
    shaping_anneal_steps: int = 500000    # Steps to anneal over

    # Self-play training (AlphaZero-style)
    # When enabled, both sides use the neural network instead of one side using heuristic
    self_play: bool = False               # Enable self-play mode
    self_play_opponent: str = "current"   # "current" = same network, "past" = past checkpoint
    opponent_update_freq: int = 100       # Games between opponent checkpoint updates (if "past")
    train_both_sides: bool = True         # Train on transitions from both sides (not just blue/black)

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 128, 64]


@dataclass
class TrainingStats:
    """Statistics from training."""
    total_steps: int = 0
    total_episodes: int = 0
    total_rewards: float = 0.0
    episode_rewards: List[float] = None
    episode_lengths: List[int] = None
    losses: List[float] = None
    q_values: List[float] = None
    epsilon_history: List[float] = None
    # Intent distribution tracking (per Peter's recommendation)
    intent_counts: Dict[int, int] = None  # action_index -> count
    # Action masking tracking (per Peter's recommendation)
    valid_action_counts: List[int] = None  # Number of valid actions per step
    intent_mask_counts: Dict[int, int] = None  # How often each intent was masked

    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.episode_lengths is None:
            self.episode_lengths = []
        if self.losses is None:
            self.losses = []
        if self.q_values is None:
            self.q_values = []
        if self.epsilon_history is None:
            self.epsilon_history = []
        if self.intent_counts is None:
            self.intent_counts = {i: 0 for i in range(8)}  # 8 actions
        if self.valid_action_counts is None:
            self.valid_action_counts = []
        if self.intent_mask_counts is None:
            self.intent_mask_counts = {i: 0 for i in range(8)}  # 8 actions

    def record_action(self, action: int, valid_actions: List[int] = None):
        """Record action selection and masking for tracking."""
        if action in self.intent_counts:
            self.intent_counts[action] += 1

        # Track valid action counts
        if valid_actions is not None:
            self.valid_action_counts.append(len(valid_actions))
            # Track which actions were masked (not in valid_actions)
            for i in range(8):
                if i not in valid_actions:
                    self.intent_mask_counts[i] += 1

    def get_intent_distribution(self) -> Dict[str, float]:
        """Get intent distribution as percentages."""
        total = sum(self.intent_counts.values())
        if total == 0:
            return {}
        intent_names = [
            'HOOP_RUN', 'ROQUET_NEAR', 'ROQUET_PARTNER',
            'ROQUET_OPP1', 'ROQUET_OPP2', 'APPROACH', 'DEFENSIVE', 'PEG_OUT'
        ]
        return {
            intent_names[i]: (self.intent_counts[i] / total) * 100
            for i in range(8)
        }

    def get_masking_stats(self) -> Dict:
        """Get action masking statistics."""
        total_steps = len(self.valid_action_counts)
        if total_steps == 0:
            return {'avg_valid_actions': 0, 'mask_rates': {}}

        avg_valid = sum(self.valid_action_counts) / total_steps
        intent_names = [
            'HOOP_RUN', 'ROQUET_NEAR', 'ROQUET_PARTNER',
            'ROQUET_OPP1', 'ROQUET_OPP2', 'APPROACH', 'DEFENSIVE', 'PEG_OUT'
        ]
        mask_rates = {
            intent_names[i]: (self.intent_mask_counts[i] / total_steps) * 100
            for i in range(8)
        }
        return {
            'avg_valid_actions': avg_valid,
            'mask_rates': mask_rates
        }

    def to_dict(self) -> dict:
        intent_dist = self.get_intent_distribution()
        masking = self.get_masking_stats()
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'total_rewards': self.total_rewards,
            'avg_episode_reward': sum(self.episode_rewards[-100:]) / max(1, len(self.episode_rewards[-100:])),
            'avg_episode_length': sum(self.episode_lengths[-100:]) / max(1, len(self.episode_lengths[-100:])),
            'avg_loss': sum(self.losses[-100:]) / max(1, len(self.losses[-100:])) if self.losses else 0,
            'avg_q_value': sum(self.q_values[-100:]) / max(1, len(self.q_values[-100:])) if self.q_values else 0,
            'current_epsilon': self.epsilon_history[-1] if self.epsilon_history else 1.0,
            'intent_distribution': intent_dist,
            'avg_valid_actions': masking['avg_valid_actions'],
        }


class DQNTrainer:
    """
    Deep Q-Network trainer for croquet AI.

    Uses:
    - Online network: Updated every training step
    - Target network: Updated slowly for stable Q-targets
    - Experience replay: Random sampling from buffer
    - Epsilon-greedy: Exploration during training
    """

    def __init__(self, config: TrainingConfig = None):
        """
        Initialize DQN trainer.

        Args:
            config: Training configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for DQN training. "
                            "Install with: pip install torch")

        self.config = config or TrainingConfig()
        self.device = get_device()
        print(f"Using device: {self.device}")

        # State encoder
        self.encoder = StateEncoder()

        # Select network architecture (standard or dueling)
        NetworkClass = DuelingCroquetNet if self.config.use_dueling else CroquetNet

        # Networks
        self.online_net = NetworkClass(
            state_size=self.encoder.get_state_size(),
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout
        ).to(self.device)

        self.target_net = NetworkClass(
            state_size=self.encoder.get_state_size(),
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout
        ).to(self.device)

        # Copy online weights to target
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network in eval mode

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=self.config.learning_rate
        )

        # Learning rate scheduler - step-based decay
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10000,
            gamma=0.95
        )

        # Track LR decay stages (two-stage decay per Peter's recommendation)
        self.lr_decay_stage = 0  # 0 = no decay, 1 = first decay, 2 = second decay

        # Huber loss function (smooth L1) for stability
        self.huber_loss = nn.SmoothL1Loss(reduction='none')

        # Select replay buffer based on config
        # Priority: demo_mixing > n-step+prioritized > n-step > prioritized > standard
        if self.config.use_demo_mixing:
            # DemoMixingReplayBuffer for DQfD-style training
            # Guarantees demo fraction per batch and protects demos from overwrite
            self.replay_buffer = DemoMixingReplayBuffer(
                capacity=self.config.buffer_size,
                demo_fraction=self.config.demo_fraction,
                demo_capacity=self.config.buffer_size // 4  # 25% reserved for demos
            )
            print(f"Using DemoMixingReplayBuffer (demo_fraction={self.config.demo_fraction:.0%})")
        elif self.config.n_step > 1:
            if self.config.use_prioritized:
                self.replay_buffer = NStepPrioritizedReplayBuffer(
                    capacity=self.config.buffer_size,
                    n_step=self.config.n_step,
                    gamma=self.config.gamma
                )
            else:
                self.replay_buffer = NStepReplayBuffer(
                    capacity=self.config.buffer_size,
                    n_step=self.config.n_step,
                    gamma=self.config.gamma
                )
        elif self.config.use_prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.config.buffer_size
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=self.config.buffer_size
            )

        # Training state
        self.stats = TrainingStats()
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        # Create save directory
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def get_epsilon(self) -> float:
        """Get current exploration rate."""
        progress = min(1.0, self.stats.total_steps / self.config.epsilon_decay)
        epsilon = self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )
        return epsilon

    def get_shaping_weight(self) -> float:
        """
        Get current reward shaping weight (AlphaZero-style annealing).

        When shaping_anneal is enabled, this decays from shaping_start to
        shaping_end over shaping_anneal_steps, allowing gradual transition
        from expert-guided rewards to sparse (outcome-based) rewards.

        Returns:
            Weight to multiply tactical shaping rewards by (0.0 to 1.0)
        """
        if not self.config.shaping_anneal:
            return 1.0  # No annealing - full shaping weight

        progress = min(1.0, self.stats.total_steps / self.config.shaping_anneal_steps)
        weight = self.config.shaping_start + progress * (
            self.config.shaping_end - self.config.shaping_start
        )
        return weight

    def select_action(
        self,
        state: 'torch.Tensor',
        valid_actions: List[int] = None,
        training: bool = True
    ) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state tensor
            valid_actions: List of valid action indices
            training: Whether in training mode (uses epsilon)

        Returns:
            Tuple of (action_index, q_value)
        """
        epsilon = self.get_epsilon() if training else 0.0
        state = state.to(self.device)

        action, q_value = self.online_net.get_action(state, epsilon, valid_actions)

        # Record action and masking for tracking
        if training:
            self.stats.record_action(action, valid_actions)

        return action, q_value

    def store_transition(
        self,
        state: 'torch.Tensor',
        action: int,
        reward: float,
        next_state: 'torch.Tensor',
        done: bool,
        info: dict = None
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(
            state.cpu(), action, reward, next_state.cpu(), done, info
        )

        # Update episode tracking
        self.current_episode_reward += reward
        self.current_episode_length += 1

        if done:
            self.stats.episode_rewards.append(self.current_episode_reward)
            self.stats.episode_lengths.append(self.current_episode_length)
            self.stats.total_episodes += 1
            self.stats.total_rewards += self.current_episode_reward
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        self.stats.total_steps += 1

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value if training occurred, None otherwise
        """
        # Check if ready to train
        if not self.replay_buffer.is_ready(self.config.min_buffer_size):
            return None

        # Only train every N steps
        if self.stats.total_steps % self.config.train_freq != 0:
            return None

        # Sample batch - always get raw experiences to access info dict
        if self.config.use_prioritized:
            experiences, indices, weights = self.replay_buffer.sample(
                self.config.batch_size
            )
            weights = weights.to(self.device)
        else:
            experiences = self.replay_buffer.sample(self.config.batch_size)
            indices = None
            weights = None

        # Convert experiences to tensors
        states = torch.stack([e.state for e in experiences]).to(self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([e.next_state for e in experiences]).to(self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32).to(self.device)

        # Extract next_valid_actions from info dicts (default to all actions if not present)
        next_valid_actions_list = [
            e.info.get('next_valid_actions', list(range(8))) if e.info else list(range(8))
            for e in experiences
        ]

        # Build next-state action mask [batch_size, num_actions]
        # Invalid actions get large negative value so they're never selected by argmax
        # Use -1e9 instead of -inf for numerical safety (avoids potential NaN issues)
        num_actions = 8
        NEG_MASK = -1e9
        next_action_mask = torch.full(
            (len(next_valid_actions_list), num_actions),
            NEG_MASK,
            device=self.device
        )
        empty_fallback_count = 0
        for i, valid_actions in enumerate(next_valid_actions_list):
            # Guard against empty valid_actions (edge case / bug)
            if len(valid_actions) == 0:
                # Fallback: allow all actions rather than masking everything
                next_action_mask[i, :] = 0.0
                empty_fallback_count += 1
            else:
                for a in valid_actions:
                    next_action_mask[i, a] = 0.0

        # Track empty fallback rate (should be ~0; if not, upstream legality bug)
        if not hasattr(self, '_empty_fallback_count'):
            self._empty_fallback_count = 0
        self._empty_fallback_count += empty_fallback_count

        # Compute current Q values
        current_q = self.online_net(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            # Double DQN: Use online net to select action, target net to evaluate
            # CRITICAL: Apply mask to prevent selecting invalid actions
            q_next_online = self.online_net(next_states)

            # Diagnostic: count how often masking changes the argmax (should be >0)
            unmasked_actions = q_next_online.argmax(1)
            q_next_online_masked = q_next_online + next_action_mask
            next_actions = q_next_online_masked.argmax(1)

            # Track mask effectiveness (how often mask changed the selection)
            mask_changed = (unmasked_actions != next_actions).sum().item()
            if not hasattr(self, '_mask_change_count'):
                self._mask_change_count = 0
                self._mask_total_count = 0
            self._mask_change_count += mask_changed
            self._mask_total_count += len(next_actions)

            next_q = self.target_net(next_states)
            next_q = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # For n-step returns, use γⁿ as the discount factor
            # The reward already contains accumulated discounted rewards r + γr' + γ²r'' + ...
            # So we just need to bootstrap with γⁿ * V(s_{t+n})
            # Note: dones is already float32 from tensor creation above
            if self.config.n_step > 1:
                gamma_n = self.config.gamma ** self.config.n_step
                target_q = rewards + (1.0 - dones) * gamma_n * next_q
            else:
                target_q = rewards + (1.0 - dones) * self.config.gamma * next_q

        # Compute loss - use Huber loss for stability (reduces impact of outliers)
        td_errors = target_q - current_q

        if self.config.use_huber_loss:
            # Huber loss (smooth L1) - less sensitive to outliers than MSE
            element_wise_loss = self.huber_loss(current_q, target_q)
        else:
            # Original MSE loss
            element_wise_loss = td_errors.pow(2)

        if weights is not None:
            loss = (weights * element_wise_loss).mean()
            # Update priorities with CLIPPED TD errors to prevent blow-ups
            clipped_td = torch.clamp(
                td_errors.abs(),
                max=self.config.td_error_clip
            )
            self.replay_buffer.update_priorities(
                indices, clipped_td.detach().cpu().numpy()
            )
        else:
            loss = element_wise_loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(
                self.online_net.parameters(),
                self.config.gradient_clip
            )

        self.optimizer.step()
        self.scheduler.step()

        # Two-stage LR decay as epsilon approaches floor (per Peter's recommendation)
        # This prevents instability when exploration stops
        if self.config.lr_decay_after_epsilon:
            epsilon = self.get_epsilon()

            # Stage 1: decay at epsilon <= 0.10 -> LR *= 0.5
            if (self.lr_decay_stage < 1 and
                epsilon <= self.config.lr_decay_epsilon_threshold_1):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.config.lr_decay_factor
                self.lr_decay_stage = 1
                new_lr = self.optimizer.param_groups[0]['lr']
                print(f"  [LR DECAY STAGE 1] eps={epsilon:.3f} <= {self.config.lr_decay_epsilon_threshold_1}, "
                      f"LR -> {new_lr:.2e}")

            # Stage 2: decay at epsilon <= 0.05 -> LR *= 0.5 (ending at ~2.5e-5)
            if (self.lr_decay_stage < 2 and
                epsilon <= self.config.lr_decay_epsilon_threshold_2):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.config.lr_decay_factor
                self.lr_decay_stage = 2
                new_lr = self.optimizer.param_groups[0]['lr']
                print(f"  [LR DECAY STAGE 2] eps={epsilon:.3f} <= {self.config.lr_decay_epsilon_threshold_2}, "
                      f"LR -> {new_lr:.2e}")

        # Update target network
        if self.stats.total_steps % self.config.target_update_freq == 0:
            self._soft_update_target()

        # Record stats
        loss_value = loss.item()
        self.stats.losses.append(loss_value)
        self.stats.q_values.append(current_q.mean().item())
        self.stats.epsilon_history.append(self.get_epsilon())

        # Checkpoint
        if self.stats.total_steps % self.config.checkpoint_freq == 0:
            self.save_checkpoint()

        return loss_value

    def _soft_update_target(self):
        """Soft update target network weights."""
        tau = self.config.tau
        for target_param, online_param in zip(
            self.target_net.parameters(),
            self.online_net.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1 - tau) * target_param.data
            )

    def save_checkpoint(self, suffix: str = "", eval_metrics: dict = None):
        """Save training checkpoint.

        Args:
            suffix: Suffix for checkpoint filename (e.g., "_ep1000")
            eval_metrics: Optional greedy eval metrics for plateau detection
        """
        checkpoint = {
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'stats': self.stats.to_dict(),
            'config': vars(self.config),
            'total_steps': self.stats.total_steps,
        }

        path = self.save_dir / f"checkpoint{suffix}.pt"
        torch.save(checkpoint, path)

        # Save stats as JSON for easy reading
        # Include eval metrics for plateau detection
        stats_dict = self.stats.to_dict()
        if eval_metrics:
            stats_dict.update(eval_metrics)

        stats_path = self.save_dir / f"stats{suffix}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)

    def load_checkpoint(self, path: str = None):
        """Load training checkpoint."""
        if path is None:
            path = self.save_dir / "checkpoint.pt"

        checkpoint = torch.load(path, map_location=self.device)

        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.stats.total_steps = checkpoint['total_steps']

    def pretrain_from_demos(
        self,
        demo_path: str,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        verbose: bool = True,
        post_pretrain_lr: float = None,
        post_pretrain_epsilon: float = None,
        keep_demos_in_buffer: bool = True,
        demo_priority_bonus: float = 1.0
    ) -> Dict:
        """
        Pretrain the network using behavior cloning from demonstrations.

        This trains the policy to predict the expert/heuristic actions
        before RL fine-tuning begins. The network learns basic croquet
        strategy from the TacticalDecisionMaker.

        Args:
            demo_path: Path to demos.pt file from collect_demos.py
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate for pretraining (typically higher than RL)
            verbose: Print progress
            post_pretrain_lr: Learning rate to use after pretraining (prevents
                catastrophic forgetting by using lower LR during RL). If None,
                uses config.learning_rate (default).
            post_pretrain_epsilon: Starting epsilon after pretraining. Lower value
                (e.g., 0.3) preserves pretrained policy better by reducing exploration.
                If None, uses config.epsilon_start (default 1.0).
            keep_demos_in_buffer: If True, adds demo transitions to replay buffer
                so they're mixed into RL training batches (DQfD-style).
            demo_priority_bonus: Priority multiplier for demo transitions in buffer.
                Higher values make demos sampled more frequently.

        Returns:
            Dict with pretraining statistics
        """
        if verbose:
            print(f"\n{'='*60}")
            print("BEHAVIOR CLONING PRETRAINING")
            print(f"{'='*60}")

        # Load demonstrations
        demos = torch.load(demo_path, map_location=self.device)
        states = demos['states'].to(self.device)
        valid_masks = demos['valid_masks'].to(self.device)
        expert_actions = demos['expert_actions'].to(self.device)
        num_demos = demos['num_demos']

        if verbose:
            print(f"Loaded {num_demos} demonstrations from {demo_path}")
            print(f"Action distribution: {demos.get('action_distribution', {})}")
            print(f"Training for {epochs} epochs with batch_size={batch_size}, lr={lr}")
            print()

        # Create optimizer just for pretraining (separate from RL optimizer)
        pretrain_optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        ce_loss_fn = nn.CrossEntropyLoss(reduction='none')

        # Training loop
        num_batches = (num_demos + batch_size - 1) // batch_size
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0

            # Shuffle indices
            indices = torch.randperm(num_demos, device=self.device)

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_demos)
                batch_indices = indices[start:end]

                # Get batch
                batch_states = states[batch_indices]
                batch_masks = valid_masks[batch_indices]
                batch_actions = expert_actions[batch_indices]

                # Forward pass
                q_values = self.online_net(batch_states)

                # Apply action mask: invalid actions get large negative logits
                # This ensures we only train on valid action predictions
                masked_logits = q_values.clone()
                masked_logits[~batch_masks] = -1e9

                # Compute cross-entropy loss (predict expert action)
                loss = ce_loss_fn(masked_logits, batch_actions).mean()

                # Backward pass
                pretrain_optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.online_net.parameters(),
                    self.config.gradient_clip
                )

                pretrain_optimizer.step()

                # Track accuracy
                predictions = masked_logits.argmax(dim=1)
                correct = (predictions == batch_actions).sum().item()

                epoch_loss += loss.item() * len(batch_indices)
                epoch_correct += correct

            # Epoch stats
            epoch_loss /= num_demos
            epoch_acc = epoch_correct / num_demos * 100

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += num_demos

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.1f}%")

        # Sync target network after pretraining
        self.target_net.load_state_dict(self.online_net.state_dict())

        # Final stats
        avg_loss = total_loss / epochs
        final_acc = total_correct / total_samples * 100

        # =====================================================
        # POST-PRETRAIN ADJUSTMENTS (prevent catastrophic forgetting)
        # =====================================================

        # 1. Lower learning rate after pretrain to preserve weights
        if post_pretrain_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = post_pretrain_lr
            if verbose:
                print(f"  Post-pretrain LR set to: {post_pretrain_lr:.2e}")

        # 2. Lower starting epsilon to reduce exploration (policy is already good)
        if post_pretrain_epsilon is not None:
            # Override epsilon by adjusting the step count
            # epsilon = start + (end - start) * progress
            # We want epsilon = post_pretrain_epsilon at current step
            # Solve: post_pretrain_epsilon = 1.0 + (0.05 - 1.0) * (steps / decay)
            # steps = decay * (post_pretrain_epsilon - 1.0) / (0.05 - 1.0)
            epsilon_range = self.config.epsilon_end - self.config.epsilon_start
            if epsilon_range != 0:
                target_progress = (post_pretrain_epsilon - self.config.epsilon_start) / epsilon_range
                self.stats.total_steps = int(target_progress * self.config.epsilon_decay)
                if verbose:
                    print(f"  Post-pretrain epsilon set to: {post_pretrain_epsilon:.2f} (via step offset)")

        # 3. Add demos to replay buffer for DQfD-style training
        demos_added = 0
        if keep_demos_in_buffer:
            # Create synthetic transitions from demos
            # We need (state, action, reward, next_state, done) but demos only have (state, action)
            # Use small positive reward for expert actions, dummy next_state

            # Determine capacity based on buffer type
            if isinstance(self.replay_buffer, DemoMixingReplayBuffer):
                max_demos = self.replay_buffer.demo_capacity
            else:
                max_demos = self.config.buffer_size // 4  # 25% of buffer

            for i in range(min(num_demos, max_demos)):
                demo_state = states[i].cpu()
                demo_action = expert_actions[i].item()

                # Expert action gets small positive reward to encourage imitation
                demo_reward = 1.0  # Small positive reward for expert actions

                # Dummy next state (same as current - will be bootstrapped)
                demo_next_state = demo_state.clone()
                demo_done = False  # Assume not terminal

                # Store with info indicating it's a demo (for potential priority boost)
                demo_info = {
                    'is_demo': True,
                    'next_valid_actions': list(range(8))  # All actions valid for dummy
                }

                # Use push_demo for DemoMixingReplayBuffer, regular push otherwise
                if isinstance(self.replay_buffer, DemoMixingReplayBuffer):
                    self.replay_buffer.push_demo(
                        demo_state, demo_action, demo_reward,
                        demo_next_state, demo_done, demo_info
                    )
                else:
                    self.replay_buffer.push(
                        demo_state, demo_action, demo_reward,
                        demo_next_state, demo_done, demo_info
                    )
                demos_added += 1

            if verbose:
                print(f"  Added {demos_added} demo transitions to replay buffer")
                if isinstance(self.replay_buffer, DemoMixingReplayBuffer):
                    print(f"  Demos protected from overwrite (DemoMixingReplayBuffer)")

        if verbose:
            print()
            print(f"Pretraining complete!")
            print(f"  Final accuracy: {final_acc:.1f}%")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"{'='*60}\n")

        return {
            'epochs': epochs,
            'final_accuracy': final_acc,
            'average_loss': avg_loss,
            'num_demos': num_demos,
            'demos_in_buffer': demos_added,
            'post_pretrain_lr': post_pretrain_lr,
            'post_pretrain_epsilon': post_pretrain_epsilon
        }

    def get_model(self) -> CroquetNet:
        """Get the trained online network."""
        return self.online_net

    def get_stats_summary(self) -> str:
        """Get training statistics summary."""
        stats = self.stats.to_dict()
        summary = (
            f"Steps: {stats['total_steps']} | "
            f"Episodes: {self.stats.total_episodes} | "
            f"Avg Reward: {stats['avg_episode_reward']:.2f} | "
            f"Avg Loss: {stats['avg_loss']:.4f} | "
            f"Avg Q: {stats['avg_q_value']:.2f} | "
            f"Epsilon: {stats['current_epsilon']:.3f}"
        )
        # Add mask effectiveness diagnostic
        if hasattr(self, '_mask_total_count') and self._mask_total_count > 0:
            mask_pct = 100.0 * self._mask_change_count / self._mask_total_count
            summary += f" | Mask-changed: {mask_pct:.1f}%"
        return summary

    def get_mask_stats(self) -> dict:
        """Get action mask effectiveness statistics."""
        if not hasattr(self, '_mask_total_count') or self._mask_total_count == 0:
            return {'mask_change_pct': 0.0, 'mask_changes': 0, 'mask_total': 0, 'empty_fallback_pct': 0.0}
        empty_count = getattr(self, '_empty_fallback_count', 0)
        return {
            'mask_change_pct': 100.0 * self._mask_change_count / self._mask_total_count,
            'mask_changes': self._mask_change_count,
            'mask_total': self._mask_total_count,
            'empty_fallback_pct': 100.0 * empty_count / self._mask_total_count,
            'empty_fallback_count': empty_count
        }


def create_reward_function(use_expert_shaping: bool = False) -> Callable:
    """
    Create reward function for croquet training.

    Basic rewards:
    - Running a hoop: +10
    - Roqueting a ball: +2
    - Making a good approach: +1
    - Hitting peg (when rover): +20
    - Turn ending without progress: -1
    - Missing a shot: -0.5

    If use_expert_shaping=True, adds comprehensive expert tactical bonuses from:
    - Aiton's approach quality and break building principles
    - Wylie's Expert Croquet Tactics
    - Oxford Croquet Rules of Thumb
    - Defensive play tactics
    - MacRobertson Shield tournament analysis

    Key tactical principles encoded:
    1. APPROACH: 1 yard from hoop with good angle is ideal (Aiton)
    2. BREAK BUILDING: 4-ball break is preferred; pioneers placed via croquet (Rules of Thumb)
    3. RUSH: Rush is key to break building; avoid fine cut rushes (Wylie)
    4. LEAVE QUALITY: Good leaves win games; boundaries are defensive (Oxford)
    5. POSITIONAL: Don't join in middle; stay near boundaries (Defensive Play)
    6. SHOT SELECTION: Short strokes > long; narrow croquet > wide (Rules of Thumb)
    """
    def reward_fn(
        hoop_run: bool,
        roqueted: bool,
        good_approach: bool,
        pegged_out: bool,
        turn_ended: bool,
        shot_hit: bool,
        # Expert tactical context (optional)
        approach_quality: str = None,  # 'excellent', 'good', 'fair', 'poor'
        approach_distance: float = None,  # Distance to hoop in yards
        approach_angle: float = None,  # Angle quality 0-1
        has_rush: bool = False,
        rush_quality: str = None,  # 'straight', 'slight_cut', 'fine_cut'
        rush_distance: float = None,  # Distance of rush
        break_balls: int = 0,  # Number of balls in break position
        has_pioneer_at_next: bool = False,
        has_pioneer_at_next_but_one: bool = False,
        pioneer_placed_via_croquet: bool = False,  # Rules of Thumb: croquet > rush for pioneer
        pivot_position_quality: str = None,  # 'good', 'near_peg', 'outside_hoops'
        # Leave/positional quality
        good_leave: bool = False,
        leave_type: str = None,  # 'nsl', 'osl', 'defensive', 'wide_join'
        at_boundary: bool = False,  # At boundary = defensive = good (usually)
        in_middle_of_court: bool = False,  # Middle = bad (Rules of Thumb)
        near_opponent_hoop: bool = False,  # Bad - gives them pioneer
        balls_wired: bool = False,  # Good defensive position
        partner_ball_exposed: bool = False,  # Bad - gives double target
        # Shot quality
        stroke_length: str = None,  # 'short', 'medium', 'long'
        croquet_angle: str = None,  # 'narrow', 'medium', 'wide'
        rushed_pioneer_to_position: bool = False,  # Sin per Rules of Thumb
        # Turn continuation
        turn_continues: bool = False,
        consecutive_hoops: int = 0,  # Hoops run this turn
        # Opponent context
        opponent_gave_easy_break: bool = False,  # We got 4-ball position from opponent
        # Time/efficiency penalty
        steps_since_last_hoop: int = 0,  # Steps without scoring a hoop
        # Rover/endgame state
        is_rover: bool = False,  # Ball has run all 12 hoops
        steps_as_rover: int = 0,  # Steps since becoming rover without pegging out
        # Action tracking
        chose_defensive: bool = False,  # Did the model choose DEFENSIVE action?
        # DELTA-BASED PARAMETERS (per Peter's feedback)
        # Reward CREATING a position, not maintaining it
        prev_break_balls: int = 0,  # Break balls before this shot
        prev_has_pioneer_at_next: bool = False,  # Pioneer status before shot
        prev_has_pioneer_at_next_but_one: bool = False,
        # NEW TACTICAL DELTA PARAMETERS
        has_pilot: bool = False,  # Pilot at current hoop
        prev_has_pilot: bool = False,
        has_rush_to_hoop: bool = False,  # Rush available to next hoop
        prev_has_rush_to_hoop: bool = False,
        cluster_quality: float = 0.0,  # 0-1, how tight the ball cluster is
        prev_cluster_quality: float = 0.0,
        opponent_separation: float = 0.0,  # Distance between opponent balls (yards)
        prev_opponent_separation: float = 0.0,
        # ONCE-PER-TURN CAPS (per Peter's calibration feedback)
        turn_tactical_awarded: dict = None,  # Tracks what's been awarded this turn
        turn_loss_counts: dict = None,  # Tracks loss penalty counts this turn
    ) -> tuple:
        """
        Returns (base_reward, tactical_reward, tactical_awards_dict, loss_increments_dict) where:
        - base_reward: Core game rewards (hoops, pegs) that should NOT be capped
        - tactical_reward: Shaping rewards that CAN be capped by per-turn budget
        - tactical_awards_dict indicates which features were awarded this step
        - loss_increments_dict indicates which loss penalties were applied (for capping)
        """
        base_reward = 0.0  # Hoops, peg - never capped
        tactical_reward = 0.0  # Shaping rewards - can be capped
        tactical_awards = {
            'pilot_created': False,
            'rush_created': False,
            'pioneer_next_created': False,
            'pioneer_next_but_one_created': False
        }
        loss_increments = {
            'pilot_lost': 0,
            'rush_lost': 0,
            'pioneer_next_lost': 0,
            'pioneer_next_but_one_lost': 0
        }
        if turn_tactical_awarded is None:
            turn_tactical_awarded = tactical_awards.copy()
        if turn_loss_counts is None:
            turn_loss_counts = loss_increments.copy()

        # Max loss penalties per feature per turn (prevents over-punishment during repositioning)
        MAX_LOSS_PER_TURN = 2

        # ===========================================
        # BASE REWARDS - Core game objectives (NEVER CAPPED)
        # RESCALED per Peter's feedback: narrower range for stability
        # Old: +200/+50 with wide spread. New: +20/+10 with tighter ratios
        # ===========================================
        if pegged_out:
            base_reward += 20.0  # Winning action (rescaled from 200)
        elif hoop_run:
            base_reward += 10.0  # Base hoop reward (rescaled from 50)
            # SUBLINEAR streak bonus (sqrt) to prevent farming behavior
            # sqrt grows slower: 2nd=+1.4, 3rd=+1.7, 4th=+2.0, 6th=+2.4
            if consecutive_hoops >= 2:
                import math
                streak_bonus = 3.0 * math.sqrt(consecutive_hoops - 1)
                base_reward += min(streak_bonus, 8.0)  # Cap at +8 for very long streaks
        elif roqueted:
            # Roquet value depends on break context
            if break_balls >= 4:
                base_reward += 1.5  # More valuable in 4-ball break context
            elif break_balls >= 3:
                base_reward += 1.0
            else:
                base_reward += 0.5  # Base roquet - means to end, not the goal
        elif good_approach:
            base_reward += 0.3

        # ===========================================
        # BREAK BUILDING REWARDS (DELTA-BASED per Peter's feedback)
        # These are TACTICAL SHAPING - can be capped by per-turn budget
        # Reward CREATING/IMPROVING position, not maintaining it.
        # This prevents the agent from farming rewards by sitting still.
        # ===========================================
        # Reward improving break position (delta-based)
        break_ball_delta = break_balls - prev_break_balls
        if break_ball_delta > 0:
            # Gained break balls - reward the improvement
            if break_balls >= 4:
                tactical_reward += 2.0  # Achieved full 4-ball break
            elif break_balls == 3:
                tactical_reward += 1.0  # Achieved 3-ball break
            else:
                tactical_reward += 0.5 * break_ball_delta  # Incremental improvement
        elif break_ball_delta < 0:
            # Lost break balls - small penalty
            tactical_reward += 0.3 * break_ball_delta  # Negative delta = penalty

        # Pioneer placement - DELTA-BASED with ONCE-PER-TURN CAP
        # Only reward if we GAINED a pioneer we didn't have before AND haven't awarded this turn
        if has_pioneer_at_next and not prev_has_pioneer_at_next:
            if not turn_tactical_awarded.get('pioneer_next_created', False):
                tactical_reward += 1.5  # Created pioneer at next hoop
                tactical_awards['pioneer_next_created'] = True
        if has_pioneer_at_next_but_one and not prev_has_pioneer_at_next_but_one:
            if not turn_tactical_awarded.get('pioneer_next_but_one_created', False):
                tactical_reward += 0.75  # Created pioneer at next-but-one
                tactical_awards['pioneer_next_but_one_created'] = True

        # Small penalty for LOSING a pioneer (wasted setup) - CAPPED per turn
        if prev_has_pioneer_at_next and not has_pioneer_at_next:
            if turn_loss_counts.get('pioneer_next_lost', 0) < MAX_LOSS_PER_TURN:
                tactical_reward -= 0.5  # Lost pioneer at next
                loss_increments['pioneer_next_lost'] = 1
        if prev_has_pioneer_at_next_but_one and not has_pioneer_at_next_but_one:
            if turn_loss_counts.get('pioneer_next_but_one_lost', 0) < MAX_LOSS_PER_TURN:
                tactical_reward -= 0.25  # Lost pioneer at next-but-one
                loss_increments['pioneer_next_but_one_lost'] = 1

        # ===========================================
        # NEW TACTICAL DELTA REWARDS (per Peter's recommendation)
        # These are TACTICAL SHAPING - can be capped by per-turn budget
        # Reward CREATING tactical positions, not maintaining them
        # With ONCE-PER-TURN CAPS to prevent farming
        # Loss penalties also capped to avoid punishing necessary repositioning
        # ===========================================

        # Pilot created: ball positioned at current hoop for approach
        if has_pilot and not prev_has_pilot:
            if not turn_tactical_awarded.get('pilot_created', False):
                tactical_reward += 1.0  # Created pilot at current hoop
                tactical_awards['pilot_created'] = True
        if prev_has_pilot and not has_pilot:
            if turn_loss_counts.get('pilot_lost', 0) < MAX_LOSS_PER_TURN:
                tactical_reward -= 0.3  # Lost pilot (used it or misplaced)
                loss_increments['pilot_lost'] = 1

        # Rush availability: two balls together pointing to next hoop
        if has_rush_to_hoop and not prev_has_rush_to_hoop:
            if not turn_tactical_awarded.get('rush_created', False):
                tactical_reward += 1.2  # Created rush to next hoop - very valuable
                tactical_awards['rush_created'] = True
        if prev_has_rush_to_hoop and not has_rush_to_hoop:
            if turn_loss_counts.get('rush_lost', 0) < MAX_LOSS_PER_TURN:
                tactical_reward -= 0.4  # Lost rush opportunity
                loss_increments['rush_lost'] = 1

        # Cluster quality improvement (delta-based, continuous)
        # No cap needed - continuous values naturally don't farm
        cluster_delta = cluster_quality - prev_cluster_quality
        if cluster_delta > 0.1:
            tactical_reward += 0.8 * cluster_delta  # Reward tightening the cluster
        elif cluster_delta < -0.1:
            tactical_reward += 0.4 * cluster_delta  # Penalty for scattering (negative delta)

        # Opponent separation improvement (defense)
        # Gate by game phase: only reward when NOT in a strong break position
        # If cluster_quality is high (we have control), don't distract with separation rewards
        if cluster_quality < 0.6:  # Not in full control - defense matters
            separation_delta = opponent_separation - prev_opponent_separation
            if separation_delta > 2.0:
                tactical_reward += 0.5  # Separated opponents significantly
            elif separation_delta < -3.0:
                tactical_reward -= 0.3  # Let opponents get together (bad)

        # Base penalties (these are BASE rewards, not tactical shaping)
        if turn_ended and base_reward == 0 and tactical_reward <= 0:
            base_reward -= 2.0  # Penalty for ending turn with nothing

        if not shot_hit and not hoop_run:
            base_reward -= 1.5  # Increased miss penalty (was -1.0)

        # ===========================================
        # TIME/EFFICIENCY PENALTY (BASE - not tactical)
        # Encourage faster hoop progress - penalize slow play
        # ===========================================
        if steps_since_last_hoop > 10:
            # Gradual penalty that increases the longer we go without scoring
            time_penalty = 0.2 * (steps_since_last_hoop - 10)  # Faster penalty growth
            base_reward -= min(time_penalty, 5.0)  # Higher cap at -5 per step

        # ===========================================
        # ROVER PENALTY (BASE - not tactical)
        # Rovers MUST peg out - heavily penalize dawdling
        # ===========================================
        if is_rover and not pegged_out:
            # Strong penalty that grows quickly - rovers should peg out ASAP
            if steps_as_rover > 3:
                rover_penalty = 1.0 * (steps_as_rover - 3)  # -1 per step after 3
                base_reward -= min(rover_penalty, 10.0)  # Cap at -10 per step

        # ===========================================
        # DEFENSIVE ACTION PENALTY (Tactical shaping)
        # Discourage overuse of defensive play - it's passive
        # ===========================================
        if chose_defensive:
            tactical_reward -= 0.5  # Small penalty for playing defensively (was -2.0, too harsh)

        # ===========================================
        # EXPERT TACTICAL SHAPING (all tactical - can be capped)
        # ===========================================
        if use_expert_shaping:
            # -------------------------------------------
            # 1. APPROACH QUALITY (Aiton + Rules of Thumb)
            # "1 yard from hoop with good angle is ideal"
            # -------------------------------------------
            if approach_quality == 'excellent':
                tactical_reward += 3.0
            elif approach_quality == 'good':
                tactical_reward += 1.5
            elif approach_quality == 'fair':
                tactical_reward += 0.5
            elif approach_quality == 'poor':
                tactical_reward -= 0.5

            # Distance-based approach bonus (1 yard = ~0.9m is ideal)
            # Only applies if approach_distance is provided (i.e., on correct side)
            if approach_distance is not None:
                if 0.7 <= approach_distance <= 1.2:  # Ideal range
                    tactical_reward += 2.0
                elif 0.5 <= approach_distance <= 2.0:  # Good range
                    tactical_reward += 1.0
                elif approach_distance > 4.0:  # Too far
                    tactical_reward -= 1.0

            # Approach angle bonus/penalty (0 = wrong side, 1 = perfect approach line)
            if approach_angle is not None:
                if approach_angle < 0.3:
                    # On wrong side of hoop - significant penalty
                    tactical_reward -= 2.0
                elif approach_angle > 0.7:
                    # Great approach angle
                    tactical_reward += 1.0

            # -------------------------------------------
            # 2. RUSH QUALITY (Wylie + Rules of Thumb)
            # "Avoid very fine cut rushes - they're very difficult"
            # "A near-straight rush is tolerant of minor deviations"
            # -------------------------------------------
            if has_rush:
                tactical_reward += 1.5  # Base rush bonus
                if rush_quality == 'straight':
                    tactical_reward += 1.5  # Straight rush is best
                elif rush_quality == 'slight_cut':
                    tactical_reward += 0.5
                elif rush_quality == 'fine_cut':
                    tactical_reward -= 1.0  # Fine cuts are dangerous

                # Short rushes are more accurate
                if rush_distance is not None and rush_distance < 3.0:
                    tactical_reward += 0.5

            # -------------------------------------------
            # 3. BREAK BUILDING (Aiton + Rules of Thumb)
            # "4-ball break is much more forgiving"
            # "Two balls together equals a rush anywhere"
            # -------------------------------------------
            if break_balls >= 4:
                tactical_reward += 3.0  # Full 4-ball break
            elif break_balls == 3:
                tactical_reward += 2.0
            elif break_balls == 2:
                tactical_reward += 1.0

            # Pioneer placement bonuses
            if has_pioneer_at_next:
                tactical_reward += 1.0
            if has_pioneer_at_next_but_one:
                tactical_reward += 0.5

            # Rules of Thumb: "Never rush a pioneer to your next-but-1 hoop"
            # "Always contrive to croquet pioneers into position"
            if pioneer_placed_via_croquet:
                tactical_reward += 0.5  # Correct technique
            if rushed_pioneer_to_position:
                tactical_reward -= 1.5  # "Rushing pioneers is a sin"

            # Pivot position (Rules of Thumb: "Don't put pivot near peg")
            if pivot_position_quality == 'good':
                tactical_reward += 0.5
            elif pivot_position_quality == 'near_peg':
                tactical_reward -= 1.0  # "Inaccessible from large area of court"
            elif pivot_position_quality == 'outside_hoops':
                tactical_reward -= 0.5

            # -------------------------------------------
            # 4. SHOT QUALITY (Rules of Thumb)
            # "Short strokes are easier and more accurate"
            # "Narrow croquet strokes are more successful"
            # -------------------------------------------
            if stroke_length == 'short':
                tactical_reward += 0.5
            elif stroke_length == 'long':
                tactical_reward -= 0.5

            if croquet_angle == 'narrow':
                tactical_reward += 0.5
            elif croquet_angle == 'wide':
                tactical_reward -= 0.5

            # -------------------------------------------
            # 5. LEAVE AND POSITION QUALITY
            # (Oxford Croquet Defensive Play + Rules of Thumb)
            # "Boundaries are defensive"
            # "Don't join up in middle of court"
            # "Don't join up near opponent's hoops"
            # -------------------------------------------
            if turn_ended:
                # Good leave types
                if good_leave:
                    tactical_reward += 2.0
                if leave_type in ['nsl', 'osl']:  # Standard leaves
                    tactical_reward += 1.5
                elif leave_type == 'wide_join':
                    tactical_reward += 1.0
                elif leave_type == 'defensive':
                    tactical_reward += 0.5

                # Position quality at end of turn
                # "Boundaries are defensive" - but context matters
                if at_boundary and not in_middle_of_court:
                    tactical_reward += 1.0  # Good defensive position

                if in_middle_of_court:
                    tactical_reward -= 2.0  # "Don't join in middle - opponent sails to safety"

                if near_opponent_hoop:
                    tactical_reward -= 1.5  # "Don't join up at their hoops - gives them pioneer"

                if balls_wired:
                    tactical_reward += 1.5  # Good defensive wiring

                if partner_ball_exposed:
                    tactical_reward -= 1.0  # "Don't leave a pawnbrokers (double target)"

            # During turn - positional rewards
            else:
                # Middle of turn, being at boundary with continuation is OK
                if at_boundary and turn_continues:
                    tactical_reward += 0.3  # Maintaining control at boundary

            # -------------------------------------------
            # 6. EXPLOITING OPPONENT MISTAKES
            # -------------------------------------------
            if opponent_gave_easy_break:
                tactical_reward += 1.0  # Reward for capitalizing on opponent error

        return base_reward, tactical_reward, tactical_awards, loss_increments

    return reward_fn


# Backward compatibility alias
def create_reward_function_legacy(use_aiton_shaping: bool = False) -> Callable:
    """Legacy wrapper - use_aiton_shaping now maps to use_expert_shaping."""
    return create_reward_function(use_expert_shaping=use_aiton_shaping)
