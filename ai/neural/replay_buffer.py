"""
Experience Replay Buffer for DQN training.

Stores transitions (state, action, reward, next_state, done) and
provides random sampling for stable training. Breaking the correlation
between consecutive experiences is crucial for DQN stability.

Features:
- Circular buffer with configurable capacity
- Random batch sampling
- Priority replay support (optional)
- Serialization for checkpointing
"""
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pickle
from pathlib import Path

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class Experience:
    """
    Single transition experience for replay buffer.

    Attributes:
        state: Encoded state tensor (or raw state dict)
        action: Action index taken
        reward: Reward received
        next_state: Resulting state tensor
        done: Whether episode ended
        info: Optional additional information
    """
    state: any  # torch.Tensor or dict
    action: int
    reward: float
    next_state: any  # torch.Tensor or dict
    done: bool
    info: dict = None

    def __post_init__(self):
        if self.info is None:
            self.info = {}


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Implements a circular buffer that stores transitions and provides
    random batch sampling for training stability.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def push(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        info: dict = None
    ):
        """
        Add an experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
            info: Optional additional info
        """
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {}
        )
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of Experience objects
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def sample_tensors(
        self,
        batch_size: int
    ) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor',
               'torch.Tensor', 'torch.Tensor']:
        """
        Sample batch and return as tensors for training.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for tensor sampling")

        batch = self.sample(batch_size)

        # Stack tensors
        states = torch.stack([e.state for e in batch])
        actions = torch.tensor([e.action for e in batch], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in batch])
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= min_size

    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()

    def save(self, path: str):
        """Save buffer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to numpy for serialization
        save_data = []
        for exp in self.buffer:
            save_data.append({
                'state': exp.state.numpy() if hasattr(exp.state, 'numpy') else exp.state,
                'action': exp.action,
                'reward': exp.reward,
                'next_state': exp.next_state.numpy() if hasattr(exp.next_state, 'numpy') else exp.next_state,
                'done': exp.done,
                'info': exp.info
            })

        with open(path, 'wb') as f:
            pickle.dump({
                'capacity': self.capacity,
                'data': save_data
            }, f)

    def load(self, path: str):
        """Load buffer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.capacity = data['capacity']
        self.buffer = deque(maxlen=self.capacity)

        for exp_dict in data['data']:
            state = exp_dict['state']
            next_state = exp_dict['next_state']

            # Convert numpy back to tensor if available
            if TORCH_AVAILABLE:
                if hasattr(state, 'shape'):
                    state = torch.from_numpy(state).float()
                if hasattr(next_state, 'shape'):
                    next_state = torch.from_numpy(next_state).float()

            self.push(
                state=state,
                action=exp_dict['action'],
                reward=exp_dict['reward'],
                next_state=next_state,
                done=exp_dict['done'],
                info=exp_dict.get('info', {})
            )


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.

    Samples experiences based on TD-error priority, allowing the network
    to learn more from surprising (high error) transitions.

    Reference: Schaul et al., "Prioritized Experience Replay" (2015)
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight
            beta_frames: Frames over which to anneal beta
        """
        super().__init__(capacity)

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 0

        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        info: dict = None
    ):
        """Add experience with maximum priority."""
        super().push(state, action, reward, next_state, done, info)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], 'torch.Tensor']:
        """
        Sample batch based on priorities.

        Returns:
            Tuple of (experiences, indices, importance_weights)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        # Calculate sampling probabilities
        priorities = list(self.priorities)
        probs = [p ** self.alpha for p in priorities]
        total = sum(probs)
        probs = [p / total for p in probs]

        # Sample indices
        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        experiences = [self.buffer[i] for i in indices]

        # Calculate importance sampling weights
        beta = self._get_beta()
        weights = []
        max_weight = (len(self.buffer) * min(probs)) ** (-beta)

        for i in indices:
            weight = (len(self.buffer) * probs[i]) ** (-beta)
            weights.append(weight / max_weight)

        weights = torch.tensor(weights, dtype=torch.float32)

        return experiences, indices, weights

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            priority = abs(error) + 1e-6  # Small constant for stability
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def _get_beta(self) -> float:
        """Get current beta value for importance sampling."""
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def step_frame(self):
        """Increment frame counter for beta annealing."""
        self.frame += 1


class NStepReplayBuffer(ReplayBuffer):
    """
    N-step returns replay buffer.

    Instead of storing single-step transitions (s, a, r, s'), stores n-step
    transitions with accumulated discounted rewards:
        (s_t, a_t, R_n, s_{t+n})
    where R_n = r_t + γr_{t+1} + γ²r_{t+2} + ... + γⁿ⁻¹r_{t+n-1}

    This helps with credit assignment over multi-shot sequences in croquet,
    where the value of a roquet isn't immediate but unfolds over several shots.

    Reference: Sutton & Barto, "Reinforcement Learning" (n-step TD methods)
    """

    def __init__(
        self,
        capacity: int = 100000,
        n_step: int = 3,
        gamma: float = 0.99
    ):
        """
        Initialize n-step replay buffer.

        Args:
            capacity: Maximum buffer size
            n_step: Number of steps to look ahead (typically 3-5)
            gamma: Discount factor for future rewards
        """
        super().__init__(capacity)

        self.n_step = n_step
        self.gamma = gamma

        # Temporary buffer for accumulating n-step transitions
        self.n_step_buffer = deque(maxlen=n_step)

    def push(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        info: dict = None
    ):
        """
        Add experience to n-step buffer.

        The experience is held until we have n steps, then the accumulated
        return is computed and the transition is stored in the main buffer.
        """
        # Create single-step experience
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {}
        )

        self.n_step_buffer.append(experience)

        # If episode ended, flush remaining transitions
        if done:
            self._flush_n_step_buffer()
        # Otherwise, if we have n steps, compute and store
        elif len(self.n_step_buffer) == self.n_step:
            self._compute_and_store_n_step()

    def _compute_and_store_n_step(self):
        """
        Compute n-step return and store transition.

        Takes the oldest transition in the n-step buffer and computes:
        - Accumulated discounted reward over n steps
        - The state after n steps (or terminal state if episode ended)
        """
        if len(self.n_step_buffer) == 0:
            return

        # Get the oldest experience (the one we're storing)
        first_exp = self.n_step_buffer[0]

        # Compute n-step discounted return
        n_step_reward = 0.0
        gamma_power = 1.0
        final_next_state = first_exp.next_state
        final_done = first_exp.done

        for i, exp in enumerate(self.n_step_buffer):
            n_step_reward += gamma_power * exp.reward
            gamma_power *= self.gamma

            # Update final state (the state after n steps)
            final_next_state = exp.next_state
            final_done = exp.done

            # If we hit terminal state, stop accumulating
            if exp.done:
                break

        # Store the n-step transition
        n_step_exp = Experience(
            state=first_exp.state,
            action=first_exp.action,
            reward=n_step_reward,
            next_state=final_next_state,
            done=final_done,
            info={
                **first_exp.info,
                'n_step': len(self.n_step_buffer),
                'gamma_power': gamma_power  # For correct target computation
            }
        )
        self.buffer.append(n_step_exp)

    def _flush_n_step_buffer(self):
        """
        Flush remaining transitions when episode ends.

        Computes partial n-step returns for remaining transitions
        and adds them to the main buffer.
        """
        while len(self.n_step_buffer) > 0:
            self._compute_and_store_n_step()
            self.n_step_buffer.popleft()

    def reset_episode(self):
        """
        Reset the n-step buffer for a new episode.

        Call this at the start of each episode to ensure clean state.
        """
        self.n_step_buffer.clear()

    def get_n_step(self) -> int:
        """Return the n-step value for target computation."""
        return self.n_step

    def get_gamma(self) -> float:
        """Return gamma for target computation."""
        return self.gamma


class DemoMixingReplayBuffer(ReplayBuffer):
    """
    Replay buffer that guarantees a minimum fraction of demo transitions per batch.

    This implements DQfD-style (Deep Q-learning from Demonstrations) training where:
    - Demo transitions are stored separately and NEVER overwritten
    - Each sampled batch contains a guaranteed minimum fraction of demos
    - Regular RL transitions fill the rest of the batch

    This prevents catastrophic forgetting of pretrained behavior.
    """

    def __init__(
        self,
        capacity: int = 100000,
        demo_fraction: float = 0.25,
        demo_capacity: int = 25000
    ):
        """
        Initialize demo-mixing replay buffer.

        Args:
            capacity: Maximum size for RL transitions (demos don't count)
            demo_fraction: Minimum fraction of demos per batch (e.g., 0.25 = 25%)
            demo_capacity: Maximum number of demo transitions to store
        """
        super().__init__(capacity)

        self.demo_fraction = demo_fraction
        self.demo_capacity = demo_capacity

        # Separate protected buffer for demos (never overwritten by RL)
        self.demo_buffer = []
        self.demo_count = 0

    def push_demo(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        info: dict = None
    ):
        """
        Add a demonstration transition to the protected demo buffer.

        These transitions are NEVER overwritten by regular RL transitions.
        """
        if self.demo_count >= self.demo_capacity:
            return  # Demo buffer full, ignore (or could implement FIFO for demos)

        info = info or {}
        info['is_demo'] = True

        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info
        )
        self.demo_buffer.append(experience)
        self.demo_count += 1

    def push(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        info: dict = None
    ):
        """Add an RL transition (goes to regular buffer, can be overwritten)."""
        info = info or {}

        # Route demos to protected buffer
        if info.get('is_demo', False):
            self.push_demo(state, action, reward, next_state, done, info)
            return

        # Regular RL transitions go to standard circular buffer
        super().push(state, action, reward, next_state, done, info)

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample batch with guaranteed demo fraction.

        Returns batch containing at least demo_fraction demos (if available).
        """
        # Calculate demo and RL sample sizes
        num_demos_needed = int(batch_size * self.demo_fraction)
        num_rl_needed = batch_size - num_demos_needed

        # Sample demos (with replacement if needed)
        demo_samples = []
        if len(self.demo_buffer) > 0 and num_demos_needed > 0:
            if len(self.demo_buffer) >= num_demos_needed:
                demo_samples = random.sample(self.demo_buffer, num_demos_needed)
            else:
                # Not enough demos, sample with replacement
                demo_samples = random.choices(self.demo_buffer, k=num_demos_needed)

        # Sample RL transitions
        rl_samples = []
        if len(self.buffer) > 0 and num_rl_needed > 0:
            actual_rl = min(num_rl_needed, len(self.buffer))
            rl_samples = random.sample(list(self.buffer), actual_rl)

        # Combine and shuffle
        batch = demo_samples + rl_samples
        random.shuffle(batch)

        return batch

    def __len__(self) -> int:
        """Return total size (demos + RL)."""
        return len(self.buffer) + len(self.demo_buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples (counting both demo and RL)."""
        return len(self) >= min_size

    def get_demo_count(self) -> int:
        """Return number of demo transitions stored."""
        return self.demo_count

    def get_rl_count(self) -> int:
        """Return number of RL transitions stored."""
        return len(self.buffer)

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'demo_count': self.demo_count,
            'rl_count': len(self.buffer),
            'total': len(self),
            'demo_fraction': self.demo_fraction,
            'demo_capacity': self.demo_capacity
        }


class NStepPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    Combines n-step returns with prioritized experience replay.

    This is the recommended buffer for advanced DQN training:
    - N-step returns improve credit assignment
    - Prioritized replay focuses learning on surprising transitions
    """

    def __init__(
        self,
        capacity: int = 100000,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000
    ):
        """
        Initialize combined n-step + prioritized buffer.

        Args:
            capacity: Maximum buffer size
            n_step: Number of steps for n-step returns
            gamma: Discount factor
            alpha: Priority exponent
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight
            beta_frames: Frames for beta annealing
        """
        super().__init__(capacity, alpha, beta_start, beta_end, beta_frames)

        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def push(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        info: dict = None
    ):
        """Add experience with n-step accumulation."""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {}
        )

        self.n_step_buffer.append(experience)

        if done:
            self._flush_n_step_buffer()
        elif len(self.n_step_buffer) == self.n_step:
            self._compute_and_store_n_step()

    def _compute_and_store_n_step(self):
        """Compute n-step return and store with max priority."""
        if len(self.n_step_buffer) == 0:
            return

        first_exp = self.n_step_buffer[0]

        n_step_reward = 0.0
        gamma_power = 1.0
        final_next_state = first_exp.next_state
        final_done = first_exp.done

        for exp in self.n_step_buffer:
            n_step_reward += gamma_power * exp.reward
            gamma_power *= self.gamma
            final_next_state = exp.next_state
            final_done = exp.done
            if exp.done:
                break

        # Store in parent buffer with priority
        n_step_exp = Experience(
            state=first_exp.state,
            action=first_exp.action,
            reward=n_step_reward,
            next_state=final_next_state,
            done=final_done,
            info={
                **first_exp.info,
                'n_step': len(self.n_step_buffer),
                'gamma_power': gamma_power
            }
        )
        self.buffer.append(n_step_exp)
        self.priorities.append(self.max_priority)

    def _flush_n_step_buffer(self):
        """Flush remaining transitions when episode ends."""
        while len(self.n_step_buffer) > 0:
            self._compute_and_store_n_step()
            self.n_step_buffer.popleft()

    def reset_episode(self):
        """Reset n-step buffer for new episode."""
        self.n_step_buffer.clear()

    def get_n_step(self) -> int:
        """Return n-step value."""
        return self.n_step

    def get_gamma(self) -> float:
        """Return gamma."""
        return self.gamma
