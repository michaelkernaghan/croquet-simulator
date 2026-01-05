"""
Croquet Neural Network - Deep Q-Network for shot selection.

Architecture designed for croquet game state:
- Input: Encoded game state (ball positions, hoops, deadness, etc.)
- Hidden layers: Process spatial and tactical features
- Output: Q-values for each possible action type

The network learns to estimate Q(s,a) = expected future reward
for taking action a in state s.
"""
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide stub classes for when PyTorch isn't installed
    class nn:
        class Module:
            pass

from models.ball import Ball, Vector2
from models.court import Court


@dataclass
class EncodedState:
    """Encoded game state for neural network input."""
    tensor: 'torch.Tensor'  # The actual tensor
    ball_positions: Dict[str, Tuple[float, float]]  # For debugging
    striker_color: str
    target_hoop: int


class StateEncoder:
    """
    Encodes croquet game state into neural network input tensor.

    State representation (total ~80 features):
    - Ball positions (4 balls x 2 coords = 8)
    - Ball hoops run (4 balls = 4)
    - Ball has_pegged_out (4 balls = 4)
    - Target hoop position and direction (4)
    - Next hoop position and direction (4)
    - Deadness matrix (4x4 = 16 binary)
    - Striker info (one-hot 4 + hoops_run 1 = 5)
    - Distance to each ball from striker (4)
    - Distance to target hoop (1)
    - Strokes remaining (1)
    - Is continuation (1)
    - Court center relative positions (8)
    - Peg position relative (2)
    """

    # Feature dimensions
    NUM_BALLS = 4
    BALL_COLORS = ['blue', 'black', 'red', 'yellow']

    # Total features
    STATE_SIZE = 80

    def __init__(self, court: Court = None):
        """Initialize encoder with optional court reference."""
        self.court = court or Court()
        # Normalization constants
        self.court_width = self.court.width
        self.court_height = self.court.height
        self.max_distance = math.sqrt(self.court_width**2 + self.court_height**2)

    def encode(
        self,
        striker: Ball,
        balls: Dict[str, Ball],
        court: Court,
        deadness: Dict[str, set],
        strokes_remaining: int = 1,
        is_continuation: bool = False
    ) -> 'torch.Tensor':
        """
        Encode game state into tensor for neural network.

        Args:
            striker: The ball making the shot
            balls: All balls on court
            court: The court
            deadness: Which balls each ball is dead on
            strokes_remaining: Strokes left in turn
            is_continuation: Whether this is a continuation stroke

        Returns:
            Tensor of shape (STATE_SIZE,)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network features")

        features = []

        # 1. Ball positions (normalized to 0-1)
        for color in self.BALL_COLORS:
            if color in balls:
                ball = balls[color]
                features.append(ball.position.x / self.court_width)
                features.append(ball.position.y / self.court_height)
            else:
                features.extend([0.5, 0.5])  # Default center

        # 2. Ball hoops run (normalized 0-1, max 12 hoops)
        for color in self.BALL_COLORS:
            if color in balls:
                features.append(balls[color].hoops_run / 12.0)
            else:
                features.append(0.0)

        # 3. Ball has_pegged_out (binary)
        for color in self.BALL_COLORS:
            if color in balls:
                features.append(1.0 if balls[color].has_pegged_out else 0.0)
            else:
                features.append(0.0)

        # 4. Target hoop position and direction
        target_hoop = court.get_hoop_for_ball(striker.hoops_run)
        if target_hoop:
            features.append(target_hoop.position.x / self.court_width)
            features.append(target_hoop.position.y / self.court_height)
            features.append((target_hoop.direction.x + 1) / 2)  # Normalize -1,1 to 0,1
            features.append((target_hoop.direction.y + 1) / 2)
        else:
            features.extend([0.5, 0.5, 0.5, 0.5])

        # 5. Next hoop position and direction
        next_hoop = court.get_hoop_for_ball(striker.hoops_run + 1)
        if next_hoop:
            features.append(next_hoop.position.x / self.court_width)
            features.append(next_hoop.position.y / self.court_height)
            features.append((next_hoop.direction.x + 1) / 2)
            features.append((next_hoop.direction.y + 1) / 2)
        else:
            features.extend([0.5, 0.5, 0.5, 0.5])

        # 6. Deadness matrix (16 binary features)
        for striker_color in self.BALL_COLORS:
            dead_on = deadness.get(striker_color, set())
            for target_color in self.BALL_COLORS:
                features.append(1.0 if target_color in dead_on else 0.0)

        # 7. Striker info (one-hot + hoops)
        for color in self.BALL_COLORS:
            features.append(1.0 if color == striker.color else 0.0)
        features.append(striker.hoops_run / 12.0)

        # 8. Distance to each ball from striker (normalized)
        for color in self.BALL_COLORS:
            if color in balls and color != striker.color:
                dist = (balls[color].position - striker.position).magnitude()
                features.append(dist / self.max_distance)
            else:
                features.append(1.0)  # Max distance if same ball or missing

        # 9. Distance to target hoop
        if target_hoop:
            dist = (target_hoop.position - striker.position).magnitude()
            features.append(dist / self.max_distance)
        else:
            features.append(1.0)

        # 10. Strokes remaining (normalized, assume max 3)
        features.append(min(strokes_remaining, 3) / 3.0)

        # 11. Is continuation
        features.append(1.0 if is_continuation else 0.0)

        # 12. Court center relative positions for each ball
        center = Vector2(self.court_width / 2, self.court_height / 2)
        for color in self.BALL_COLORS:
            if color in balls:
                rel = balls[color].position - center
                features.append(rel.x / self.court_width + 0.5)
                features.append(rel.y / self.court_height + 0.5)
            else:
                features.extend([0.5, 0.5])

        # 13. Peg position relative to striker
        peg_rel = court.peg_position - striker.position
        features.append(peg_rel.x / self.court_width + 0.5)
        features.append(peg_rel.y / self.court_height + 0.5)

        # Pad to STATE_SIZE if needed
        while len(features) < self.STATE_SIZE:
            features.append(0.0)

        # Truncate if too many
        features = features[:self.STATE_SIZE]

        return torch.tensor(features, dtype=torch.float32)

    def get_state_size(self) -> int:
        """Return the size of encoded state."""
        return self.STATE_SIZE


class CroquetNet(nn.Module if TORCH_AVAILABLE else object):
    """
    Deep Q-Network for croquet shot selection.

    Architecture:
    - Input layer: STATE_SIZE features
    - Hidden layer 1: 256 units with ReLU + Dropout
    - Hidden layer 2: 128 units with ReLU + Dropout
    - Hidden layer 3: 64 units with ReLU
    - Output layer: NUM_ACTIONS Q-values

    Actions correspond to shot types:
    0: HOOP_RUN - Attempt to run the target hoop
    1: ROQUET_NEAREST - Roquet the nearest live ball
    2: ROQUET_PARTNER - Roquet partner ball
    3: ROQUET_OPPONENT1 - Roquet first opponent
    4: ROQUET_OPPONENT2 - Roquet second opponent
    5: APPROACH - Approach shot toward hoop
    6: DEFENSIVE - Defensive shot to boundary
    7: PEG_OUT - Peg out (if rover)
    """

    NUM_ACTIONS = 8

    ACTION_NAMES = [
        'HOOP_RUN',
        'ROQUET_NEAREST',
        'ROQUET_PARTNER',
        'ROQUET_OPPONENT1',
        'ROQUET_OPPONENT2',
        'APPROACH',
        'DEFENSIVE',
        'PEG_OUT'
    ]

    def __init__(
        self,
        state_size: int = StateEncoder.STATE_SIZE,
        hidden_sizes: List[int] = None,
        dropout: float = 0.2
    ):
        """
        Initialize the network.

        Args:
            state_size: Input feature size
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate for regularization
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network features. "
                            "Install with: pip install torch")

        super(CroquetNet, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.state_size = state_size
        self.hidden_sizes = hidden_sizes

        # Build network layers
        layers = []
        prev_size = state_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:  # No dropout on last hidden layer
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(prev_size, self.NUM_ACTIONS)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with Xavier initialization."""
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass through network.

        Args:
            state: Input state tensor of shape (batch_size, state_size)
                   or (state_size,) for single state

        Returns:
            Q-values tensor of shape (batch_size, NUM_ACTIONS)
        """
        # Handle single state (no batch dimension)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = self.hidden_layers(state)
        q_values = self.output_layer(x)

        return q_values

    def get_action(
        self,
        state: 'torch.Tensor',
        epsilon: float = 0.0,
        valid_actions: List[int] = None
    ) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state tensor
            epsilon: Exploration rate (0-1)
            valid_actions: List of valid action indices (None = all valid)

        Returns:
            Tuple of (action_index, q_value)
        """
        import random

        if valid_actions is None:
            valid_actions = list(range(self.NUM_ACTIONS))

        # Epsilon-greedy exploration
        if random.random() < epsilon:
            action = random.choice(valid_actions)
            with torch.no_grad():
                q_values = self.forward(state)
                q_value = q_values[0, action].item()
            return action, q_value

        # Greedy action selection
        with torch.no_grad():
            q_values = self.forward(state)

            # Mask invalid actions with very negative value
            mask = torch.full((self.NUM_ACTIONS,), float('-inf'))
            for a in valid_actions:
                mask[a] = 0

            masked_q = q_values + mask
            action = masked_q.argmax(dim=1).item()
            q_value = q_values[0, action].item()

        return action, q_value

    def save(self, path: str):
        """Save model weights to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'state_size': self.state_size,
            'hidden_sizes': self.hidden_sizes,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'CroquetNet':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            state_size=checkpoint['state_size'],
            hidden_sizes=checkpoint['hidden_sizes']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model


class DuelingCroquetNet(nn.Module):
    """
    Dueling DQN architecture for croquet shot selection.

    Separates the Q-value into:
    - V(s): State value - how good is this state?
    - A(s,a): Advantage - how much better is action a than average?

    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    This helps when many actions have similar values, allowing the network
    to learn the state value independently of action advantages.

    Reference: Wang et al., "Dueling Network Architectures for Deep RL" (2016)
    """

    NUM_ACTIONS = 8

    ACTION_NAMES = [
        'HOOP_RUN',
        'ROQUET_NEAREST',
        'ROQUET_PARTNER',
        'ROQUET_OPPONENT1',
        'ROQUET_OPPONENT2',
        'APPROACH',
        'DEFENSIVE',
        'PEG_OUT'
    ]

    def __init__(
        self,
        state_size: int = StateEncoder.STATE_SIZE,
        hidden_sizes: List[int] = None,
        dropout: float = 0.2
    ):
        """
        Initialize the dueling network.

        Args:
            state_size: Input feature size
            hidden_sizes: List of hidden layer sizes for shared layers
            dropout: Dropout rate for regularization
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network features. "
                            "Install with: pip install torch")

        super(DuelingCroquetNet, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]  # Shared layers (shorter for dueling)

        self.state_size = state_size
        self.hidden_sizes = hidden_sizes

        # Shared feature extraction layers
        shared_layers = []
        prev_size = state_size

        for i, hidden_size in enumerate(hidden_sizes):
            shared_layers.append(nn.Linear(prev_size, hidden_size))
            shared_layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:
                shared_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.shared_layers = nn.Sequential(*shared_layers)

        # Value stream: V(s) - single output
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream: A(s,a) - one output per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.NUM_ACTIONS)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with Xavier initialization."""
        for module in [self.shared_layers, self.value_stream, self.advantage_stream]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, state: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass through dueling network.

        Args:
            state: Input state tensor of shape (batch_size, state_size)
                   or (state_size,) for single state

        Returns:
            Q-values tensor of shape (batch_size, NUM_ACTIONS)
        """
        # Handle single state (no batch dimension)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Shared feature extraction
        features = self.shared_layers(state)

        # Compute value and advantage
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, NUM_ACTIONS)

        # Combine: Q = V + (A - mean(A))
        # Subtracting mean ensures identifiability and stability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_action(
        self,
        state: 'torch.Tensor',
        epsilon: float = 0.0,
        valid_actions: List[int] = None
    ) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state tensor
            epsilon: Exploration rate (0-1)
            valid_actions: List of valid action indices (None = all valid)

        Returns:
            Tuple of (action_index, q_value)
        """
        import random

        if valid_actions is None:
            valid_actions = list(range(self.NUM_ACTIONS))

        # Epsilon-greedy exploration
        if random.random() < epsilon:
            action = random.choice(valid_actions)
            with torch.no_grad():
                q_values = self.forward(state)
                q_value = q_values[0, action].item()
            return action, q_value

        # Greedy action selection
        with torch.no_grad():
            q_values = self.forward(state)

            # Mask invalid actions with very negative value
            mask = torch.full((self.NUM_ACTIONS,), float('-inf'))
            for a in valid_actions:
                mask[a] = 0

            masked_q = q_values + mask
            action = masked_q.argmax(dim=1).item()
            q_value = q_values[0, action].item()

        return action, q_value

    def save(self, path: str):
        """Save model weights to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'state_size': self.state_size,
            'hidden_sizes': self.hidden_sizes,
            'dueling': True,  # Mark as dueling architecture
        }, path)

    @classmethod
    def load(cls, path: str) -> 'DuelingCroquetNet':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            state_size=checkpoint['state_size'],
            hidden_sizes=checkpoint['hidden_sizes']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model


def check_torch_available() -> bool:
    """Check if PyTorch is available."""
    return TORCH_AVAILABLE


def get_device() -> 'torch.device':
    """Get the best available device (CUDA > MPS > CPU)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon
    else:
        return torch.device('cpu')
