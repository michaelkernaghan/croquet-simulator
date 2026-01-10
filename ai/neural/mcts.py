"""
Monte Carlo Tree Search (MCTS) for croquet planning.

AlphaZero-style MCTS that uses a neural network for:
- Prior policy: guides initial exploration via action probabilities
- Value estimation: evaluates leaf nodes without requiring full rollouts

This module implements:
- MCTSConfig: Configuration parameters for tree search
- MCTSNode: Individual nodes in the search tree
- MCTS: Main search algorithm combining UCB selection with neural guidance
- CroquetSimulator: Lightweight game state simulator for MCTS rollouts

Key differences from pure MCTS:
- No random rollouts - uses neural value estimation instead
- Policy prior from network guides exploration
- Dirichlet noise at root for exploration during training
"""
import math
import copy
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search."""

    # Search parameters
    num_simulations: int = 50        # Simulations per move (AlphaZero uses 800)
    c_puct: float = 1.5              # Exploration constant (higher = more exploration)

    # Root exploration (training only)
    dirichlet_alpha: float = 0.3     # Dirichlet noise parameter
    root_noise_frac: float = 0.25    # Fraction of noise vs prior at root

    # Temperature for action selection
    temperature: float = 1.0         # 1.0 = proportional to visits, 0 = greedy
    temperature_threshold: int = 30  # Steps after which temperature drops to 0

    # Value estimation
    use_value_head: bool = True      # Use neural value vs rollout
    max_rollout_depth: int = 10      # If not using value head, max rollout steps


class MCTSNode:
    """
    Node in the Monte Carlo search tree.

    Each node represents a game state and tracks:
    - Visit count: How many times this node was visited
    - Value sum: Cumulative value from backpropagation
    - Prior: Neural network's policy prior for reaching this node
    - Children: Child nodes for each valid action
    """

    def __init__(self, prior: float = 1.0):
        """
        Initialize node.

        Args:
            prior: Policy prior probability from neural network
        """
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, 'MCTSNode'] = {}  # action -> child node

    @property
    def value(self) -> float:
        """Average value across all visits."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        """Check if node has been expanded (has children)."""
        return len(self.children) > 0

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """
        Select best child using PUCT (Predictor + UCB for Trees).

        PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            c_puct: Exploration constant

        Returns:
            Tuple of (action, child_node) with highest PUCT score
        """
        best_score = float('-inf')
        best_action = -1
        best_child = None

        sqrt_parent = math.sqrt(self.visit_count) if self.visit_count > 0 else 1.0

        for action, child in self.children.items():
            # PUCT score combines exploitation (Q) with exploration (U)
            q_value = child.value  # Exploitation: average value
            u_value = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)  # Exploration
            puct_score = q_value + u_value

            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child

        return best_action, best_child

    def select_action_by_visit_count(self, temperature: float = 1.0) -> int:
        """
        Select action based on visit counts (for final move selection).

        Args:
            temperature: Controls randomness. 0 = greedy, 1 = proportional to visits

        Returns:
            Selected action index
        """
        actions = list(self.children.keys())
        visits = np.array([self.children[a].visit_count for a in actions], dtype=np.float64)

        if temperature == 0:
            # Greedy selection
            return actions[np.argmax(visits)]
        else:
            # Temperature-scaled softmax over visit counts
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()
            return np.random.choice(actions, p=probs)

    def get_policy_target(self) -> np.ndarray:
        """
        Get policy target based on visit counts (for training).

        Returns:
            Numpy array of shape [8] with visit count distribution
        """
        policy = np.zeros(8)  # NUM_ACTIONS
        total_visits = sum(child.visit_count for child in self.children.values())

        if total_visits > 0:
            for action, child in self.children.items():
                policy[action] = child.visit_count / total_visits

        return policy


class CroquetSimulator:
    """
    Lightweight game state simulator for MCTS.

    This wraps the croquet game state to allow:
    - Cloning game state for simulation
    - Executing actions and getting new states
    - Checking valid actions
    - Determining terminal states

    Note: This is a simplified simulator that doesn't run full physics.
    It uses heuristic state transitions suitable for MCTS planning.
    """

    def __init__(self, encoder, dm_class):
        """
        Initialize simulator.

        Args:
            encoder: StateEncoder instance for encoding states
            dm_class: TacticalDecisionMaker class for action validation
        """
        self.encoder = encoder
        self.dm_class = dm_class

    def clone_state(
        self,
        ball,
        balls: Dict,
        court,
        deadness,
        rules_state: Dict
    ) -> Dict:
        """
        Clone the current game state for simulation.

        Args:
            ball: Current striker ball
            balls: Dict of all balls
            court: Court object
            deadness: Deadness matrix
            rules_state: Dict with strokes_remaining, is_continuation, etc.

        Returns:
            Dict containing cloned state
        """
        # Deep copy mutable state
        cloned_balls = {}
        for color, b in balls.items():
            cloned_balls[color] = copy.deepcopy(b)

        cloned_deadness = copy.deepcopy(deadness)

        return {
            'striker_color': ball.color,
            'balls': cloned_balls,
            'court': court,  # Court is immutable
            'deadness': cloned_deadness,
            'strokes_remaining': rules_state.get('strokes_remaining', 1),
            'is_continuation': rules_state.get('is_continuation', False),
            'hoops_run': {c: b.hoops_run for c, b in cloned_balls.items()},
        }

    def get_valid_actions(self, state: Dict) -> List[int]:
        """Get valid actions from a state dict."""
        dm = self.dm_class()
        ball = state['balls'][state['striker_color']]
        return dm._get_valid_neural_actions(
            ball, state['balls'], state['court'], state['deadness']
        )

    def encode_state(self, state: Dict) -> 'torch.Tensor':
        """Encode state dict to tensor."""
        ball = state['balls'][state['striker_color']]
        return self.encoder.encode(
            ball, state['balls'], state['court'], state['deadness'],
            state['strokes_remaining'], state['is_continuation']
        )

    def step(self, state: Dict, action: int) -> Tuple[Dict, float, bool]:
        """
        Execute action in state and return new state.

        This is a simplified simulation that doesn't run physics.
        It estimates the effect of actions for MCTS planning.

        Args:
            state: Current state dict
            action: Action index to execute

        Returns:
            Tuple of (new_state, reward, done)
        """
        from models.ball import Vector2

        # Clone state for modification
        new_state = self.clone_state(
            state['balls'][state['striker_color']],
            state['balls'],
            state['court'],
            state['deadness'],
            state
        )

        ball = new_state['balls'][new_state['striker_color']]
        balls = new_state['balls']
        court = new_state['court']

        # Simplified action effects (heuristic, not physics-based)
        reward = 0.0
        done = False

        # Action mapping:
        # 0: HOOP_RUN, 1: ROQUET_NEAREST, 2: ROQUET_PARTNER
        # 3: ROQUET_OPP1, 4: ROQUET_OPP2, 5: APPROACH, 6: DEFENSIVE, 7: PEG_OUT

        if action == 0:  # HOOP_RUN
            # Estimate 30% success rate for hoop attempt
            if np.random.random() < 0.3:
                ball.hoops_run = min(12, ball.hoops_run + 1)
                reward = 10.0
                new_state['strokes_remaining'] = 1  # Continuation
            else:
                reward = -0.5
                new_state['strokes_remaining'] = 0

        elif action in [1, 2, 3, 4]:  # ROQUET
            # Estimate 50% success rate for roquet
            if np.random.random() < 0.5:
                reward = 1.0
                new_state['strokes_remaining'] = 2  # Croquet + continuation
            else:
                reward = -0.5
                new_state['strokes_remaining'] = 0

        elif action == 5:  # APPROACH
            reward = 0.2
            new_state['strokes_remaining'] = 0

        elif action == 6:  # DEFENSIVE
            reward = 0.1
            new_state['strokes_remaining'] = 0

        elif action == 7:  # PEG_OUT
            if ball.hoops_run >= 12:  # Is rover
                # 40% success rate
                if np.random.random() < 0.4:
                    ball.has_pegged_out = True
                    reward = 20.0
                else:
                    reward = -1.0
            new_state['strokes_remaining'] = 0

        # Check for game end
        bb_out = sum(1 for c in ["blue", "black"] if balls[c].has_pegged_out)
        ry_out = sum(1 for c in ["red", "yellow"] if balls[c].has_pegged_out)
        done = (bb_out == 2) or (ry_out == 2)

        # Determine winner if game over
        if done:
            if bb_out == 2:
                # Blue/black wins - positive for blue/black, negative for red/yellow
                if new_state['striker_color'] in ['blue', 'black']:
                    reward = 100.0
                else:
                    reward = -100.0
            else:
                # Red/yellow wins
                if new_state['striker_color'] in ['red', 'yellow']:
                    reward = 100.0
                else:
                    reward = -100.0

        return new_state, reward, done

    def is_terminal(self, state: Dict) -> bool:
        """Check if state is terminal (game over)."""
        balls = state['balls']
        bb_out = sum(1 for c in ["blue", "black"] if balls[c].has_pegged_out)
        ry_out = sum(1 for c in ["red", "yellow"] if balls[c].has_pegged_out)
        return (bb_out == 2) or (ry_out == 2)


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    This implements AlphaZero-style MCTS where:
    - Policy network provides prior probabilities for tree expansion
    - Value network evaluates leaf nodes (no random rollouts)
    - PUCT formula balances exploitation and exploration
    - Dirichlet noise at root encourages exploration during training

    Usage:
        mcts = MCTS(network, config)
        policy = mcts.search(state, valid_actions, simulator)
        action = mcts.select_action(temperature=1.0)
    """

    def __init__(
        self,
        network: 'PolicyValueNet',
        config: MCTSConfig = None,
        device: str = 'cpu'
    ):
        """
        Initialize MCTS.

        Args:
            network: PolicyValueNet for policy prior and value estimation
            config: MCTSConfig with search parameters
            device: Torch device for network inference
        """
        self.network = network
        self.config = config or MCTSConfig()
        self.device = device
        self.root: Optional[MCTSNode] = None

    def search(
        self,
        state_tensor: 'torch.Tensor',
        valid_actions: List[int],
        simulator: CroquetSimulator,
        game_state: Dict,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Run MCTS from current state.

        Args:
            state_tensor: Encoded state tensor for network
            valid_actions: Legal actions from this state
            simulator: Game simulator for state transitions
            game_state: Current game state dict (for simulation)
            add_noise: Add Dirichlet noise at root (training only)

        Returns:
            Policy vector [NUM_ACTIONS] based on visit counts
        """
        # Create root node
        self.root = MCTSNode()

        # Expand root with network prior
        self._expand_node(self.root, state_tensor, valid_actions, add_noise=add_noise)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = self.root
            current_state = copy.deepcopy(game_state)
            search_path = [node]

            # Selection: traverse tree to leaf using PUCT
            while node.expanded() and not simulator.is_terminal(current_state):
                action, node = node.select_child(self.config.c_puct)
                search_path.append(node)

                # Simulate action
                current_state, _, done = simulator.step(current_state, action)

                if done:
                    break

            # Evaluate leaf
            if simulator.is_terminal(current_state):
                # Terminal state - use actual outcome
                balls = current_state['balls']
                bb_out = sum(1 for c in ["blue", "black"] if balls[c].has_pegged_out)
                value = 1.0 if bb_out == 2 else -1.0
            elif not node.expanded():
                # Expand and evaluate with network
                state_tensor = simulator.encode_state(current_state)
                next_valid = simulator.get_valid_actions(current_state)

                if next_valid:
                    self._expand_node(node, state_tensor, next_valid, add_noise=False)

                # Get value from network
                with torch.no_grad():
                    _, value_tensor = self.network(state_tensor.to(self.device))
                    value = value_tensor.item()
            else:
                # Expanded but terminal
                value = 0.0

            # Backpropagate value through search path
            self._backpropagate(search_path, value)

        # Return visit count distribution as policy
        return self.root.get_policy_target()

    def _expand_node(
        self,
        node: MCTSNode,
        state_tensor: 'torch.Tensor',
        valid_actions: List[int],
        add_noise: bool = False
    ):
        """
        Expand node by creating children with policy prior.

        Args:
            node: Node to expand
            state_tensor: Encoded state for network
            valid_actions: Valid actions at this state
            add_noise: Add Dirichlet noise for exploration
        """
        with torch.no_grad():
            policy_logits, _ = self.network(state_tensor.to(self.device))

            # Mask invalid actions
            mask = torch.full_like(policy_logits, float('-inf'))
            for a in valid_actions:
                mask[0, a] = 0
            policy = F.softmax(policy_logits + mask, dim=-1).squeeze().cpu().numpy()

        # Add Dirichlet noise at root for exploration
        if add_noise and len(valid_actions) > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_actions))
            for i, a in enumerate(valid_actions):
                policy[a] = (1 - self.config.root_noise_frac) * policy[a] + \
                           self.config.root_noise_frac * noise[i]

        # Create child nodes with policy prior
        for action in valid_actions:
            node.children[action] = MCTSNode(prior=float(policy[action]))

    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """
        Backpropagate value up the search path.

        In two-player games, value alternates sign at each level
        because players have opposing objectives.

        Args:
            search_path: List of nodes from root to leaf
            value: Value at leaf node
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent's perspective

    def select_action(self, temperature: float = None) -> int:
        """
        Select action from root based on visit counts.

        Args:
            temperature: Softmax temperature (None = use config default)

        Returns:
            Selected action index
        """
        if self.root is None or not self.root.expanded():
            raise ValueError("Must call search() before select_action()")

        temp = temperature if temperature is not None else self.config.temperature
        return self.root.select_action_by_visit_count(temp)

    def get_action_probabilities(self) -> np.ndarray:
        """Get action probabilities from root visit counts."""
        if self.root is None:
            return np.zeros(8)
        return self.root.get_policy_target()

    def get_root_value(self) -> float:
        """Get estimated value of root state."""
        if self.root is None:
            return 0.0
        return self.root.value


def create_mcts_for_training(
    network: 'PolicyValueNet',
    encoder,
    dm_class,
    num_simulations: int = 50,
    device: str = 'cpu'
) -> Tuple[MCTS, CroquetSimulator]:
    """
    Factory function to create MCTS with simulator for training.

    Args:
        network: PolicyValueNet instance
        encoder: StateEncoder instance
        dm_class: TacticalDecisionMaker class
        num_simulations: Number of MCTS simulations per move
        device: Torch device

    Returns:
        Tuple of (MCTS instance, CroquetSimulator instance)
    """
    config = MCTSConfig(
        num_simulations=num_simulations,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        root_noise_frac=0.25
    )

    mcts = MCTS(network, config, device)
    simulator = CroquetSimulator(encoder, dm_class)

    return mcts, simulator
