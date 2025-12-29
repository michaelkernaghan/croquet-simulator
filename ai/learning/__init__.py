"""
Learning system for the AI croquet players.

Implements a hybrid approach combining:
- Position evaluation (neural network or weighted features)
- Shot outcome simulation (Monte Carlo)
- Tactical rules (expert knowledge)
- Self-play learning (experience replay)
"""

from .position_evaluator import PositionEvaluator
from .shot_simulator import ShotSimulator
from .tactical_rules import TacticalRules
from .learner import CroquetLearner

__all__ = ['PositionEvaluator', 'ShotSimulator', 'TacticalRules', 'CroquetLearner']
