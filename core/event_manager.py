"""
Event Manager for decoupled communication between game components.
Uses the Observer/Mediator pattern.
"""
from typing import Callable, Dict, List, Any
from enum import Enum, auto


class EventType(Enum):
    """Types of events that can occur in the game."""
    # Game lifecycle
    GAME_START = auto()
    GAME_END = auto()

    # Turn events
    TURN_START = auto()
    TURN_END = auto()

    # Shot events
    SHOT_SELECTED = auto()
    SHOT_EXECUTING = auto()
    SHOT_COMPLETE = auto()

    # Ball events
    BALL_MOVING = auto()
    BALL_STOPPED = auto()
    ALL_BALLS_STOPPED = auto()

    # Croquet events
    ROQUET_OCCURRED = auto()
    HOOP_RUN = auto()
    PEG_HIT = auto()

    # State changes
    DEADNESS_CHANGED = auto()
    SCORE_UPDATED = auto()

    # Physics events
    COLLISION_BALL_BALL = auto()
    COLLISION_BALL_BOUNDARY = auto()


class Event:
    """Represents a game event with associated data."""

    def __init__(self, event_type: EventType, data: Dict[str, Any] = None):
        self.type = event_type
        self.data = data or {}

    def __repr__(self):
        return f"Event({self.type.name}, {self.data})"


class EventManager:
    """
    Central event management system.
    Components register as listeners and can post events.
    """

    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}
        self._event_queue: List[Event] = []

    def register(self, event_type: EventType, callback: Callable[[Event], None]):
        """Register a callback for a specific event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        if callback not in self._listeners[event_type]:
            self._listeners[event_type].append(callback)

    def unregister(self, event_type: EventType, callback: Callable[[Event], None]):
        """Remove a callback from an event type."""
        if event_type in self._listeners:
            if callback in self._listeners[event_type]:
                self._listeners[event_type].remove(callback)

    def post(self, event: Event):
        """Post an event to be processed immediately."""
        if event.type in self._listeners:
            for callback in self._listeners[event.type]:
                callback(event)

    def queue(self, event: Event):
        """Queue an event for later processing."""
        self._event_queue.append(event)

    def process_queue(self):
        """Process all queued events."""
        while self._event_queue:
            event = self._event_queue.pop(0)
            self.post(event)

    def clear_listeners(self):
        """Remove all registered listeners."""
        self._listeners.clear()
