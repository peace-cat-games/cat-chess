"""Cat Chess - A three-faction chess variant with peacekeeping cats."""

from .engine import (
    CatChessEngine,
    GameState,
    Move,
    Piece,
    Square,
    Faction,
    PieceType,
    TURN_ORDER,
)

from .ai import CatChessAI, RandomAI, play_ai_game

__all__ = [
    "CatChessEngine",
    "GameState",
    "Move",
    "Piece",
    "Square",
    "Faction",
    "PieceType",
    "TURN_ORDER",
    "CatChessAI",
    "RandomAI",
    "play_ai_game",
]

__version__ = "1.0.0"
