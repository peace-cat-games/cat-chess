"""
Cat Chess AI - Simple heuristic-based AI for all three factions.
"""

import random
from typing import List, Optional, Tuple
from copy import deepcopy

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


# Piece values for evaluation
PIECE_VALUES = {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 320,
    PieceType.BISHOP: 330,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 20000,
    PieceType.CAT: 0,
}

# Positional bonuses for pawns (from white's perspective)
PAWN_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

KNIGHT_TABLE = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
]

CENTER_SQUARES = {Square(3, 3), Square(3, 4), Square(4, 3), Square(4, 4)}


class CatChessAI:
    """AI player for Cat Chess."""

    def __init__(self, engine: CatChessEngine, faction: Faction, depth: int = 3):
        self.engine = engine
        self.faction = faction
        self.depth = depth

    def get_best_move(self) -> Optional[Move]:
        """Get the best move for the current faction."""
        if self.faction == Faction.CATS:
            return self._get_cat_move()
        else:
            return self._get_warring_faction_move()

    def _get_warring_faction_move(self) -> Optional[Move]:
        """Get best move for white or black using minimax."""
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None

        best_move = None
        best_score = float("-inf")

        for move in legal_moves:
            # Make move temporarily
            original_state = deepcopy(self.engine.state)
            self.engine.make_move(move)

            # Evaluate
            score = self._minimax(self.depth - 1, float("-inf"), float("inf"), False)

            # Restore state
            self.engine.state = original_state

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _minimax(
        self, depth: int, alpha: float, beta: float, maximizing: bool
    ) -> float:
        """Minimax with alpha-beta pruning."""
        result = self.engine.get_game_result()
        if result:
            if result == self.faction.value:
                return 100000 + depth  # Win sooner is better
            elif result == Faction.CATS.value:
                return -50000  # Draw is bad for warring factions
            else:
                return -100000 - depth  # Lose later is better

        if depth == 0:
            return self._evaluate_position()

        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return self._evaluate_position()

        if maximizing:
            max_eval = float("-inf")
            for move in legal_moves:
                original_state = deepcopy(self.engine.state)
                self.engine.make_move(move)
                eval_score = self._minimax(depth - 1, alpha, beta, False)
                self.engine.state = original_state
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in legal_moves:
                original_state = deepcopy(self.engine.state)
                self.engine.make_move(move)
                eval_score = self._minimax(depth - 1, alpha, beta, True)
                self.engine.state = original_state
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate_position(self) -> float:
        """Evaluate the current position from the faction's perspective."""
        score = 0.0
        disarmed_squares = self.engine.state.get_disarmed_squares()

        for piece in self.engine.state.board.values():
            if piece.piece_type == PieceType.CAT:
                continue

            value = PIECE_VALUES[piece.piece_type]

            # Positional bonus
            value += self._get_positional_bonus(piece)

            # Penalty for being disarmed
            if piece.square in disarmed_squares:
                value *= 0.7  # 30% penalty for being disarmed

            # Add or subtract based on faction
            if piece.faction == self.faction:
                score += value
            else:
                score -= value

        # Mobility bonus
        our_moves = len(self.engine.get_legal_moves(self.faction))
        enemy = Faction.BLACK if self.faction == Faction.WHITE else Faction.WHITE
        enemy_moves = len(self.engine.get_legal_moves(enemy))
        score += (our_moves - enemy_moves) * 5

        return score

    def _get_positional_bonus(self, piece: Piece) -> float:
        """Get positional bonus for a piece."""
        file, rank = piece.square.file, piece.square.rank

        # Flip rank for black
        if piece.faction == Faction.BLACK:
            rank = 7 - rank

        if piece.piece_type == PieceType.PAWN:
            return PAWN_TABLE[7 - rank][file]
        elif piece.piece_type == PieceType.KNIGHT:
            return KNIGHT_TABLE[7 - rank][file]

        # Center control bonus for other pieces
        if piece.square in CENTER_SQUARES:
            return 10

        return 0

    def _get_cat_move(self) -> Optional[Move]:
        """Get best move for cats - goal is to maximize draw likelihood."""
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None

        best_move = None
        best_score = float("-inf")

        for move in legal_moves:
            score = self._evaluate_cat_move(move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _evaluate_cat_move(self, move: Move) -> float:
        """Evaluate a cat move based on peacekeeper strategy."""
        score = 0.0
        target = move.to_square

        # Make move temporarily to evaluate
        original_state = deepcopy(self.engine.state)
        self.engine.make_move(move)

        new_disarmed = self.engine.state.get_disarmed_squares()

        # Count pieces that would be disarmed
        disarmed_count = 0
        disarmed_value = 0
        for piece in self.engine.state.board.values():
            if piece.piece_type != PieceType.CAT and piece.square in new_disarmed:
                disarmed_count += 1
                disarmed_value += PIECE_VALUES[piece.piece_type]

        # Reward disarming more pieces
        score += disarmed_count * 50

        # Reward disarming valuable pieces
        score += disarmed_value * 0.1

        # Reward being near kings (protect them from mate)
        for faction in [Faction.WHITE, Faction.BLACK]:
            king = self.engine.state.get_king(faction)
            if king:
                distance = target.chebyshev_distance(king.square)
                if distance <= 2:
                    score += (3 - distance) * 30

        # Reward being near the center
        center = Square(3, 3)
        center_dist = target.chebyshev_distance(center)
        score += (7 - center_dist) * 5

        # Reward positions that create more symmetry (balance of power)
        white_material = sum(
            PIECE_VALUES[p.piece_type]
            for p in self.engine.state.get_pieces(Faction.WHITE)
        )
        black_material = sum(
            PIECE_VALUES[p.piece_type]
            for p in self.engine.state.get_pieces(Faction.BLACK)
        )
        material_diff = abs(white_material - black_material)
        score -= material_diff * 0.01  # Prefer balanced positions

        # Check if this position leads to fewer legal moves for both sides (stalemate potential)
        white_moves = len(self.engine.get_legal_moves(Faction.WHITE))
        black_moves = len(self.engine.get_legal_moves(Faction.BLACK))
        if white_moves < 5 or black_moves < 5:
            score += 100  # Potential stalemate situation

        # Restore state
        self.engine.state = original_state

        return score


class RandomAI:
    """Simple random move AI for testing."""

    def __init__(self, engine: CatChessEngine):
        self.engine = engine

    def get_best_move(self) -> Optional[Move]:
        legal_moves = self.engine.get_legal_moves()
        return random.choice(legal_moves) if legal_moves else None


def play_ai_game(
    white_ai: bool = True,
    black_ai: bool = True,
    cat_ai: bool = True,
    depth: int = 2,
    max_moves: int = 200,
    verbose: bool = True,
) -> str:
    """Play a game with AI players."""
    engine = CatChessEngine()

    ais = {}
    if white_ai:
        ais[Faction.WHITE] = CatChessAI(engine, Faction.WHITE, depth)
    if black_ai:
        ais[Faction.BLACK] = CatChessAI(engine, Faction.BLACK, depth)
    if cat_ai:
        ais[Faction.CATS] = CatChessAI(engine, Faction.CATS, depth)

    move_count = 0

    while move_count < max_moves:
        result = engine.get_game_result()
        if result:
            if verbose:
                print(f"\nGame Over! Winner: {result}")
                print(engine.render_board())
            return result

        current_turn = engine.state.turn

        if verbose:
            print(f"\n--- Move {move_count + 1} ({current_turn.value}) ---")
            print(engine.render_board())

        if current_turn in ais:
            move = ais[current_turn].get_best_move()
            if move:
                if verbose:
                    print(f"AI plays: {move.to_algebraic()}")
                engine.make_move(move)
            else:
                # No legal moves
                break
        else:
            # Human would play here
            print("Human turn - no move made")
            break

        move_count += 1

    # If we hit max moves, it's a draw (cats win)
    if verbose:
        print(f"\nGame ended after {move_count} moves")
    return Faction.CATS.value


if __name__ == "__main__":
    result = play_ai_game(depth=2, max_moves=100, verbose=True)
    print(f"\nFinal Result: {result}")
