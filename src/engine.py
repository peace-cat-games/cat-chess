"""
Cat Chess Engine - Core game logic implementation.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Set, List, Tuple
from copy import deepcopy


class Faction(Enum):
    WHITE = "white"
    BLACK = "black"
    CATS = "cats"


class PieceType(Enum):
    KING = "K"
    QUEEN = "Q"
    ROOK = "R"
    BISHOP = "B"
    KNIGHT = "N"
    PAWN = "P"
    CAT = "C"


TURN_ORDER = [Faction.WHITE, Faction.BLACK, Faction.CATS]


@dataclass
class Square:
    """Represents a board square using 0-indexed coordinates."""

    file: int  # 0-7 (a-h)
    rank: int  # 0-7 (1-8)

    def __hash__(self):
        return hash((self.file, self.rank))

    def __eq__(self, other):
        if not isinstance(other, Square):
            return False
        return self.file == other.file and self.rank == other.rank

    def is_valid(self) -> bool:
        return 0 <= self.file < 8 and 0 <= self.rank < 8

    def to_algebraic(self) -> str:
        return f"{chr(ord('a') + self.file)}{self.rank + 1}"

    @classmethod
    def from_algebraic(cls, notation: str) -> "Square":
        file = ord(notation[0].lower()) - ord("a")
        rank = int(notation[1]) - 1
        return cls(file, rank)

    def chebyshev_distance(self, other: "Square") -> int:
        return max(abs(self.file - other.file), abs(self.rank - other.rank))


@dataclass
class Piece:
    """Represents a chess piece."""

    piece_type: PieceType
    faction: Faction
    square: Square

    def symbol(self) -> str:
        s = self.piece_type.value
        if self.faction == Faction.BLACK:
            return s.lower()
        elif self.faction == Faction.CATS:
            return "C" if self.piece_type == PieceType.CAT else s
        return s


@dataclass
class Move:
    """Represents a move in the game."""

    piece: Piece
    from_square: Square
    to_square: Square
    captured_piece: Optional[Piece] = None
    promotion: Optional[PieceType] = None
    is_castling: bool = False
    is_en_passant: bool = False

    def to_algebraic(self) -> str:
        # Pass move (cat stays in place)
        if self.from_square == self.to_square:
            return f"{self.from_square.to_algebraic()}(pass)"
        base = f"{self.from_square.to_algebraic()}{self.to_square.to_algebraic()}"
        if self.promotion:
            base += self.promotion.value.lower()
        return base

    def is_pass(self) -> bool:
        """Check if this is a pass move (piece stays in place)."""
        return self.from_square == self.to_square


@dataclass
class GameState:
    """Complete game state representation."""

    board: dict  # Square -> Piece
    turn: Faction
    cat_positions: List[Square]
    halfmove_clock: int = 0
    fullmove_number: int = 1
    castling_rights: dict = field(
        default_factory=lambda: {
            Faction.WHITE: {"kingside": True, "queenside": True},
            Faction.BLACK: {"kingside": True, "queenside": True},
        }
    )
    en_passant_target: Optional[Square] = None
    position_history: List[str] = field(default_factory=list)

    def get_disarmed_squares(self) -> Set[Square]:
        """Get all squares in cat no-kill zones."""
        disarmed = set()
        for cat_pos in self.cat_positions:
            for df in range(-1, 2):
                for dr in range(-1, 2):
                    sq = Square(cat_pos.file + df, cat_pos.rank + dr)
                    if sq.is_valid():
                        disarmed.add(sq)
        return disarmed

    def is_disarmed(self, square: Square) -> bool:
        """Check if a piece on this square is disarmed."""
        for cat_pos in self.cat_positions:
            if square.chebyshev_distance(cat_pos) <= 1:
                return True
        return False

    def get_piece_at(self, square: Square) -> Optional[Piece]:
        return self.board.get(square)

    def get_king(self, faction: Faction) -> Optional[Piece]:
        for piece in self.board.values():
            if piece.piece_type == PieceType.KING and piece.faction == faction:
                return piece
        return None

    def get_pieces(self, faction: Faction) -> List[Piece]:
        return [p for p in self.board.values() if p.faction == faction]

    def get_cats(self) -> List[Piece]:
        return [p for p in self.board.values() if p.piece_type == PieceType.CAT]

    def position_key(self) -> str:
        """Generate a position key for repetition detection."""
        pieces = []
        for sq, piece in sorted(
            self.board.items(), key=lambda x: (x[0].file, x[0].rank)
        ):
            pieces.append(f"{sq.to_algebraic()}{piece.symbol()}")
        return f"{','.join(pieces)}|{self.turn.value}"


class CatChessEngine:
    """Main engine class for Cat Chess."""

    def __init__(self):
        self.state = self._create_initial_state()

    def _create_initial_state(self) -> GameState:
        """Set up the initial board position."""
        board = {}

        # White pieces (rank 1)
        back_rank_order = [
            PieceType.ROOK,
            PieceType.KNIGHT,
            PieceType.BISHOP,
            PieceType.QUEEN,
            PieceType.KING,
            PieceType.BISHOP,
            PieceType.KNIGHT,
            PieceType.ROOK,
        ]
        for file, piece_type in enumerate(back_rank_order):
            sq = Square(file, 0)
            board[sq] = Piece(piece_type, Faction.WHITE, sq)

        # White pawns (rank 2)
        for file in range(8):
            sq = Square(file, 1)
            board[sq] = Piece(PieceType.PAWN, Faction.WHITE, sq)

        # Black pieces (rank 8)
        for file, piece_type in enumerate(back_rank_order):
            sq = Square(file, 7)
            board[sq] = Piece(piece_type, Faction.BLACK, sq)

        # Black pawns (rank 7)
        for file in range(8):
            sq = Square(file, 6)
            board[sq] = Piece(PieceType.PAWN, Faction.BLACK, sq)

        # Cats
        cat1_sq = Square.from_algebraic("a4")
        cat2_sq = Square.from_algebraic("h5")
        board[cat1_sq] = Piece(PieceType.CAT, Faction.CATS, cat1_sq)
        board[cat2_sq] = Piece(PieceType.CAT, Faction.CATS, cat2_sq)

        return GameState(
            board=board, turn=Faction.WHITE, cat_positions=[cat1_sq, cat2_sq]
        )

    def reset(self):
        """Reset the game to initial state."""
        self.state = self._create_initial_state()

    def get_attacked_squares(
        self, faction: Faction, exclude_king: bool = False
    ) -> Set[Square]:
        """Get all squares attacked by armed pieces of a faction."""
        attacked = set()
        disarmed_squares = self.state.get_disarmed_squares()

        for piece in self.state.get_pieces(faction):
            if piece.square in disarmed_squares:
                continue  # Disarmed pieces don't attack
            if exclude_king and piece.piece_type == PieceType.KING:
                continue
            attacked.update(self._get_piece_attacks(piece))

        return attacked

    def _get_piece_attacks(self, piece: Piece) -> Set[Square]:
        """Get squares a piece attacks (ignoring capture restrictions)."""
        attacks = set()
        sq = piece.square

        if piece.piece_type == PieceType.PAWN:
            direction = 1 if piece.faction == Faction.WHITE else -1
            for df in [-1, 1]:
                target = Square(sq.file + df, sq.rank + direction)
                if target.is_valid():
                    attacks.add(target)

        elif piece.piece_type == PieceType.KNIGHT:
            knight_moves = [
                (-2, -1),
                (-2, 1),
                (-1, -2),
                (-1, 2),
                (1, -2),
                (1, 2),
                (2, -1),
                (2, 1),
            ]
            for df, dr in knight_moves:
                target = Square(sq.file + df, sq.rank + dr)
                if target.is_valid():
                    attacks.add(target)

        elif piece.piece_type == PieceType.BISHOP:
            attacks.update(
                self._get_sliding_attacks(sq, [(-1, -1), (-1, 1), (1, -1), (1, 1)])
            )

        elif piece.piece_type == PieceType.ROOK:
            attacks.update(
                self._get_sliding_attacks(sq, [(-1, 0), (1, 0), (0, -1), (0, 1)])
            )

        elif piece.piece_type == PieceType.QUEEN:
            attacks.update(
                self._get_sliding_attacks(
                    sq,
                    [
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                    ],
                )
            )

        elif piece.piece_type == PieceType.KING:
            for df in range(-1, 2):
                for dr in range(-1, 2):
                    if df == 0 and dr == 0:
                        continue
                    target = Square(sq.file + df, sq.rank + dr)
                    if target.is_valid():
                        attacks.add(target)

        return attacks

    def _get_sliding_attacks(
        self, sq: Square, directions: List[Tuple[int, int]]
    ) -> Set[Square]:
        """Get attacked squares for sliding pieces (bishop, rook, queen)."""
        attacks = set()
        for df, dr in directions:
            for dist in range(1, 8):
                target = Square(sq.file + df * dist, sq.rank + dr * dist)
                if not target.is_valid():
                    break
                attacks.add(target)
                if self.state.get_piece_at(target):
                    break
        return attacks

    def is_in_check(self, faction: Faction) -> bool:
        """Check if a faction's king is in check."""
        if faction == Faction.CATS:
            return False
        king = self.state.get_king(faction)
        if not king:
            return False
        enemy = Faction.BLACK if faction == Faction.WHITE else Faction.WHITE
        return king.square in self.get_attacked_squares(enemy)

    def get_legal_moves(self, faction: Optional[Faction] = None) -> List[Move]:
        """Generate all legal moves for a faction."""
        if faction is None:
            faction = self.state.turn

        if faction == Faction.CATS:
            return self._get_cat_moves()

        moves = []
        disarmed_squares = self.state.get_disarmed_squares()

        for piece in self.state.get_pieces(faction):
            moves.extend(self._get_piece_moves(piece, disarmed_squares))

        # Filter out moves that leave king in check
        legal_moves = []
        for move in moves:
            if self._is_move_legal(move, faction):
                legal_moves.append(move)

        return legal_moves

    def _get_cat_moves(self) -> List[Move]:
        """Generate moves for cat pieces.

        Cats can move one square in any direction to an empty square,
        or pass (stay in place) if their position is already optimal.
        """
        moves = []
        for cat in self.state.get_cats():
            sq = cat.square
            # Pass move - cat stays in place
            moves.append(Move(cat, sq, sq))
            # Regular moves - one square in any direction
            for df in range(-1, 2):
                for dr in range(-1, 2):
                    if df == 0 and dr == 0:
                        continue
                    target = Square(sq.file + df, sq.rank + dr)
                    if target.is_valid() and not self.state.get_piece_at(target):
                        moves.append(Move(cat, sq, target))
        return moves

    def _get_piece_moves(
        self, piece: Piece, disarmed_squares: Set[Square]
    ) -> List[Move]:
        """Generate pseudo-legal moves for a piece."""
        moves = []
        sq = piece.square
        is_disarmed = sq in disarmed_squares

        if piece.piece_type == PieceType.PAWN:
            moves.extend(self._get_pawn_moves(piece, is_disarmed, disarmed_squares))

        elif piece.piece_type == PieceType.KNIGHT:
            knight_moves = [
                (-2, -1),
                (-2, 1),
                (-1, -2),
                (-1, 2),
                (1, -2),
                (1, 2),
                (2, -1),
                (2, 1),
            ]
            for df, dr in knight_moves:
                target = Square(sq.file + df, sq.rank + dr)
                if target.is_valid():
                    move = self._create_move_if_valid(
                        piece, target, is_disarmed, disarmed_squares
                    )
                    if move:
                        moves.append(move)

        elif piece.piece_type == PieceType.BISHOP:
            moves.extend(
                self._get_sliding_moves(
                    piece,
                    [(-1, -1), (-1, 1), (1, -1), (1, 1)],
                    is_disarmed,
                    disarmed_squares,
                )
            )

        elif piece.piece_type == PieceType.ROOK:
            moves.extend(
                self._get_sliding_moves(
                    piece,
                    [(-1, 0), (1, 0), (0, -1), (0, 1)],
                    is_disarmed,
                    disarmed_squares,
                )
            )

        elif piece.piece_type == PieceType.QUEEN:
            moves.extend(
                self._get_sliding_moves(
                    piece,
                    [
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                    ],
                    is_disarmed,
                    disarmed_squares,
                )
            )

        elif piece.piece_type == PieceType.KING:
            moves.extend(self._get_king_moves(piece, is_disarmed, disarmed_squares))

        return moves

    def _get_pawn_moves(
        self, piece: Piece, is_disarmed: bool, disarmed_squares: Set[Square]
    ) -> List[Move]:
        """Generate pawn moves."""
        moves = []
        sq = piece.square
        direction = 1 if piece.faction == Faction.WHITE else -1
        start_rank = 1 if piece.faction == Faction.WHITE else 6
        promo_rank = 7 if piece.faction == Faction.WHITE else 0

        # Forward move
        forward = Square(sq.file, sq.rank + direction)
        if forward.is_valid() and not self.state.get_piece_at(forward):
            if forward.rank == promo_rank:
                for promo in [
                    PieceType.QUEEN,
                    PieceType.ROOK,
                    PieceType.BISHOP,
                    PieceType.KNIGHT,
                ]:
                    moves.append(Move(piece, sq, forward, promotion=promo))
            else:
                moves.append(Move(piece, sq, forward))

            # Double move from start
            if sq.rank == start_rank:
                double = Square(sq.file, sq.rank + 2 * direction)
                if not self.state.get_piece_at(double):
                    moves.append(Move(piece, sq, double))

        # Captures (only if not disarmed and target not in no-kill zone)
        if not is_disarmed:
            for df in [-1, 1]:
                target = Square(sq.file + df, sq.rank + direction)
                if not target.is_valid():
                    continue
                if target in disarmed_squares:
                    continue  # Can't capture into no-kill zone

                target_piece = self.state.get_piece_at(target)
                if (
                    target_piece
                    and target_piece.faction != piece.faction
                    and target_piece.piece_type != PieceType.CAT
                ):
                    if target.rank == promo_rank:
                        for promo in [
                            PieceType.QUEEN,
                            PieceType.ROOK,
                            PieceType.BISHOP,
                            PieceType.KNIGHT,
                        ]:
                            moves.append(
                                Move(
                                    piece,
                                    sq,
                                    target,
                                    captured_piece=target_piece,
                                    promotion=promo,
                                )
                            )
                    else:
                        moves.append(
                            Move(piece, sq, target, captured_piece=target_piece)
                        )

                # En passant
                if (
                    self.state.en_passant_target
                    and target == self.state.en_passant_target
                ):
                    captured_sq = Square(target.file, sq.rank)
                    captured = self.state.get_piece_at(captured_sq)
                    if captured and captured_sq not in disarmed_squares:
                        moves.append(
                            Move(
                                piece,
                                sq,
                                target,
                                captured_piece=captured,
                                is_en_passant=True,
                            )
                        )

        return moves

    def _get_sliding_moves(
        self,
        piece: Piece,
        directions: List[Tuple[int, int]],
        is_disarmed: bool,
        disarmed_squares: Set[Square],
    ) -> List[Move]:
        """Generate moves for sliding pieces."""
        moves = []
        sq = piece.square

        for df, dr in directions:
            for dist in range(1, 8):
                target = Square(sq.file + df * dist, sq.rank + dr * dist)
                if not target.is_valid():
                    break

                target_piece = self.state.get_piece_at(target)
                if not target_piece:
                    moves.append(Move(piece, sq, target))
                else:
                    # Can capture if not disarmed and target not in no-kill zone and not a cat
                    if (
                        not is_disarmed
                        and target not in disarmed_squares
                        and target_piece.faction != piece.faction
                        and target_piece.piece_type != PieceType.CAT
                    ):
                        moves.append(
                            Move(piece, sq, target, captured_piece=target_piece)
                        )
                    break

        return moves

    def _get_king_moves(
        self, piece: Piece, is_disarmed: bool, disarmed_squares: Set[Square]
    ) -> List[Move]:
        """Generate king moves including castling."""
        moves = []
        sq = piece.square
        enemy = Faction.BLACK if piece.faction == Faction.WHITE else Faction.WHITE
        attacked = self.get_attacked_squares(enemy)

        for df in range(-1, 2):
            for dr in range(-1, 2):
                if df == 0 and dr == 0:
                    continue
                target = Square(sq.file + df, sq.rank + dr)
                if not target.is_valid():
                    continue
                if target in attacked:
                    continue  # Can't move into check

                target_piece = self.state.get_piece_at(target)
                if not target_piece:
                    moves.append(Move(piece, sq, target))
                elif (
                    not is_disarmed
                    and target not in disarmed_squares
                    and target_piece.faction != piece.faction
                    and target_piece.piece_type != PieceType.CAT
                ):
                    moves.append(Move(piece, sq, target, captured_piece=target_piece))

        # Castling
        if not self.is_in_check(piece.faction):
            moves.extend(self._get_castling_moves(piece, attacked))

        return moves

    def _get_castling_moves(self, king: Piece, attacked: Set[Square]) -> List[Move]:
        """Generate castling moves."""
        moves = []
        rights = self.state.castling_rights.get(king.faction, {})
        rank = 0 if king.faction == Faction.WHITE else 7

        # Kingside
        if rights.get("kingside", False):
            if self._can_castle(king, rank, 5, 6, 7, attacked=attacked):
                target = Square(6, rank)
                moves.append(Move(king, king.square, target, is_castling=True))

        # Queenside
        if rights.get("queenside", False):
            if self._can_castle(
                king, rank, 1, 2, 3, 0, attacked=attacked, check_files=[2, 3]
            ):
                target = Square(2, rank)
                moves.append(Move(king, king.square, target, is_castling=True))

        return moves

    def _can_castle(
        self,
        king: Piece,
        rank: int,
        *files,
        attacked: Set[Square],
        check_files: Optional[List[int]] = None,
    ) -> bool:
        """Check if castling is possible."""
        # Check squares between king and rook are empty
        empty_files = files[:-1]  # All except rook file
        for f in empty_files:
            if self.state.get_piece_at(Square(f, rank)):
                return False

        # Check king doesn't pass through or end on attacked square
        pass_files = check_files or [files[0], files[1]]
        for f in pass_files:
            if Square(f, rank) in attacked:
                return False

        # Check rook exists
        rook_sq = Square(files[-1], rank)
        rook = self.state.get_piece_at(rook_sq)
        if (
            not rook
            or rook.piece_type != PieceType.ROOK
            or rook.faction != king.faction
        ):
            return False

        return True

    def _create_move_if_valid(
        self,
        piece: Piece,
        target: Square,
        is_disarmed: bool,
        disarmed_squares: Set[Square],
    ) -> Optional[Move]:
        """Create a move if it's valid according to basic rules."""
        target_piece = self.state.get_piece_at(target)

        if not target_piece:
            return Move(piece, piece.square, target)
        elif (
            not is_disarmed
            and target not in disarmed_squares
            and target_piece.faction != piece.faction
            and target_piece.piece_type != PieceType.CAT
        ):
            return Move(piece, piece.square, target, captured_piece=target_piece)
        return None

    def _is_move_legal(self, move: Move, faction: Faction) -> bool:
        """Check if a move is legal (doesn't leave king in check)."""
        # Make the move temporarily
        original_state = deepcopy(self.state)
        self._apply_move_to_state(move)

        # Check if king is in check
        in_check = self.is_in_check(faction)

        # Restore state
        self.state = original_state
        return not in_check

    def _apply_move_to_state(self, move: Move):
        """Apply a move to the current state (mutates state)."""
        # Remove piece from origin
        del self.state.board[move.from_square]

        # Handle capture
        if move.captured_piece:
            if move.is_en_passant:
                captured_sq = Square(move.to_square.file, move.from_square.rank)
                del self.state.board[captured_sq]
            else:
                del self.state.board[move.to_square]

        # Place piece at destination
        new_piece = Piece(
            move.promotion if move.promotion else move.piece.piece_type,
            move.piece.faction,
            move.to_square,
        )
        self.state.board[move.to_square] = new_piece

        # Handle castling - move rook
        if move.is_castling:
            rank = move.from_square.rank
            if move.to_square.file == 6:  # Kingside
                rook_from = Square(7, rank)
                rook_to = Square(5, rank)
            else:  # Queenside
                rook_from = Square(0, rank)
                rook_to = Square(3, rank)

            rook = self.state.board[rook_from]
            del self.state.board[rook_from]
            self.state.board[rook_to] = Piece(PieceType.ROOK, rook.faction, rook_to)

        # Update cat positions
        if move.piece.piece_type == PieceType.CAT:
            self.state.cat_positions = [
                move.to_square if cp == move.from_square else cp
                for cp in self.state.cat_positions
            ]

        # Update castling rights
        if move.piece.piece_type == PieceType.KING:
            self.state.castling_rights[move.piece.faction] = {
                "kingside": False,
                "queenside": False,
            }
        elif move.piece.piece_type == PieceType.ROOK:
            faction = move.piece.faction
            if move.from_square.file == 0:
                self.state.castling_rights[faction]["queenside"] = False
            elif move.from_square.file == 7:
                self.state.castling_rights[faction]["kingside"] = False

        # Update en passant target
        if (
            move.piece.piece_type == PieceType.PAWN
            and abs(move.to_square.rank - move.from_square.rank) == 2
        ):
            direction = 1 if move.piece.faction == Faction.WHITE else -1
            self.state.en_passant_target = Square(
                move.from_square.file, move.from_square.rank + direction
            )
        else:
            self.state.en_passant_target = None

        # Update clocks
        if move.captured_piece or move.piece.piece_type == PieceType.PAWN:
            self.state.halfmove_clock = 0
        else:
            self.state.halfmove_clock += 1

    def make_move(self, move: Move) -> bool:
        """Execute a move and advance turn."""
        legal_moves = self.get_legal_moves()
        if move not in legal_moves:
            # Try to find matching move
            for lm in legal_moves:
                if (
                    lm.from_square == move.from_square
                    and lm.to_square == move.to_square
                    and lm.promotion == move.promotion
                ):
                    move = lm
                    break
            else:
                return False

        self._apply_move_to_state(move)

        # Record position for repetition detection
        self.state.position_history.append(self.state.position_key())

        # Advance turn
        current_idx = TURN_ORDER.index(self.state.turn)
        self.state.turn = TURN_ORDER[(current_idx + 1) % 3]

        # Update fullmove number after cats move
        if self.state.turn == Faction.WHITE:
            self.state.fullmove_number += 1

        return True

    def make_move_algebraic(
        self, from_sq: str, to_sq: str, promotion: str = None
    ) -> bool:
        """Make a move using algebraic notation."""
        from_square = Square.from_algebraic(from_sq)
        to_square = Square.from_algebraic(to_sq)
        piece = self.state.get_piece_at(from_square)

        if not piece:
            return False

        promo = None
        if promotion:
            promo_map = {
                "q": PieceType.QUEEN,
                "r": PieceType.ROOK,
                "b": PieceType.BISHOP,
                "n": PieceType.KNIGHT,
            }
            promo = promo_map.get(promotion.lower())

        move = Move(piece, from_square, to_square, promotion=promo)
        return self.make_move(move)

    def get_game_result(self) -> Optional[str]:
        """Check for game end and return winner."""
        current = self.state.turn

        # Skip cats for checkmate/stalemate (they always have moves or the game continues)
        if current != Faction.CATS:
            legal_moves = self.get_legal_moves()

            if len(legal_moves) == 0:
                if self.is_in_check(current):
                    # Checkmate - other side wins
                    return (
                        Faction.BLACK.value
                        if current == Faction.WHITE
                        else Faction.WHITE.value
                    )
                else:
                    # Stalemate - cats win
                    return Faction.CATS.value

        # Fifty-move rule
        if self.state.halfmove_clock >= 100:
            return Faction.CATS.value

        # Threefold repetition
        position_key = self.state.position_key()
        if self.state.position_history.count(position_key) >= 3:
            return Faction.CATS.value

        # Insufficient material (simplified)
        if self._is_insufficient_material():
            return Faction.CATS.value

        return None

    def _is_insufficient_material(self) -> bool:
        """Check for insufficient material to checkmate."""
        white_pieces = [
            p
            for p in self.state.get_pieces(Faction.WHITE)
            if p.piece_type != PieceType.KING
        ]
        black_pieces = [
            p
            for p in self.state.get_pieces(Faction.BLACK)
            if p.piece_type != PieceType.KING
        ]

        # King vs King
        if not white_pieces and not black_pieces:
            return True

        # King + minor vs King
        if not black_pieces and len(white_pieces) == 1:
            if white_pieces[0].piece_type in [PieceType.BISHOP, PieceType.KNIGHT]:
                return True

        if not white_pieces and len(black_pieces) == 1:
            if black_pieces[0].piece_type in [PieceType.BISHOP, PieceType.KNIGHT]:
                return True

        return False

    def render_board(self) -> str:
        """Render the board as ASCII art."""
        lines = []
        disarmed = self.state.get_disarmed_squares()

        lines.append("  +---+---+---+---+---+---+---+---+")
        for rank in range(7, -1, -1):
            row = f"{rank + 1} |"
            for file in range(8):
                sq = Square(file, rank)
                piece = self.state.get_piece_at(sq)
                if piece:
                    symbol = piece.symbol()
                    if sq in disarmed and piece.piece_type != PieceType.CAT:
                        symbol = f"({symbol[0]})" if len(symbol) == 1 else symbol
                        row += f"{symbol}|"
                    else:
                        row += f" {symbol} |"
                else:
                    if sq in disarmed:
                        row += " . |"
                    else:
                        row += "   |"
            lines.append(row)
            lines.append("  +---+---+---+---+---+---+---+---+")

        lines.append("    a   b   c   d   e   f   g   h")
        lines.append(f"\nTurn: {self.state.turn.value}")

        if self.state.turn != Faction.CATS and self.is_in_check(self.state.turn):
            lines.append("CHECK!")

        return "\n".join(lines)
