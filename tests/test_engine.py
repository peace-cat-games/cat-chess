"""
Unit tests for Cat Chess engine.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine import (
    CatChessEngine,
    Square,
    Piece,
    Move,
    GameState,
    Faction,
    PieceType,
    TURN_ORDER,
)


class TestStartingPosition:
    """Test 9.1 - Starting Position"""

    def test_white_pieces_setup(self):
        """Correct white pieces on ranks 1-2."""
        engine = CatChessEngine()

        # Back rank
        assert engine.state.get_piece_at(Square(0, 0)).piece_type == PieceType.ROOK
        assert engine.state.get_piece_at(Square(1, 0)).piece_type == PieceType.KNIGHT
        assert engine.state.get_piece_at(Square(2, 0)).piece_type == PieceType.BISHOP
        assert engine.state.get_piece_at(Square(3, 0)).piece_type == PieceType.QUEEN
        assert engine.state.get_piece_at(Square(4, 0)).piece_type == PieceType.KING
        assert engine.state.get_piece_at(Square(5, 0)).piece_type == PieceType.BISHOP
        assert engine.state.get_piece_at(Square(6, 0)).piece_type == PieceType.KNIGHT
        assert engine.state.get_piece_at(Square(7, 0)).piece_type == PieceType.ROOK

        # Pawns
        for file in range(8):
            piece = engine.state.get_piece_at(Square(file, 1))
            assert piece.piece_type == PieceType.PAWN
            assert piece.faction == Faction.WHITE

    def test_black_pieces_setup(self):
        """Correct black pieces on ranks 7-8."""
        engine = CatChessEngine()

        # Back rank
        assert engine.state.get_piece_at(Square(0, 7)).piece_type == PieceType.ROOK
        assert engine.state.get_piece_at(Square(4, 7)).piece_type == PieceType.KING

        # Pawns
        for file in range(8):
            piece = engine.state.get_piece_at(Square(file, 6))
            assert piece.piece_type == PieceType.PAWN
            assert piece.faction == Faction.BLACK

    def test_cats_on_a4_and_h5(self):
        """Cats on a4 and h5."""
        engine = CatChessEngine()

        cat1 = engine.state.get_piece_at(Square.from_algebraic("a4"))
        cat2 = engine.state.get_piece_at(Square.from_algebraic("h5"))

        assert cat1 is not None
        assert cat1.piece_type == PieceType.CAT
        assert cat1.faction == Faction.CATS

        assert cat2 is not None
        assert cat2.piece_type == PieceType.CAT
        assert cat2.faction == Faction.CATS

    def test_turn_starts_white(self):
        """Turn starts as white."""
        engine = CatChessEngine()
        assert engine.state.turn == Faction.WHITE

    def test_turn_order(self):
        """Turn order is white -> black -> cats."""
        assert TURN_ORDER == [Faction.WHITE, Faction.BLACK, Faction.CATS]


class TestDisarmLogic:
    """Test 9.2 - Disarm Logic"""

    def test_squares_adjacent_to_cats_are_disarmed(self):
        """Squares adjacent to Cats flagged correctly."""
        engine = CatChessEngine()
        disarmed = engine.state.get_disarmed_squares()

        # Cat on a4 (file=0, rank=3)
        # Adjacent squares: a3, a5, b3, b4, b5 plus a4 itself
        assert Square.from_algebraic("a3") in disarmed
        assert Square.from_algebraic("a4") in disarmed
        assert Square.from_algebraic("a5") in disarmed
        assert Square.from_algebraic("b3") in disarmed
        assert Square.from_algebraic("b4") in disarmed
        assert Square.from_algebraic("b5") in disarmed

        # Cat on h5 (file=7, rank=4)
        assert Square.from_algebraic("g4") in disarmed
        assert Square.from_algebraic("g5") in disarmed
        assert Square.from_algebraic("g6") in disarmed
        assert Square.from_algebraic("h4") in disarmed
        assert Square.from_algebraic("h5") in disarmed
        assert Square.from_algebraic("h6") in disarmed

    def test_pieces_inside_zone_are_disarmed(self):
        """Pieces inside zone become disarmed."""
        engine = CatChessEngine()

        # Move a piece into cat zone and verify it's disarmed
        # b3 is in cat zone (adjacent to cat on a4)
        assert engine.state.is_disarmed(Square.from_algebraic("b3"))

        # e4 is not in cat zone
        assert not engine.state.is_disarmed(Square.from_algebraic("e4"))

    def test_disarm_recalculated_after_cat_move(self):
        """Disarm zones update when cats move."""
        engine = CatChessEngine()

        # Initially b4 is disarmed (cat on a4)
        assert engine.state.is_disarmed(Square.from_algebraic("b4"))

        # Make white, black, then cat moves
        engine.make_move_algebraic("e2", "e4")  # White
        engine.make_move_algebraic("e7", "e5")  # Black

        # Move cat from a4 to a5
        cat = engine.state.get_piece_at(Square.from_algebraic("a4"))
        move = Move(cat, Square.from_algebraic("a4"), Square.from_algebraic("b5"))
        engine.make_move(move)

        # Now b4 should be in zone (adjacent to b5)
        assert engine.state.is_disarmed(Square.from_algebraic("b4"))
        # a3 should no longer be disarmed
        assert not engine.state.is_disarmed(Square.from_algebraic("a3"))


class TestCaptureRestriction:
    """Test 9.3 - Capture Restriction"""

    def test_capture_blocked_if_origin_in_no_kill_zone(self):
        """Captures blocked if origin is in no-kill zone."""
        engine = CatChessEngine()

        # Set up a position where a piece is in cat zone and could capture
        # We need to create a custom position for this test
        engine.state.board.clear()

        # Place white queen on b4 (in cat zone of cat on a4)
        cat_sq = Square.from_algebraic("a4")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        queen_sq = Square.from_algebraic("b4")
        engine.state.board[queen_sq] = Piece(PieceType.QUEEN, Faction.WHITE, queen_sq)

        # Place black pawn on d4 (outside cat zone, capturable target)
        pawn_sq = Square.from_algebraic("d4")
        engine.state.board[pawn_sq] = Piece(PieceType.PAWN, Faction.BLACK, pawn_sq)

        # Place kings
        wk_sq = Square.from_algebraic("e1")
        bk_sq = Square.from_algebraic("e8")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        # Get legal moves for white queen
        moves = engine.get_legal_moves()
        queen_moves = [m for m in moves if m.piece.piece_type == PieceType.QUEEN]

        # Queen should be able to move but not capture (it's disarmed)
        captures = [m for m in queen_moves if m.captured_piece is not None]
        assert len(captures) == 0

    def test_capture_blocked_if_destination_in_no_kill_zone(self):
        """Captures blocked if destination is in no-kill zone."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Place cat on e4
        cat_sq = Square.from_algebraic("e4")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # Place white rook on d1 (outside cat zone)
        rook_sq = Square.from_algebraic("d1")
        engine.state.board[rook_sq] = Piece(PieceType.ROOK, Faction.WHITE, rook_sq)

        # Place black pawn on d4 (inside cat zone - adjacent to e4)
        pawn_sq = Square.from_algebraic("d4")
        engine.state.board[pawn_sq] = Piece(PieceType.PAWN, Faction.BLACK, pawn_sq)

        # Place kings far away
        wk_sq = Square.from_algebraic("a1")
        bk_sq = Square.from_algebraic("h8")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        moves = engine.get_legal_moves()
        rook_moves = [m for m in moves if m.piece.piece_type == PieceType.ROOK]
        captures = [m for m in rook_moves if m.captured_piece is not None]

        # Rook cannot capture pawn on d4 because it's in cat zone
        assert len(captures) == 0

    def test_capture_allowed_outside_zone(self):
        """Captures allowed outside the no-kill zone."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Place cat on a1 (corner, away from action)
        cat_sq = Square.from_algebraic("a1")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # Place white rook on e4 (outside cat zone)
        rook_sq = Square.from_algebraic("e4")
        engine.state.board[rook_sq] = Piece(PieceType.ROOK, Faction.WHITE, rook_sq)

        # Place black pawn on e7 (outside cat zone)
        pawn_sq = Square.from_algebraic("e7")
        engine.state.board[pawn_sq] = Piece(PieceType.PAWN, Faction.BLACK, pawn_sq)

        # Place kings far from rook's path
        wk_sq = Square.from_algebraic("h1")
        bk_sq = Square.from_algebraic("h8")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        moves = engine.get_legal_moves()
        rook_captures = [
            m
            for m in moves
            if m.piece.piece_type == PieceType.ROOK and m.captured_piece
        ]

        # Rook can capture pawn on e7
        assert len(rook_captures) == 1
        assert rook_captures[0].to_square == pawn_sq


class TestAttackLogic:
    """Test 9.4 - Attack Logic"""

    def test_disarmed_pieces_have_empty_attack_sets(self):
        """Disarmed pieces have empty attack sets."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Place cat on e4
        cat_sq = Square.from_algebraic("e4")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # Place white queen on d4 (inside cat zone - adjacent to e4)
        queen_sq = Square.from_algebraic("d4")
        engine.state.board[queen_sq] = Piece(PieceType.QUEEN, Faction.WHITE, queen_sq)

        # Place kings far away
        wk_sq = Square.from_algebraic("a8")
        bk_sq = Square.from_algebraic("h8")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        # Get attacked squares by white
        attacked = engine.get_attacked_squares(Faction.WHITE)

        # d4 queen is disarmed, so only king attacks matter
        # d5, d6, etc. should NOT be attacked by disarmed queen
        d5 = Square.from_algebraic("d5")
        d6 = Square.from_algebraic("d6")
        assert d5 not in attacked
        assert d6 not in attacked

    def test_armed_pieces_generate_correct_attacks(self):
        """Armed pieces generate correct attack rays."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Place cat far away in corner
        cat_sq = Square.from_algebraic("a1")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # Place white rook on e4 (outside cat zone)
        rook_sq = Square.from_algebraic("e4")
        engine.state.board[rook_sq] = Piece(PieceType.ROOK, Faction.WHITE, rook_sq)

        # Place kings far from the rook
        wk_sq = Square.from_algebraic("h1")
        bk_sq = Square.from_algebraic("h8")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        attacked = engine.get_attacked_squares(Faction.WHITE)

        # Rook on e4 should attack e-file and 4th rank
        assert Square.from_algebraic("e5") in attacked
        assert Square.from_algebraic("e6") in attacked
        assert Square.from_algebraic("e7") in attacked
        assert Square.from_algebraic("e8") in attacked
        # Rook attacks along rank 4 (d4, c4, b4, a4)
        assert Square.from_algebraic("d4") in attacked
        assert Square.from_algebraic("c4") in attacked


class TestCheckAndBlocking:
    """Test 9.5 - Checking & Blocking"""

    def test_disarmed_pieces_can_block_sliding_checks(self):
        """Disarmed pieces can block sliding checks."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Place cat to create disarm zone
        cat_sq = Square.from_algebraic("d4")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # White king on e1
        wk_sq = Square.from_algebraic("e1")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)

        # Black queen on e8 giving check along e-file
        bq_sq = Square.from_algebraic("e8")
        engine.state.board[bq_sq] = Piece(PieceType.QUEEN, Faction.BLACK, bq_sq)

        # White bishop on e4 (in cat zone, disarmed) - can block
        wb_sq = Square.from_algebraic("e4")
        engine.state.board[wb_sq] = Piece(PieceType.BISHOP, Faction.WHITE, wb_sq)

        # Black king
        bk_sq = Square.from_algebraic("a8")
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        # The disarmed bishop on e4 blocks the check from e8 to e1
        # So white king is NOT in check
        assert not engine.is_in_check(Faction.WHITE)

    def test_disarmed_pieces_never_give_check(self):
        """Disarmed pieces never give check."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Place cat on e4
        cat_sq = Square.from_algebraic("e4")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # White king on a1
        wk_sq = Square.from_algebraic("a1")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)

        # Black queen on d4 (in cat zone, disarmed)
        bq_sq = Square.from_algebraic("d4")
        engine.state.board[bq_sq] = Piece(PieceType.QUEEN, Faction.BLACK, bq_sq)

        # Black king
        bk_sq = Square.from_algebraic("h8")
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        # Queen could normally check king on a1 via diagonal
        # But queen is disarmed, so no check
        assert not engine.is_in_check(Faction.WHITE)


class TestKingCatSafety:
    """Test 9.6 - King/Cat Safety Rules"""

    def test_king_forbidden_from_attacked_squares(self):
        """King forbidden from moving onto attacked squares."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Cat far away
        cat_sq = Square.from_algebraic("a1")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # White king on e1
        wk_sq = Square.from_algebraic("e1")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)

        # Black rook on f8 (attacks f-file)
        br_sq = Square.from_algebraic("f8")
        engine.state.board[br_sq] = Piece(PieceType.ROOK, Faction.BLACK, br_sq)

        # Black king
        bk_sq = Square.from_algebraic("h8")
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        moves = engine.get_legal_moves()
        king_moves = [m for m in moves if m.piece.piece_type == PieceType.KING]
        destinations = [m.to_square.to_algebraic() for m in king_moves]

        # King cannot move to f1 or f2 (attacked by rook)
        assert "f1" not in destinations
        assert "f2" not in destinations
        # King can move to d1, d2, e2
        assert "d1" in destinations
        assert "d2" in destinations
        assert "e2" in destinations

    def test_cat_can_move_to_attacked_squares(self):
        """Cat allowed to move onto attacked squares."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Cat on e4
        cat_sq = Square.from_algebraic("e4")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # White rook on a5 (attacks 5th rank including e5)
        wr_sq = Square.from_algebraic("a5")
        engine.state.board[wr_sq] = Piece(PieceType.ROOK, Faction.WHITE, wr_sq)

        # Kings
        wk_sq = Square.from_algebraic("h1")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)
        bk_sq = Square.from_algebraic("h8")
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        # Set turn to cats
        engine.state.turn = Faction.CATS

        moves = engine.get_legal_moves()
        cat_moves = [m for m in moves if m.piece.piece_type == PieceType.CAT]
        destinations = [m.to_square.to_algebraic() for m in cat_moves]

        # Cat can move to e5 even though it's attacked
        assert "e5" in destinations

    def test_cat_cannot_move_to_occupied_squares(self):
        """Cat cannot move onto occupied squares."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Cat on e4
        cat_sq = Square.from_algebraic("e4")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # White pawn on e5
        wp_sq = Square.from_algebraic("e5")
        engine.state.board[wp_sq] = Piece(PieceType.PAWN, Faction.WHITE, wp_sq)

        # Kings
        wk_sq = Square.from_algebraic("h1")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)
        bk_sq = Square.from_algebraic("h8")
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        engine.state.turn = Faction.CATS

        moves = engine.get_legal_moves()
        cat_moves = [m for m in moves if m.piece.piece_type == PieceType.CAT]
        destinations = [m.to_square.to_algebraic() for m in cat_moves]

        # Cat cannot move to e5 (occupied)
        assert "e5" not in destinations
        # Cat can move to other adjacent empty squares
        assert "d4" in destinations or "f4" in destinations


class TestDrawCatVictory:
    """Test 9.7 - Draw -> Cat Victory"""

    def test_stalemate_results_in_cats_win(self):
        """Stalemate results in cats victory."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Cat far away
        cat_sq = Square.from_algebraic("a1")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # White king on a8 (cornered)
        wk_sq = Square.from_algebraic("a8")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)

        # Black queen on b6 (controls escape squares but not giving check)
        bq_sq = Square.from_algebraic("b6")
        engine.state.board[bq_sq] = Piece(PieceType.QUEEN, Faction.BLACK, bq_sq)

        # Black king on c7
        bk_sq = Square.from_algebraic("c7")
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        engine.state.turn = Faction.WHITE

        # White king is not in check but has no legal moves (stalemate)
        assert not engine.is_in_check(Faction.WHITE)
        assert len(engine.get_legal_moves()) == 0

        result = engine.get_game_result()
        assert result == Faction.CATS.value

    def test_insufficient_material_results_in_cats_win(self):
        """Insufficient material results in cats victory."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Cat
        cat_sq = Square.from_algebraic("a1")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # Just two kings
        wk_sq = Square.from_algebraic("e1")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)
        bk_sq = Square.from_algebraic("e8")
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        result = engine.get_game_result()
        assert result == Faction.CATS.value

    def test_checkmate_white_wins(self):
        """Checkmate by white results in white victory."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Cat far away
        cat_sq = Square.from_algebraic("a1")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # Black king on h8
        bk_sq = Square.from_algebraic("h8")
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        # White queen on g7 (giving check)
        wq_sq = Square.from_algebraic("g7")
        engine.state.board[wq_sq] = Piece(PieceType.QUEEN, Faction.WHITE, wq_sq)

        # White king on f6 (supporting queen)
        wk_sq = Square.from_algebraic("f6")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)

        engine.state.turn = Faction.BLACK

        # Black is in checkmate
        assert engine.is_in_check(Faction.BLACK)
        assert len(engine.get_legal_moves()) == 0

        result = engine.get_game_result()
        assert result == Faction.WHITE.value


class TestMoveGeneration:
    """Additional tests for move generation."""

    def test_pawn_moves(self):
        """Test pawn move generation."""
        engine = CatChessEngine()

        # e2-e4 should be legal (double move)
        moves = engine.get_legal_moves()
        e2_moves = [m for m in moves if m.from_square.to_algebraic() == "e2"]

        destinations = [m.to_square.to_algebraic() for m in e2_moves]
        assert "e3" in destinations
        assert "e4" in destinations

    def test_knight_moves(self):
        """Test knight move generation."""
        engine = CatChessEngine()

        moves = engine.get_legal_moves()
        g1_moves = [m for m in moves if m.from_square.to_algebraic() == "g1"]

        destinations = [m.to_square.to_algebraic() for m in g1_moves]
        assert "f3" in destinations
        assert "h3" in destinations

    def test_turn_advances_correctly(self):
        """Test that turns advance in correct order."""
        engine = CatChessEngine()

        assert engine.state.turn == Faction.WHITE
        engine.make_move_algebraic("e2", "e4")
        assert engine.state.turn == Faction.BLACK
        engine.make_move_algebraic("e7", "e5")
        assert engine.state.turn == Faction.CATS


class TestCatPassMove:
    """Test cat pass (stay in place) functionality."""

    def test_cat_can_pass(self):
        """Cat can choose to stay in place (pass move)."""
        engine = CatChessEngine()

        engine.state.board.clear()

        # Cat on e4
        cat_sq = Square.from_algebraic("e4")
        engine.state.board[cat_sq] = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        engine.state.cat_positions = [cat_sq]

        # Kings
        wk_sq = Square.from_algebraic("h1")
        engine.state.board[wk_sq] = Piece(PieceType.KING, Faction.WHITE, wk_sq)
        bk_sq = Square.from_algebraic("h8")
        engine.state.board[bk_sq] = Piece(PieceType.KING, Faction.BLACK, bk_sq)

        engine.state.turn = Faction.CATS

        moves = engine.get_legal_moves()
        pass_moves = [m for m in moves if m.is_pass()]

        # Should have exactly one pass move per cat
        assert len(pass_moves) == 1
        assert pass_moves[0].from_square == cat_sq
        assert pass_moves[0].to_square == cat_sq

    def test_pass_move_algebraic_notation(self):
        """Pass move shows correct algebraic notation."""
        cat_sq = Square.from_algebraic("a4")
        cat = Piece(PieceType.CAT, Faction.CATS, cat_sq)
        pass_move = Move(cat, cat_sq, cat_sq)

        assert pass_move.to_algebraic() == "a4(pass)"
        assert pass_move.is_pass()

    def test_pass_move_preserves_position(self):
        """Pass move keeps cat in same position."""
        engine = CatChessEngine()

        # Play white and black moves to get to cats turn
        engine.make_move_algebraic("e2", "e4")
        engine.make_move_algebraic("e7", "e5")

        assert engine.state.turn == Faction.CATS

        original_positions = list(engine.state.cat_positions)

        # Make a pass move for the cat on a4
        cat = engine.state.get_piece_at(Square.from_algebraic("a4"))
        pass_move = Move(cat, cat.square, cat.square)
        engine.make_move(pass_move)

        # Cat position should be unchanged
        assert Square.from_algebraic("a4") in engine.state.cat_positions

        # Turn should have advanced
        assert engine.state.turn == Faction.WHITE

    def test_pass_move_advances_turn(self):
        """Pass move advances the turn correctly."""
        engine = CatChessEngine()

        engine.make_move_algebraic("e2", "e4")
        engine.make_move_algebraic("e7", "e5")

        assert engine.state.turn == Faction.CATS

        cat = engine.state.get_piece_at(Square.from_algebraic("a4"))
        pass_move = Move(cat, cat.square, cat.square)
        engine.make_move(pass_move)

        assert engine.state.turn == Faction.WHITE

    def test_two_cats_can_each_pass(self):
        """Both cats have pass moves available."""
        engine = CatChessEngine()

        engine.make_move_algebraic("e2", "e4")
        engine.make_move_algebraic("e7", "e5")

        moves = engine.get_legal_moves()
        pass_moves = [m for m in moves if m.is_pass()]

        # Should have 2 pass moves (one per cat)
        assert len(pass_moves) == 2

        pass_squares = {m.from_square.to_algebraic() for m in pass_moves}
        assert pass_squares == {"a4", "h5"}


class TestSquareClass:
    """Test Square utility class."""

    def test_algebraic_conversion(self):
        """Test algebraic notation conversion."""
        sq = Square.from_algebraic("e4")
        assert sq.file == 4
        assert sq.rank == 3
        assert sq.to_algebraic() == "e4"

    def test_chebyshev_distance(self):
        """Test Chebyshev distance calculation."""
        sq1 = Square.from_algebraic("a1")
        sq2 = Square.from_algebraic("h8")
        assert sq1.chebyshev_distance(sq2) == 7

        sq3 = Square.from_algebraic("e4")
        sq4 = Square.from_algebraic("f5")
        assert sq3.chebyshev_distance(sq4) == 1

    def test_validity(self):
        """Test square validity."""
        assert Square(0, 0).is_valid()
        assert Square(7, 7).is_valid()
        assert not Square(-1, 0).is_valid()
        assert not Square(8, 0).is_valid()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
