#!/usr/bin/env python3
"""
Cat Chess CLI - Command-line interface for playing Cat Chess.
"""

import sys
import argparse
from typing import Optional

from .engine import CatChessEngine, Square, Move, Faction, PieceType
from .ai import CatChessAI, RandomAI


class CatChessCLI:
    """Command-line interface for Cat Chess."""

    def __init__(
        self,
        white_ai: bool = False,
        black_ai: bool = False,
        cat_ai: bool = True,
        ai_depth: int = 3,
    ):
        self.engine = CatChessEngine()
        self.ais = {}

        if white_ai:
            self.ais[Faction.WHITE] = CatChessAI(self.engine, Faction.WHITE, ai_depth)
        if black_ai:
            self.ais[Faction.BLACK] = CatChessAI(self.engine, Faction.BLACK, ai_depth)
        if cat_ai:
            self.ais[Faction.CATS] = CatChessAI(self.engine, Faction.CATS, ai_depth)

    def run(self):
        """Main game loop."""
        print("\n" + "=" * 50)
        print("   Welcome to CAT CHESS!")
        print("   A three-faction chess variant")
        print("=" * 50)
        print("\nRules:")
        print("- White and Black play standard chess")
        print("- Cats (C) create no-kill zones around them")
        print("- Pieces in no-kill zones are shown in (parentheses)")
        print("- Disarmed pieces can't capture or be captured")
        print("- If the game draws, CATS WIN!")
        print("\nCommands:")
        print("  <from><to>  - Make a move (e.g., e2e4)")
        print("  moves       - Show legal moves")
        print("  resign      - Resign the game")
        print("  help        - Show this help")
        print("  quit        - Exit the game")
        print()

        while True:
            # Check for game end
            result = self.engine.get_game_result()
            if result:
                self._show_game_end(result)
                break

            # Display board
            print(self.engine.render_board())

            current_turn = self.engine.state.turn

            # AI turn
            if current_turn in self.ais:
                print(f"\n{current_turn.value.capitalize()} AI is thinking...")
                move = self.ais[current_turn].get_best_move()
                if move:
                    print(f"AI plays: {move.to_algebraic()}")
                    self.engine.make_move(move)
                else:
                    print("AI has no legal moves!")
                    break
                continue

            # Human turn
            print(f"\n{current_turn.value.capitalize()}'s turn")
            command = input("Enter move or command: ").strip().lower()

            if not command:
                continue

            if command == "quit" or command == "exit":
                print("Thanks for playing!")
                break

            if command == "help":
                self._show_help()
                continue

            if command == "moves":
                self._show_legal_moves()
                continue

            if command == "resign":
                winner = self._handle_resign()
                self._show_game_end(winner)
                break

            # Try to parse as a move
            if not self._try_make_move(command):
                print("Invalid move or command. Type 'help' for assistance.")

    def _try_make_move(self, move_str: str) -> bool:
        """Try to parse and make a move from user input."""
        # Remove any spaces or dashes
        move_str = move_str.replace(" ", "").replace("-", "")

        # Handle promotion (e.g., e7e8q)
        promotion = None
        if len(move_str) == 5:
            promotion = move_str[4]
            move_str = move_str[:4]

        if len(move_str) != 4:
            return False

        try:
            from_sq = move_str[:2]
            to_sq = move_str[2:4]

            # Validate squares
            Square.from_algebraic(from_sq)
            Square.from_algebraic(to_sq)

            return self.engine.make_move_algebraic(from_sq, to_sq, promotion)
        except (ValueError, IndexError):
            return False

    def _show_legal_moves(self):
        """Display all legal moves."""
        moves = self.engine.get_legal_moves()
        if not moves:
            print("No legal moves available!")
            return

        print(f"\nLegal moves ({len(moves)} total):")

        # Group by piece
        by_from = {}
        for move in moves:
            key = move.from_square.to_algebraic()
            if key not in by_from:
                by_from[key] = []
            by_from[key].append(move)

        for from_sq, piece_moves in sorted(by_from.items()):
            piece = self.engine.state.get_piece_at(Square.from_algebraic(from_sq))
            piece_name = piece.piece_type.name if piece else "?"
            targets = [m.to_square.to_algebraic() for m in piece_moves]
            captures = [m for m in piece_moves if m.captured_piece]

            moves_str = ", ".join(targets)
            capture_note = f" ({len(captures)} captures)" if captures else ""
            print(f"  {piece_name} {from_sq}: {moves_str}{capture_note}")

    def _handle_resign(self) -> str:
        """Handle resignation."""
        current = self.engine.state.turn
        if current == Faction.WHITE:
            return Faction.BLACK.value
        elif current == Faction.BLACK:
            return Faction.WHITE.value
        else:
            # Cats can't really resign, but if they do, it's a draw (cats win!)
            return Faction.CATS.value

    def _show_game_end(self, winner: str):
        """Display game end message."""
        print("\n" + "=" * 50)
        print(self.engine.render_board())
        print("\n" + "=" * 50)

        if winner == Faction.CATS.value:
            print("   GAME OVER - THE CATS WIN!")
            print("   Peace has prevailed!")
        elif winner == Faction.WHITE.value:
            print("   GAME OVER - WHITE WINS!")
            print("   Checkmate!")
        else:
            print("   GAME OVER - BLACK WINS!")
            print("   Checkmate!")

        print("=" * 50 + "\n")

    def _show_help(self):
        """Show help message."""
        print("\n--- Cat Chess Help ---")
        print("\nHow to play:")
        print("  Enter moves in the format: <from><to>")
        print("  Examples: e2e4, g1f3, e7e8q (promotion)")
        print("\nBoard symbols:")
        print("  K/k = King (White/Black)")
        print("  Q/q = Queen")
        print("  R/r = Rook")
        print("  B/b = Bishop")
        print("  N/n = Knight")
        print("  P/p = Pawn")
        print("  C   = Cat")
        print("  (X) = Disarmed piece (in cat's no-kill zone)")
        print("  .   = Empty square in no-kill zone")
        print("\nCommands:")
        print("  moves  - Show all legal moves")
        print("  resign - Resign the game")
        print("  help   - Show this help")
        print("  quit   - Exit the game")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cat Chess - A three-faction chess variant"
    )
    parser.add_argument("--white-ai", action="store_true", help="Let AI play white")
    parser.add_argument("--black-ai", action="store_true", help="Let AI play black")
    parser.add_argument(
        "--no-cat-ai", action="store_true", help="Disable cat AI (player controls cats)"
    )
    parser.add_argument("--ai-vs-ai", action="store_true", help="Watch AI vs AI game")
    parser.add_argument(
        "--depth", type=int, default=3, help="AI search depth (default: 3)"
    )

    args = parser.parse_args()

    white_ai = args.white_ai or args.ai_vs_ai
    black_ai = args.black_ai or args.ai_vs_ai
    cat_ai = not args.no_cat_ai

    cli = CatChessCLI(
        white_ai=white_ai, black_ai=black_ai, cat_ai=cat_ai, ai_depth=args.depth
    )
    cli.run()


if __name__ == "__main__":
    main()
