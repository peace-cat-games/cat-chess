# Cat Chess

A three-faction chess variant where Cats act as peacekeepers.

## Quick Start

```bash
# Install
pip install -e .

# Play (you control Cats, AI controls White and Black)
cat-chess

# Watch AI vs AI
cat-chess --ai-vs-ai

# Play as White
cat-chess --play-white
```

## How to Play

### The Basics

Cat Chess is played on a standard 8x8 board with three factions:

- **White** - Standard chess pieces, plays first
- **Black** - Standard chess pieces, plays second
- **Cats** - Two special pieces that enforce peace, play third

Turn order: White -> Black -> Cats -> White -> ...

### The Cats

Two Cat pieces start on the board at **a4** and **h5**. Cats:

- Move like Kings (one square in any direction)
- Cannot capture or be captured
- Cannot occupy squares with other pieces
- Create a **no-kill zone** in a 3x3 area around themselves

### No-Kill Zones

Any piece inside a Cat's zone becomes **disarmed**:

- Can move normally
- Cannot capture
- Cannot be captured
- Cannot give check
- Shown as `(P)` or `(K)` etc. on the board
- Empty squares in zones shown as `.`

A capture is illegal if either the attacker OR the target is in a no-kill zone.

### Winning

| Condition | Winner |
|-----------|--------|
| Checkmate White | Black wins |
| Checkmate Black | White wins |
| Any draw (stalemate, repetition, 50-move rule) | **Cats win** |

The Cats win if the war stalls!

## Controls

| Command | Action |
|---------|--------|
| `e2e4` | Move piece from e2 to e4 |
| `e7e8q` | Pawn promotion (to Queen) |
| `moves` | Show all legal moves |
| `resign` | Resign the game |
| `help` | Show help |
| `quit` | Exit |

## Command Line Options

```
--play-white   Play as White (instead of Cats)
--play-black   Play as Black (instead of Cats)
--cat-ai       Let AI control Cats
--ai-vs-ai     Watch AI play against itself
--depth N      Set AI difficulty (default: 3)
```

## Board Symbols

| Symbol | Piece |
|--------|-------|
| `K` / `k` | King (White/Black) |
| `Q` / `q` | Queen |
| `R` / `r` | Rook |
| `B` / `b` | Bishop |
| `N` / `n` | Knight |
| `P` / `p` | Pawn |
| `C` | Cat |
| `(X)` | Disarmed piece |
| `.` | Empty square in no-kill zone |

## Strategy Tips

- **As White/Black**: Avoid letting your attacking pieces get too close to Cats
- **As Cats**: Position yourself to protect threatened pieces or block checkmates
- **Draws favor Cats**: If you're losing, seek a draw - the Cats will win!
- **Cats can't be trapped**: They can move through attacked squares freely
