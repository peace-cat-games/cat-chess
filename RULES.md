# Cat Chess Rules

Cat Chess is a three-faction chess variant where Cats act as peacekeepers. White and Black wage war normally, but Cats create "no-kill bubbles" where captures cannot occur and checks cannot be given.

**If the war stalls (draw), Cats win.**

## 1. Board, Pieces, and Starting Position

- **Board**: Standard 8x8 chessboard
- **Factions**: White, Black, and Cats

### Starting Position

- **White**: Standard chess setup on ranks 1-2
- **Black**: Standard chess setup on ranks 7-8
- **Cats**: Two Cat pieces
  - Cat #1 starts on **a4**
  - Cat #2 starts on **h5**

### Cat Piece Properties

- Move like Kings (one square in any direction)
- Cannot capture or be captured
- Cannot occupy an occupied square
- Can move onto squares attacked by armed pieces (ignore danger)

## 2. Turn Order

Fixed cyclical order: **White -> Black -> Cats**

## 3. Movement & Captures

### White & Black

- Move exactly like standard chess
- Captures allowed normally, except when forbidden by Cat no-kill rule

### Cats

- Move like Kings (one square any direction)
- Cannot capture
- Cannot be captured
- Cannot stand on occupied squares

## 4. The Cat "No-Kill Zone"

Cats enforce peace in a **1-square Chebyshev radius** around themselves.

A square S is in the no-kill zone if `max(|dx|, |dy|) <= 1` from any Cat (including the Cat's own square).

### Disarm Effect Inside the Zone

Any non-cat piece (White or Black) located on a no-kill square becomes **disarmed**.

A disarmed piece:
- **CAN** move normally (non-capturing moves)
- **CANNOT** capture
- **CANNOT** be captured
- **CANNOT** attack or threaten any square
- **CANNOT** give check
- **CANNOT** deliver mate
- **CAN** block sliding lines of attack (still occupies its square)
- **CAN** exit the zone (becomes armed again)

Disarm is purely positional and reevaluated every turn.

## 5. Capture Restriction

**Fundamental Rule**: A capture is illegal if **either**:
1. The capturing piece's origin square is in a no-kill zone, OR
2. The capturing piece's destination (target) square is in a no-kill zone

## 6. Attacks, Check, and Checkmate

### Attacks

Only **armed** pieces (not in a no-kill zone) produce attack rays. Disarmed pieces have empty attack sets.

### Check

A king is in check if at least one armed enemy piece attacks its square. Disarmed pieces and Cats never give check.

### King & Cat Legality

- Kings cannot move onto squares attacked by armed enemy pieces
- Cats may move into attacked squares freely but cannot occupy occupied squares

### Checkmate

A side is checkmated if:
1. Its King is under attack by at least one armed enemy piece
2. It has no legal move to escape (move king, block, or disarm attacker)

### Stalemate & Draws

**Any draw results in Cats winning.**

Draw types:
- Stalemate
- Threefold repetition
- Fifty-move rule
- Insufficient material
- No possible progress

## 7. Win Conditions

| Outcome | Winner |
|---------|--------|
| White delivers checkmate | White |
| Black delivers checkmate | Black |
| Any draw condition | Cats |
