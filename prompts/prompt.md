Purrrrfect.
Here is the **clean, authoritative, final Cat Chess Claude prompt**, fully rewritten with all corrected laws:

---

# **CLAUDE PROMPT — COMPLETE CAT CHESS SPEC (THREE FACTIONS · NO-KILL ZONES · DRAW = CAT WIN)**

Implement **Cat Chess**, a three-faction chess variant where Cats act as peacekeepers. White and Black wage war normally, but Cats create “no-kill bubbles” where captures cannot occur and checks cannot be given.
If the war stalls (draw), **Cats win**.

This prompt defines the complete rules, engine behavior, win conditions, disarm effects, and required deliverables.

---

# **1. BOARD, PIECES, AND STARTING POSITION**

* Board: standard 8×8 chessboard.
* Factions:

  * `"white"`
  * `"black"`
  * `"cats"`

### **1.1 White & Black Starting Pieces**

Use standard chess starting position:

* White pieces on ranks 1–2.
* Black pieces on ranks 7–8.

### **1.2 Cats**

The cat faction controls two Cat pieces:

* Cat #1 starts on **a4**
* Cat #2 starts on **h5**

Cats:

* Move like Kings (one square in any direction).
* Cannot capture or be captured.
* Cannot occupy an occupied square.
* Can move onto squares attacked by armed pieces (they ignore “danger”).

---

# **2. TURN ORDER**

Use fixed cyclical order:

```
TURN_ORDER = ["white", "black", "cats"]
```

All rules and legality checks must respect this exact turn cycle.

---

# **3. MOVEMENT & CAPTURES**

## **3.1 White & Black Movement**

* Move exactly like standard chess.
* **Captures are allowed normally**, except when forbidden by the Cat no-kill rule (see Section 4).
* Capture requires moving onto the enemy’s square (standard chess capture).

## **3.2 Cats**

* Move like Kings.
* Cannot capture.
* Cannot be captured.
* Cannot stand on occupied squares.
* May enter squares attacked by armed pieces.

---

# **4. THE CAT “NO-KILL ZONE” — CORE RULE**

Cats enforce peace in a 1-square Chebyshev radius around themselves.

Define:
A square **S** is in the no-kill zone if
`max(|dx|, |dy|) = 1` from any Cat.

This includes:

* orthogonal adjacency,
* diagonal adjacency.

## **4.1 DISARM EFFECT INSIDE THE ZONE**

Any non-cat piece (White or Black) located on a no-kill square becomes **disarmed**.

A disarmed piece:

* **CAN move normally** (non-capturing moves).
* **CANNOT capture**.
* **CANNOT be captured**.
* **CANNOT attack or threaten any square.**
* **CANNOT give check**.
* **CANNOT deliver mate**.
* **CAN block** sliding lines of attack because it still occupies its square.
* **CAN exit the zone**, after which it becomes armed again.

Disarm is purely positional and reevaluated every turn.

---

# **5. CAPTURE RESTRICTION**

### **THE FUNDAMENTAL RULE:**

A capture is **illegal** if **either**:

1. The capturing piece’s **origin square** is in a no-kill zone **OR**
2. The capturing piece’s **destination (target) square** is in a no-kill zone.

Formally:

```
A capture move is legal
iff origin ∉ no-kill-zone  AND  destination ∉ no-kill-zone.
```

Thus:

* In cat-adjacent squares → no one can kill or be killed.
* Outside those squares → normal captures occur.

---

# **6. ATTACKS, CHECK, AND CHECKMATE**

### **6.1 Attacks**

Only **armed** pieces (i.e., not in a no-kill zone) produce attack rays.

Disarmed pieces have **empty attack sets**.

### **6.2 Check**

A king is in check if at least one **armed** enemy piece attacks its square.

Disarmed pieces never give check.

Cats never give check.

### **6.3 King & Cat Legality**

* Kings:

  * Cannot move onto a square attacked by any armed enemy piece.
* Cats:

  * May move into attacked squares freely.
  * Cannot occupy occupied squares.

### **6.4 Checkmate**

A side (White or Black) is checkmated if:

1. Its King is under attack by at least one **armed** enemy piece,
2. And it has no legal move that:

   * Moves the King to an unattacked empty square, or
   * Blocks the attack using any piece (armed or disarmed), or
   * Renders the attacker unable to attack (e.g., attacker enters no-kill zone and becomes disarmed).

### **6.5 Stalemate & Other Draws**

Any draw results in **Cats winning**.

Draw types include:

* Stalemate.
* Threefold repetition.
* Insufficient armed pieces to ever give check.
* No possible progress (e.g., permanent no-kill blockade).
* Any classical chess draw rule you implement.

Outcome scoring:

* White delivers mate → White wins.
* Black delivers mate → Black wins.
* **Any draw → CATS WIN.**

---

# **7. ENGINE REQUIREMENTS**

### **7.1 State Representation**

Your engine must track:

```json
{
  "board": ...,
  "turn": "white"|"black"|"cats",
  "cat_positions": [(file,rank),(file,rank)],
  "disarmed_squares": set_of_squares,
  "halfmove_clock": int,
  "fullmove_number": int
}
```

You may extend this freely.

### **7.2 Disarm Recalculation**

After every move, recompute disarm squares based on cat adjacency.

### **7.3 Legal Move Generation**

* Generate pseudo-legal moves for the current faction.
* Apply:

  * capture ban within no-kill zone,
  * king-check legality for white/black,
  * occupancy restrictions (no two pieces share a square),
  * cat rules.

### **7.4 Outcome Detection**

Return winners as:

* `"white"`
* `"black"`
* `"cats"`

---

# **8. AI REQUIREMENTS (OPTIONAL)**

Implement simple heuristic AIs:

### **White & Black AI**

* Try to achieve checkmate.
* Avoid entering cat zones unnecessarily (gets disarmed).
* Maximize armed piece activity.

### **Cat AI**

* Goal: **maximize draw likelihood.**
* Key strategies:

  * Move into hotspots.
  * Expand no-kill zones around key attackers.
  * Shut down mating nets.
  * Create blockades and equilibrium.
  * Neutralize both sides symmetrically.

### **Scores**

* White mate → White +∞, Black −∞, Cats −∞.
* Black mate → Black +∞, White −∞, Cats −∞.
* Draw → Cats +∞, White 0, Black 0.

---

# **9. UNIT TEST REQUIREMENTS**

Include tests for:

### **9.1 Starting Position**

* Correct white/black setup.
* Cats on a4 and h5.
* Turn starts = `"white"`.

### **9.2 Disarm Logic**

* Squares adjacent to Cats flagged correctly.
* Pieces inside zone become disarmed.

### **9.3 Capture Restriction**

* Captures blocked if origin or destination is in no-kill zone.
* Captures allowed outside the zone.

### **9.4 Attack Logic**

* Disarmed pieces have empty attack sets.
* Armed pieces generate correct attack rays.

### **9.5 Checking & Blocking**

* Disarmed pieces can block sliding checks.
* Disarmed pieces never give check.

### **9.6 King/Cat Safety Rules**

* King forbidden from moving onto attacked squares.
* Cat allowed to move onto attacked squares.
* Cat cannot move onto occupied squares.

### **9.7 Draw → Cat Victory**

* Stalemate → result `"cats"`.
* Repetition → `"cats"`.
* Dead positions → `"cats"`.

---

# **10. DELIVERABLES CLAUDE MUST OUTPUT**

1. **RULES.md** — full ruleset as above.
2. **Complete engine implementation** (Python default).
3. **Move generator** with:

   * no-kill capture ban,
   * disarm logic,
   * three-faction turn order,
   * armed-only attack maps.
4. **Outcome evaluator** returning `"white"|"black"|"cats"`.
5. **Unit test suite**.
6. (Optional) ASCII board renderer.
7. (Optional) Web API skeleton.

---

If you want next, I can produce:

* CAT-FEN string format
* Puzzles demonstrating cat interventions
* Example games
* an SVG Cat piece set

Just purr. 🐈
