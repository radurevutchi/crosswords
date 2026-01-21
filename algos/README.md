# Custom Backtracking Crossword Generator

A greedy placement algorithm with heuristic scoring, optimized for generating NYT-style crossword puzzles.

## Algorithm Overview

This is **not** a true backtracking algorithm (despite the name). It's a **greedy forward-only placement algorithm** with:
- Heuristic-based position scoring
- Multi-phase placement strategy
- Cross-word validation
- Randomized variation generation

### Search Strategy

The algorithm uses multiple strategies to find good solutions:

1. **Retry Loop:** If a generation attempt doesn't meet goals, retry up to 5 times
2. **Beam Search:** Maintains 3 parallel grids (beams), exploring different placement paths simultaneously. After each word, keeps the top 3 best states.
3. **Score-based selection:** Beams are scored by `(num_words × 10) - black_percent`, favoring more words and fewer black squares.

This allows the algorithm to explore multiple paths and avoid getting stuck in local optima.

---

## Algorithm Phases

The algorithm operates in 4 sequential phases (5 with `--short-fill`):

### Phase 1: Seed Top Row

**Goal:** Place a long word horizontally at position (0, 0).

**Process:**
1. Filter words to those with length 8–15 characters
2. Among those, select words in the top 30% by length
3. **Randomly** pick one from this candidate pool
4. Place it at row 0, column 0, going across

**Why randomize?** This creates variation between runs. Without randomization, the longest word always wins, producing identical puzzles.

```
Before:  # # # # # # # # # # # # # # #
After:   F A M I L Y G A M E N I G H T  (example)
```

### Phase 2: Seed Left Column

**Goal:** Place a long word vertically at position (0, 0), intersecting with the first word.

**Process:**
1. Find all words that:
   - Start with the same letter as the first word
   - Haven't been placed yet
   - Can legally be placed at (0, 0) going down
2. Sort by length (longest first)
3. **Randomly** pick from the top 50% of candidates
4. Place it

**Constraint:** The word must share its first letter with the horizontal word already placed.

```
F A M I L Y G A M E N I G H T
O
C
U
S
E
D
```

### Phase 3: Fill Top Row and Left Column

**Goal:** Maximize coverage of the top row and left column (NYT style).

**Process:**
For each remaining word (longest first):
1. Check if it can be placed **vertically starting from the top row** (i.e., intersecting with a letter in row 0)
2. Check if it can be placed **horizontally starting from the left column** (i.e., intersecting with a letter in column 0)
3. Score each valid position
4. Place at the highest-scoring position (if any valid position exists)

**Scoring bonuses in this phase:**
- +1000 for horizontal placement in row 0
- +1000 for vertical placement in column 0
- +800 for placements that touch row 0 or column 0

### Phase 4: Fill Remaining Grid

**Goal:** Place remaining words anywhere they fit, until target density is reached.

**Process:**
For each remaining word:
1. If black square percentage ≤ 18% AND edge requirements are met → **STOP**
2. Search all positions on the grid
3. Only consider positions where the word **intersects** with existing letters (inters > 0)
4. Score each position
5. Place at the highest-scoring position

**Termination conditions:**
- Black square percentage drops to target (18%)
- Top row is ≥80% white
- Left column is ≥80% white

**Beam Search in Phase 4:**
Instead of maintaining one grid, Phase 4 uses beam search:
1. Start with K=3 parallel grids (beams)
2. For each word, try all valid positions on all beams
3. Score each resulting state: `(num_words × 10) - black_percent`
4. Keep only the top K states
5. At the end, return the best beam

This explores multiple placement paths simultaneously, avoiding local dead-ends.

### Phase 5: Short-Word Gap Filling (Optional)

**Enabled with:** `--short-fill` flag

**Goal:** Fill remaining gaps with short, flexible words when Phase 4 leaves too many black squares.

**Process:**
1. Only runs if black% > 30% after Phase 4
2. Get remaining unused words, sorted shortest-first (3-6 chars only)
3. For each short word, find positions with at least 1 intersection
4. Place with +100 bonus score for gap-filling
5. Stop when black% ≤ 18%

---

## Position Scoring Function

Each candidate position receives a score based on:

| Factor | Points | Rationale |
|--------|--------|-----------|
| Intersections | +200 per letter | More intersections = tighter grid |
| Horizontal in row 0 | +1000 + (length × 100) | NYT style: dense top row |
| Vertical starting row 0 | +800 | Contributes to top row coverage |
| Vertical in column 0 | +1000 + (length × 100) | NYT style: dense left column |
| Horizontal starting col 0 | +800 | Contributes to left column coverage |
| Row 1 or Column 1 | +200 | Keep top-left quadrant dense |
| Distance from center | +2 per unit | Spread words outward |
| Adjacent filled cells | +10 per neighbor | Cluster words together |
| **Random noise** | +0 to RANDOMNESS_FACTOR | Create variation |

---

## Cross-Word Validation

When placing a word, the algorithm validates that **all perpendicular words formed are legal**.

### Rules:
1. **No word extensions:** A word cannot be placed if it would extend an existing word. The cell immediately before and after the word (in its direction) must be `#` or grid edge.

2. **Valid cross-words:** When placing a horizontal word, each letter may create or extend a vertical word (and vice versa). If the resulting cross-word is:
   - **Complete** (bounded by `#` or edges on both ends), AND
   - **3+ characters long**, AND
   - **Not in the input word list**
   
   → The placement is **rejected**.

### Example:
```
Existing grid:        Trying to place "CAT" at row 2:
  C A R                 C A R
  # # #                 C A T  ← Would create "AC" vertically
  # # #                         But "AC" is only 2 chars, so OK

  C A R                 C A R
  A # #                 A # #
  # # #         →       C A T  ← Would create "CAC" vertically
                                "CAC" is 3+ chars and not in word list
                                → REJECTED
```

---

## Randomness Sources

The algorithm produces different outputs on each run due to:

1. **Word order shuffling:** Words of the same length are shuffled randomly before processing.

2. **Phase 1 random selection:** The first word is randomly chosen from top candidates, not deterministically the longest.

3. **Phase 2 random selection:** The second word is randomly chosen from valid candidates.

4. **Score noise:** Every position score gets `random.randint(0, RANDOMNESS_FACTOR)` added, causing different tiebreakers.

### Reproducibility

Pass a `seed` parameter to `generate()` for reproducible results:

```python
generate(words, grid_size=15, seed=42)  # Same output every time
generate(words, grid_size=15, seed=None)  # Random each time
```

---

## Configuration Constants

```python
GRID_SIZE = 15                    # 15x15 grid (NYT standard)
TARGET_BLACK_PERCENT = 18         # Stop when ≤18% black squares
MIN_TOP_ROW_WHITE_PERCENT = 80    # Top row must be ≥80% white
MIN_LEFT_COL_WHITE_PERCENT = 80   # Left column must be ≥80% white
RANDOMNESS_FACTOR = 50            # Max random noise added to scores
SHORT_FILL_THRESHOLD = 30         # Switch to short words when black% > this
COMMON_LETTERS = "ETAOINSHRDLCUMWFGYPBVKJXQZ"  # For connectivity scoring
MAX_RETRIES = 5                   # Number of full generation attempts
BACKTRACK_DEPTH = 5               # How many placements can be undone when stuck
BEAM_WIDTH = 3                    # Number of parallel grids in beam search
```

---

## Experimental Flags (A/B Testing)

### `--short-fill`

**Problem:** The algorithm often gets stuck at ~35-40% black squares because remaining words can't find intersection points.

**Solution:** After Phase 4, if black% > 30%, run an additional pass prioritizing short words (3-6 characters) which are more flexible and can fill gaps.

```bash
python algos/custom_backtracking.py words.txt --short-fill
```

**How it works:**
1. After Phase 4 completes, check if black% > 30%
2. If yes, create a list of unused short words (3-6 chars), shortest first
3. Try to place each one, requiring at least 1 intersection
4. Stop when black% ≤ 18% or no more placements possible

**Trade-offs:**
- ✅ Lower black square percentage
- ❌ May result in many small words

---

### `--connectivity`

**Problem:** Words with rare letters (Q, X, Z, J) are hard to intersect with other words, leading to fragmented grids.

**Solution:** Prioritize words with common letters (E, T, A, O, I, N, S) that are more likely to create intersection opportunities.

```bash
python algos/custom_backtracking.py words.txt --connectivity
```

**How it works:**

Each word gets a "connectivity score" based on letter frequency:
```
E = 26 points (most common)
T = 25 points
A = 24 points
...
Z = 1 point (least common)
```

**Example scores:**
```
"ERATE"  → E(26) + R(16) + A(24) + T(25) + E(26) = 117 points
"JAZZY"  → J(4)  + A(24) + Z(1)  + Z(1)  + Y(7)  = 37 points
```

Words are sorted by connectivity within each length group, with the top 1/3 shuffled for variation.

**Trade-offs:**
- ✅ Better word interconnection
- ❌ May underutilize interesting words with rare letters

---

### `--full-shuffle`

**Problem:** The default longest-first ordering means similar grids every time.

**Solution:** Randomize word order for Phases 3-4, while still seeding the top row and left column with long words (Phase 1-2).

```bash
python algos/custom_backtracking.py words.txt --full-shuffle
```

**How it works:**
1. **Phase 1-2:** Still pick long words (8+ chars) for top row and left column (NYT structure preserved)
2. **Phase 3-4:** Process remaining words in fully random order

**Trade-offs:**
- ✅ Much more variation in grid fill patterns
- ✅ Top row/left column still have long words (NYT style)
- ❌ May produce different density patterns

---

### Combining Flags

```bash
python algos/custom_backtracking.py words.txt --full-shuffle --short-fill
python algos/custom_backtracking.py words.txt --connectivity --short-fill
```

| Combination | Effect |
|-------------|--------|
| `--full-shuffle` | Random order, ignore length |
| `--full-shuffle --short-fill` | Random order + gap-fill pass |
| `--connectivity` | Longest first, sorted by common letters within length |
| `--connectivity --short-fill` | Connectivity + gap-fill pass |

Note: `--full-shuffle` and `--connectivity` are mutually exclusive in effect (full-shuffle ignores all sorting)

---

## Complexity Analysis

### Time Complexity

- **Phase 1:** O(W) where W = number of words
- **Phase 2:** O(W × G) where G = grid_size (validation cost)
- **Phase 3:** O(W × G) for edge placements
- **Phase 4:** O(W × G² × L) where L = average word length

**Worst case:** O(W × G² × L) ≈ O(W × 225 × L) for a 15x15 grid

### Space Complexity

- Grid: O(G²) = O(225) for 15x15
- Word sets: O(W × L) for storing words
- Placed words list: O(P × L) where P = placed words

**Total:** O(G² + W × L)

---

## Limitations

1. **No true backtracking:** If the algorithm gets stuck (no valid placements), it doesn't undo previous choices. It simply stops.

2. **Greedy, not optimal:** The algorithm doesn't guarantee the maximum number of words or minimum black squares.

3. **Word list dependent:** Results heavily depend on the input word list. Lists with diverse starting letters and lengths produce better grids.

4. **No symmetry:** Unlike NYT puzzles, this doesn't enforce 180° rotational symmetry.

---

## Usage

### As a module:
```python
from algos.custom_backtracking import generate

words = ["HELLO", "WORLD", "PYTHON", ...]
result = generate(
    words,
    grid_size=15,
    seed=None,
    use_short_fill=False,
    use_connectivity=False,
)

if result:
    grid, word_positions, algo_name = result
```

### Standalone CLI:
```bash
# Basic usage (10 variations)
python algos/custom_backtracking.py words.txt

# Specify number of variations
python algos/custom_backtracking.py words.txt -n 20

# Enable experimental flags
python algos/custom_backtracking.py words.txt --short-fill
python algos/custom_backtracking.py words.txt --connectivity
python algos/custom_backtracking.py words.txt --short-fill --connectivity

# Custom output file
python algos/custom_backtracking.py words.txt -o my_crosswords.html

# Full example
python algos/custom_backtracking.py words.txt -n 15 --short-fill --connectivity -o output.html
```

### CLI Options:
| Option | Description |
|--------|-------------|
| `words_file` | Path to newline-separated words file (required) |
| `-n, --num-variations` | Number of variations to generate (default: 10) |
| `--full-shuffle` | Fully randomize word order (ignore length) |
| `--short-fill` | Prioritize short words when black% > 30% |
| `--connectivity` | Sort words by common letter frequency |
| `-o, --output` | Output HTML file (default: crossword_variations.html) |

---

## Output Format

```python
(
    grid,           # List[List[str]] - 2D grid, "#" for black squares
    word_positions, # List[dict] - {"word": str, "direction": str, "row": int, "col": int}
    algo_name       # str - "Custom Backtracking (Optimized)"
)
```

