"""
Custom Backtracking Crossword Generator

╔═══════════════════════════════════════════════════════════════════════════════╗
║  IMPORTANT: If you modify this file, UPDATE the README.md in this directory! ║
║  The README documents the algorithm in detail for future reference.           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Optimizes for NYT-style crosswords:
- Top row and left column mostly white
- Target ~18% black squares
- Even distribution of black squares (not clustered)
- All cross-words must be valid words from input list

Run standalone:
    python algos/custom_backtracking.py <words_file.txt> [num_variations] [flags]

Flags:
    --short-fill       Prioritize short words (3-5 chars) when black% > 30%
    --connectivity     Sort words by connectivity potential (common letters)
"""

import argparse
import random
import sys
from typing import List, Tuple, Optional, Set, Dict

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Grid dimensions
GRID_SIZE = 15

# Target percentage of black squares (NYT typically 15-20%)
TARGET_BLACK_PERCENT = 18

# Minimum percentage of white squares in top row (0-100)
MIN_TOP_ROW_WHITE_PERCENT = 80

# Minimum percentage of white squares in left column (0-100)
MIN_LEFT_COL_WHITE_PERCENT = 80

# Randomness factor (0 = deterministic, higher = more random)
# This adds random noise to position scores to create variation
RANDOMNESS_FACTOR = 50

# Short-fill threshold: switch to short words when black% exceeds this
SHORT_FILL_THRESHOLD = 30

# Common letters for connectivity scoring (ordered by frequency in English)
COMMON_LETTERS = "ETAOINSHRDLCUMWFGYPBVKJXQZ"

# Retry and backtracking settings
MAX_RETRIES = 5  # Number of full generation attempts
BACKTRACK_DEPTH = 5  # How many placements to potentially undo when stuck

# Beam search settings
BEAM_WIDTH = 7  # Number of parallel grids to maintain


def calculate_connectivity_score(word: str) -> int:
    """
    Calculate how likely a word is to intersect with other words.
    Words with common letters (E, T, A, O, I, N, S) score higher.
    """
    score = 0
    for char in word.upper():
        if char in COMMON_LETTERS:
            # Higher score for more common letters (E=26, T=25, ..., Z=1)
            score += 26 - COMMON_LETTERS.index(char)
    return score


def _evaluate_result(grid, placed_words, grid_size: int) -> float:
    """Score a result for comparison. Higher is better."""
    if grid is None:
        return -float("inf")

    total_cells = grid_size * grid_size
    black_cells = sum(
        1 for i in range(grid_size) for j in range(grid_size) if grid[i][j] == "#"
    )
    black_percent = (black_cells / total_cells) * 100
    white_cells = total_cells - black_cells

    # Penalize high black percentage (want ~18%)
    black_penalty = max(0, black_percent - TARGET_BLACK_PERCENT) * 10

    # Reward more words and white cells
    return white_cells + len(placed_words) * 5 - black_penalty


def _meets_goals(grid, grid_size: int) -> bool:
    """Check if result meets optimization goals."""
    if grid is None:
        return False

    total_cells = grid_size * grid_size
    black_cells = sum(
        1 for i in range(grid_size) for j in range(grid_size) if grid[i][j] == "#"
    )
    black_percent = (black_cells / total_cells) * 100

    top_white = sum(1 for j in range(grid_size) if grid[0][j] != "#")
    left_white = sum(1 for i in range(grid_size) if grid[i][0] != "#")
    top_percent = (top_white / grid_size) * 100
    left_percent = (left_white / grid_size) * 100

    # Allow 5% tolerance on black squares
    return (
        black_percent <= TARGET_BLACK_PERCENT + 5
        and top_percent >= MIN_TOP_ROW_WHITE_PERCENT - 10
        and left_percent >= MIN_LEFT_COL_WHITE_PERCENT - 10
    )


def _copy_grid(grid):
    """Deep copy a grid."""
    return [row[:] for row in grid]


def generate(
    words: List[str],
    grid_size: int = GRID_SIZE,
    seed: Optional[int] = None,
    use_short_fill: bool = False,
    use_connectivity: bool = False,
    use_full_shuffle: bool = False,
) -> Optional[Tuple]:
    """
    Generate crossword using custom backtracking algorithm with retry loop.

    Args:
        words: List of words to place in the crossword
        grid_size: Size of the grid (default 15x15)
        seed: Random seed for reproducibility (None = random each time)
        use_short_fill: If True, prioritize short words when black% > 30%
        use_connectivity: If True, sort words by connectivity potential
        use_full_shuffle: If True, fully randomize word order (ignore length)

    Returns:
        Tuple of (grid, word_positions, algorithm_name) or None if failed
    """
    best_result = None
    best_score = -float("inf")

    for attempt in range(MAX_RETRIES):
        result = _generate_attempt(
            words, grid_size, seed, use_short_fill, use_connectivity, use_full_shuffle
        )
        if result:
            grid, placed_words, algo_name = result
            score = _evaluate_result(grid, placed_words, grid_size)

            if _meets_goals(grid, grid_size):
                # Found a good result!
                return result

            if score > best_score:
                best_score = score
                best_result = result

    # Return best attempt even if it didn't meet goals
    return best_result


def _generate_attempt(
    words: List[str],
    grid_size: int,
    seed: Optional[int],
    use_short_fill: bool,
    use_connectivity: bool,
    use_full_shuffle: bool,
) -> Optional[Tuple]:
    """Single generation attempt with backtracking."""
    try:
        # Set random seed for reproducibility (None = random each time)
        if seed is not None:
            random.seed(seed)

        if use_full_shuffle:
            # FULL SHUFFLE: completely random order, ignore length
            sorted_words = list(words)
            random.shuffle(sorted_words)
            short_words_first = sorted(words, key=len)  # for short-fill phase
        else:
            # DEFAULT: Sort by length, then shuffle within same length for variation
            words_by_length: Dict[int, List[str]] = {}
            for w in words:
                length = len(w)
                if length not in words_by_length:
                    words_by_length[length] = []
                words_by_length[length].append(w)

            # Shuffle words within each length group
            for length in words_by_length:
                if use_connectivity:
                    # Sort by connectivity score (higher = more common letters)
                    words_by_length[length].sort(
                        key=lambda w: calculate_connectivity_score(w), reverse=True
                    )
                    # Add some randomness to top candidates
                    top_n = max(1, len(words_by_length[length]) // 3)
                    top_words = words_by_length[length][:top_n]
                    rest = words_by_length[length][top_n:]
                    random.shuffle(top_words)
                    words_by_length[length] = top_words + rest
                else:
                    random.shuffle(words_by_length[length])

            # Rebuild sorted list (longest first, shuffled within length)
            sorted_words = []
            for length in sorted(words_by_length.keys(), reverse=True):
                sorted_words.extend(words_by_length[length])

            # Also create a short-words-first list for gap filling
            short_words_first = []
            for length in sorted(words_by_length.keys()):  # shortest first
                short_words_first.extend(words_by_length[length])

        word_set = set(w.upper() for w in words)  # For fast lookup
        grid = [["#" for _ in range(grid_size)] for _ in range(grid_size)]
        placed_words = []

        def get_horizontal_word_at(row, col):
            """Get the full horizontal word that includes position (row, col).
            Returns (word, is_complete) where is_complete means bounded by #/edges."""
            if grid[row][col] == "#":
                return "", True
            start = col
            while start > 0 and grid[row][start - 1] != "#":
                start -= 1
            end = col
            while end < grid_size - 1 and grid[row][end + 1] != "#":
                end += 1
            word = "".join(grid[row][start : end + 1])
            is_complete = "#" not in word
            return word, is_complete

        def get_vertical_word_at(row, col):
            """Get the full vertical word that includes position (row, col).
            Returns (word, is_complete) where is_complete means bounded by #/edges."""
            if grid[row][col] == "#":
                return "", True
            start = row
            while start > 0 and grid[start - 1][col] != "#":
                start -= 1
            end = row
            while end < grid_size - 1 and grid[end + 1][col] != "#":
                end += 1
            word = "".join(grid[r][col] for r in range(start, end + 1))
            is_complete = "#" not in word
            return word, is_complete

        def can_place_word(word, row, col, direction):
            """Check if a word can be placed at the given position."""
            if direction == "across":
                if col + len(word) > grid_size:
                    return False

                # Must not extend existing word (cell before must be # or edge)
                if col > 0 and grid[row][col - 1] != "#":
                    return False
                # Must not be extended (cell after must be # or edge)
                if col + len(word) < grid_size and grid[row][col + len(word)] != "#":
                    return False

                for i, char in enumerate(word):
                    cell = grid[row][col + i]
                    if cell != "#" and cell != char:
                        return False

                    # Check if placing this letter creates invalid vertical word
                    if cell == "#":
                        grid[row][col + i] = char
                        vert_word, is_complete = get_vertical_word_at(row, col + i)
                        grid[row][col + i] = "#"

                        # Only validate complete words of 3+ chars
                        if is_complete and len(vert_word) >= 2:
                            if len(vert_word) == 2:
                                # 2-letter sequences are NEVER allowed
                                return False
                            elif vert_word not in word_set:
                                # 3+ letter words must be in word list
                                return False
            else:  # down
                if row + len(word) > grid_size:
                    return False

                # Must not extend existing word (cell above must be # or edge)
                if row > 0 and grid[row - 1][col] != "#":
                    return False
                # Must not be extended (cell below must be # or edge)
                if row + len(word) < grid_size and grid[row + len(word)][col] != "#":
                    return False

                for i, char in enumerate(word):
                    cell = grid[row + i][col]
                    if cell != "#" and cell != char:
                        return False

                    # Check if placing this letter creates invalid horizontal word
                    if cell == "#":
                        grid[row + i][col] = char
                        horiz_word, is_complete = get_horizontal_word_at(row + i, col)
                        grid[row + i][col] = "#"

                        # Only validate complete words of 3+ chars
                        if is_complete and len(horiz_word) >= 2:
                            if len(horiz_word) == 2:
                                # 2-letter sequences are NEVER allowed
                                return False
                            elif horiz_word not in word_set:
                                # 3+ letter words must be in word list
                                return False
            return True

        def place_word(word, row, col, direction):
            """Place a word on the grid."""
            if direction == "across":
                for i, char in enumerate(word):
                    grid[row][col + i] = char
            else:
                for i, char in enumerate(word):
                    grid[row + i][col] = char

        def get_intersections(word, row, col, direction):
            """Count how many letters intersect with existing words."""
            count = 0
            if direction == "across":
                for i, char in enumerate(word):
                    if grid[row][col + i] == char:
                        count += 1
            else:
                for i, char in enumerate(word):
                    if grid[row + i][col] == char:
                        count += 1
            return count

        def get_black_percent():
            """Calculate the percentage of black squares in the grid."""
            total = grid_size * grid_size
            black = sum(
                1
                for i in range(grid_size)
                for j in range(grid_size)
                if grid[i][j] == "#"
            )
            return (black / total) * 100

        def get_top_row_white_percent():
            """Calculate percentage of white squares in the top row."""
            white = sum(1 for j in range(grid_size) if grid[0][j] != "#")
            return (white / grid_size) * 100

        def get_left_col_white_percent():
            """Calculate percentage of white squares in the left column."""
            white = sum(1 for i in range(grid_size) if grid[i][0] != "#")
            return (white / grid_size) * 100

        def meets_edge_requirements():
            """Check if top row and left column meet minimum white requirements."""
            return (
                get_top_row_white_percent() >= MIN_TOP_ROW_WHITE_PERCENT
                and get_left_col_white_percent() >= MIN_LEFT_COL_WHITE_PERCENT
            )

        def position_score(word, row, col, direction, intersections):
            """Score a position - optimizes for top row, left col, good distribution."""
            score = intersections * 200

            # STRONG bonus for top row coverage
            if direction == "across" and row == 0:
                score += 1000 + len(word) * 100
            if direction == "down" and row == 0:
                score += 800

            # STRONG bonus for left column coverage
            if direction == "down" and col == 0:
                score += 1000 + len(word) * 100
            if direction == "across" and col == 0:
                score += 800

            # Bonus for filling second row/column (keeps top-left dense)
            if row == 1 or col == 1:
                score += 200

            # Prefer spreading out
            dist_from_center = abs(row - grid_size // 2) + abs(col - grid_size // 2)
            score += dist_from_center * 2

            # Bonus for filling cells that have letters on multiple sides
            cells_filled = []
            if direction == "across":
                cells_filled = [(row, col + i) for i in range(len(word))]
            else:
                cells_filled = [(row + i, col) for i in range(len(word))]

            for r, c in cells_filled:
                neighbors_with_letters = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        if grid[nr][nc] != "#":
                            neighbors_with_letters += 1
                score += neighbors_with_letters * 10

            # Add randomness for variation between runs
            score += random.randint(0, RANDOMNESS_FACTOR)

            return score

        # PHASE 1: Fill top row with a long word (randomly selected from top candidates)
        # Always use length-sorted candidates for Phase 1, even with --full-shuffle
        # This ensures the top row gets a long word for NYT-style structure
        top_row_candidates = sorted(
            [w for w in words if 8 <= len(w) <= grid_size], key=len, reverse=True
        )
        first_word = None
        if top_row_candidates:
            # Randomly pick from the longer words (top 30% by length)
            max_len = len(top_row_candidates[0])
            min_candidate_len = max(8, int(max_len * 0.7))
            best_candidates = [
                w for w in top_row_candidates if len(w) >= min_candidate_len
            ]
            first_word = (
                random.choice(best_candidates)
                if best_candidates
                else random.choice(top_row_candidates)
            )

        if first_word:
            place_word(first_word, 0, 0, "across")
            placed_words.append(
                {"word": first_word, "direction": "across", "row": 0, "col": 0}
            )

        # PHASE 2: Fill left column (randomly from matching candidates)
        # Always use original word list for Phase 2 to ensure long word selection
        placed_word_set = {pw["word"] for pw in placed_words}
        if first_word:
            # Find all words that could start the left column (same first letter)
            left_col_candidates = [
                w
                for w in words  # Use original words, not sorted_words
                if w not in placed_word_set
                and w[0] == first_word[0]
                and len(w) <= grid_size
                and can_place_word(w, 0, 0, "down")
            ]
            if left_col_candidates:
                # Prefer longer words but add randomness
                left_col_candidates.sort(key=len, reverse=True)
                # Pick from top 50% by length
                cutoff = max(1, len(left_col_candidates) // 2)
                second_word = random.choice(left_col_candidates[:cutoff])
                place_word(second_word, 0, 0, "down")
                placed_words.append(
                    {"word": second_word, "direction": "down", "row": 0, "col": 0}
                )
                placed_word_set.add(second_word)

        # PHASE 3: Aggressively fill top row and left column
        for word in sorted_words:
            if word in placed_word_set:
                continue

            best = None
            best_score = -1

            # Try placing DOWN from top row (fills top row)
            for col in range(grid_size):
                if grid[0][col] != "#" and word[0] == grid[0][col]:
                    if can_place_word(word, 0, col, "down"):
                        s = position_score(word, 0, col, "down", 1)
                        if s > best_score:
                            best_score = s
                            best = (0, col, "down")

            # Try placing ACROSS from left column (fills left col)
            for row in range(grid_size):
                if grid[row][0] != "#" and word[0] == grid[row][0]:
                    if can_place_word(word, row, 0, "across"):
                        s = position_score(word, row, 0, "across", 1)
                        if s > best_score:
                            best_score = s
                            best = (row, 0, "across")

            if best:
                r, c, d = best
                place_word(word, r, c, d)
                placed_words.append({"word": word, "direction": d, "row": r, "col": c})
                placed_word_set.add(word)

        # PHASE 4: Fill rest of grid using BEAM SEARCH
        # Maintain K parallel grids and keep best K after each word placement
        remaining = [w for w in sorted_words if w not in placed_word_set]

        # Beam state: list of (score, grid, placed_words, placed_word_set)
        def beam_score(b_grid, b_placed):
            """Score a beam state."""
            total = grid_size * grid_size
            black = sum(
                1
                for i in range(grid_size)
                for j in range(grid_size)
                if b_grid[i][j] == "#"
            )
            black_pct = (black / total) * 100
            # More words and fewer black squares = better
            return len(b_placed) * 10 - black_pct

        def get_horizontal_word_on_grid(b_grid, row, col):
            """Get full horizontal word at position on a specific grid."""
            if b_grid[row][col] == "#":
                return "", True
            start = col
            while start > 0 and b_grid[row][start - 1] != "#":
                start -= 1
            end = col
            while end < grid_size - 1 and b_grid[row][end + 1] != "#":
                end += 1
            word = "".join(b_grid[row][start : end + 1])
            is_complete = "#" not in word
            return word, is_complete

        def get_vertical_word_on_grid(b_grid, row, col):
            """Get full vertical word at position on a specific grid."""
            if b_grid[row][col] == "#":
                return "", True
            start = row
            while start > 0 and b_grid[start - 1][col] != "#":
                start -= 1
            end = row
            while end < grid_size - 1 and b_grid[end + 1][col] != "#":
                end += 1
            word = "".join(b_grid[r][col] for r in range(start, end + 1))
            is_complete = "#" not in word
            return word, is_complete

        def can_place_on_grid(b_grid, word, row, col, direction):
            """Check if word can be placed on a specific grid WITH cross-word validation."""
            if direction == "across":
                if col + len(word) > grid_size:
                    return False
                if col > 0 and b_grid[row][col - 1] != "#":
                    return False
                if col + len(word) < grid_size and b_grid[row][col + len(word)] != "#":
                    return False
                for i, char in enumerate(word):
                    cell = b_grid[row][col + i]
                    if cell != "#" and cell != char:
                        return False
                    # CROSS-WORD VALIDATION: Check vertical word formed
                    if cell == "#":
                        # Temporarily place letter
                        test_grid = _copy_grid(b_grid)
                        test_grid[row][col + i] = char
                        vert_word, is_complete = get_vertical_word_on_grid(
                            test_grid, row, col + i
                        )
                        if is_complete and len(vert_word) >= 2:
                            if len(vert_word) == 2:
                                # 2-letter sequences are NEVER allowed
                                return False
                            elif vert_word not in word_set:
                                # 3+ letter words must be in word list
                                return False
            else:
                if row + len(word) > grid_size:
                    return False
                if row > 0 and b_grid[row - 1][col] != "#":
                    return False
                if row + len(word) < grid_size and b_grid[row + len(word)][col] != "#":
                    return False
                for i, char in enumerate(word):
                    cell = b_grid[row + i][col]
                    if cell != "#" and cell != char:
                        return False
                    # CROSS-WORD VALIDATION: Check horizontal word formed
                    if cell == "#":
                        # Temporarily place letter
                        test_grid = _copy_grid(b_grid)
                        test_grid[row + i][col] = char
                        horiz_word, is_complete = get_horizontal_word_on_grid(
                            test_grid, row + i, col
                        )
                        if is_complete and len(horiz_word) >= 2:
                            if len(horiz_word) == 2:
                                # 2-letter sequences are NEVER allowed
                                return False
                            elif horiz_word not in word_set:
                                # 3+ letter words must be in word list
                                return False
            return True

        def get_intersections_on_grid(b_grid, word, row, col, direction):
            """Count intersections on a specific grid."""
            count = 0
            if direction == "across":
                for i, char in enumerate(word):
                    if b_grid[row][col + i] == char:
                        count += 1
            else:
                for i, char in enumerate(word):
                    if b_grid[row + i][col] == char:
                        count += 1
            return count

        def place_on_grid(b_grid, word, row, col, direction):
            """Place word on a grid copy and return new grid."""
            new_grid = _copy_grid(b_grid)
            if direction == "across":
                for i, char in enumerate(word):
                    new_grid[row][col + i] = char
            else:
                for i, char in enumerate(word):
                    new_grid[row + i][col] = char
            return new_grid

        # Initialize beams with current state
        beams = [
            (
                beam_score(grid, placed_words),
                _copy_grid(grid),
                list(placed_words),
                set(placed_word_set),
            )
        ]

        for word in remaining:
            # Check if best beam meets targets
            best_beam = max(beams, key=lambda b: b[0])
            b_grid = best_beam[1]
            total = grid_size * grid_size
            black = sum(
                1
                for i in range(grid_size)
                for j in range(grid_size)
                if b_grid[i][j] == "#"
            )
            if (black / total) * 100 <= TARGET_BLACK_PERCENT:
                break

            # Generate all possible next states across all beams
            candidates = []

            for b_score, b_grid, b_placed, b_placed_set in beams:
                if word in b_placed_set:
                    # Word already placed in this beam, keep beam as-is
                    candidates.append((b_score, b_grid, b_placed, b_placed_set))
                    continue

                found_placement = False
                for direction in ["across", "down"]:
                    for row in range(grid_size):
                        for col in range(grid_size):
                            if can_place_on_grid(b_grid, word, row, col, direction):
                                inters = get_intersections_on_grid(
                                    b_grid, word, row, col, direction
                                )
                                if inters > 0:
                                    # Create new beam state
                                    new_grid = place_on_grid(
                                        b_grid, word, row, col, direction
                                    )
                                    new_placed = b_placed + [
                                        {
                                            "word": word,
                                            "direction": direction,
                                            "row": row,
                                            "col": col,
                                        }
                                    ]
                                    new_placed_set = b_placed_set | {word}
                                    new_score = beam_score(new_grid, new_placed)
                                    # Add position score bonus
                                    new_score += (
                                        position_score(
                                            word, row, col, direction, inters
                                        )
                                        / 100
                                    )
                                    candidates.append(
                                        (
                                            new_score,
                                            new_grid,
                                            new_placed,
                                            new_placed_set,
                                        )
                                    )
                                    found_placement = True

                if not found_placement:
                    # Keep beam as-is if word can't be placed
                    candidates.append((b_score, b_grid, b_placed, b_placed_set))

            # Keep top BEAM_WIDTH beams
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:BEAM_WIDTH]

            # Also keep some diversity - if beams are too similar, add variation
            if len(beams) < BEAM_WIDTH and len(candidates) > BEAM_WIDTH:
                # Add some random candidates for diversity
                remaining_candidates = candidates[BEAM_WIDTH:]
                random.shuffle(remaining_candidates)
                beams.extend(remaining_candidates[: BEAM_WIDTH - len(beams)])

        # Select best beam
        best_beam = max(beams, key=lambda b: b[0])
        grid = best_beam[1]
        placed_words = best_beam[2]
        placed_word_set = best_beam[3]

        # Copy grid back to the main grid variable (for Phase 5)
        for i in range(grid_size):
            for j in range(grid_size):
                pass  # grid is already updated

        # PHASE 5: Short-word gap filling (if enabled and still too many black squares)
        if use_short_fill:
            current_black = get_black_percent()
            if current_black > SHORT_FILL_THRESHOLD:
                # Switch to short words (3-6 chars) to fill gaps
                short_remaining = [
                    w
                    for w in short_words_first
                    if w not in placed_word_set and 3 <= len(w) <= 6
                ]

                for word in short_remaining:
                    current_black = get_black_percent()
                    if current_black <= TARGET_BLACK_PERCENT:
                        break

                    best = None
                    best_score = -1

                    for direction in ["across", "down"]:
                        for row in range(grid_size):
                            for col in range(grid_size):
                                if can_place_word(word, row, col, direction):
                                    inters = get_intersections(
                                        word, row, col, direction
                                    )
                                    if inters > 0:
                                        # Bonus for short words that fill gaps
                                        s = position_score(
                                            word, row, col, direction, inters
                                        )
                                        s += 100  # Bonus for gap-filling
                                        if s > best_score:
                                            best_score = s
                                            best = (row, col, direction)

                    if best:
                        r, c, d = best
                        place_word(word, r, c, d)
                        placed_words.append(
                            {"word": word, "direction": d, "row": r, "col": c}
                        )
                        placed_word_set.add(word)

        # Build algorithm name based on flags
        algo_name = "Custom Backtracking"
        flags = []
        if use_full_shuffle:
            flags.append("full-shuffle")
        if use_short_fill:
            flags.append("short-fill")
        if use_connectivity:
            flags.append("connectivity")
        if flags:
            algo_name += f" ({', '.join(flags)})"

        return grid, placed_words, algo_name
    except Exception as e:
        print(f"Backtracking error: {e}")
        return None


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


def read_words_from_file(filename: str) -> List[str]:
    """Read words from a newline-separated text file."""
    with open(filename, "r") as f:
        words = [line.strip().upper() for line in f if line.strip()]
    words = [w for w in words if len(w) <= GRID_SIZE]
    return words


def calculate_statistics(grid, word_positions, input_words: List[str]) -> dict:
    """Calculate statistics for a generated crossword."""
    stats = {
        "avg_word_length_ratio": 0,
        "black_square_percentage": 0,
        "vertical_horizontal_ratio": 0,
        "num_words": 0,
        "top_row_white_percent": 0,
        "left_col_white_percent": 0,
    }

    if grid is None or word_positions is None:
        return stats

    grid_size = len(grid)
    input_avg_length = (
        sum(len(w) for w in input_words) / len(input_words) if input_words else 0
    )

    vertical_count = 0
    horizontal_count = 0
    placed_words = []

    for item in word_positions:
        if isinstance(item, dict):
            word = item.get("word", "")
            direction = item.get("direction", "across")
            if word:
                placed_words.append(word)
                if direction in ["down", "vertical", "v"]:
                    vertical_count += 1
                else:
                    horizontal_count += 1

    stats["num_words"] = len(placed_words)

    if placed_words and input_avg_length > 0:
        placed_avg_length = sum(len(w) for w in placed_words) / len(placed_words)
        stats["avg_word_length_ratio"] = placed_avg_length / input_avg_length

    total_cells = grid_size * grid_size
    black_cells = sum(
        1
        for i in range(grid_size)
        for j in range(grid_size)
        if grid[i][j] in ["#", " ", ".", None, "", 0]
    )
    stats["black_square_percentage"] = (black_cells / total_cells) * 100

    if horizontal_count > 0:
        stats["vertical_horizontal_ratio"] = vertical_count / horizontal_count
    elif vertical_count > 0:
        stats["vertical_horizontal_ratio"] = float("inf")

    # Edge stats
    top_white = sum(1 for j in range(grid_size) if grid[0][j] != "#")
    left_white = sum(1 for i in range(grid_size) if grid[i][0] != "#")
    stats["top_row_white_percent"] = (top_white / grid_size) * 100
    stats["left_col_white_percent"] = (left_white / grid_size) * 100

    return stats


def generate_variations_html(
    results: list, input_stats: dict, grid_size: int, flags: List[str] = None
) -> str:
    """Generate HTML page showing multiple crossword variations."""
    flags = flags or []

    def generate_grid_html(grid, title: str, stats: dict, variation_num: int) -> str:
        if grid is None:
            return f"""
            <div class="crossword-container">
                <h2>Variation {variation_num}: {title}</h2>
                <div class="failed">Failed to generate</div>
            </div>
            """

        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        # Build grid HTML with word numbers
        word_numbers = {}
        current_number = 1
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == "#":
                    continue
                starts_horiz = (j == 0 or grid[i][j - 1] == "#") and (
                    j < cols - 1 and grid[i][j + 1] != "#"
                )
                starts_vert = (i == 0 or grid[i - 1][j] == "#") and (
                    i < rows - 1 and grid[i + 1][j] != "#"
                )
                if starts_horiz or starts_vert:
                    word_numbers[(i, j)] = current_number
                    current_number += 1

        grid_html = f'<div class="grid" style="grid-template-columns: repeat({grid_size}, 1fr);">'
        for i in range(rows):
            for j in range(cols):
                cell = grid[i][j]
                if cell and cell not in ["#", " ", ".", None, 0, ""]:
                    num = word_numbers.get((i, j), "")
                    num_html = f'<span class="word-number">{num}</span>' if num else ""
                    grid_html += f'<div class="cell white">{num_html}<span class="letter">{cell}</span></div>'
                else:
                    grid_html += '<div class="cell black"></div>'
        grid_html += "</div>"

        return f"""
        <div class="crossword-container">
            <h2>Variation {variation_num}</h2>
            {grid_html}
            <div class="stats">
                <div class="stat"><span class="label">Words:</span> <span class="value">{stats.get('num_words', 0)}</span></div>
                <div class="stat"><span class="label">Black %:</span> <span class="value">{stats.get('black_square_percentage', 0):.1f}%</span></div>
                <div class="stat"><span class="label">Top row white:</span> <span class="value">{stats.get('top_row_white_percent', 0):.0f}%</span></div>
                <div class="stat"><span class="label">Left col white:</span> <span class="value">{stats.get('left_col_white_percent', 0):.0f}%</span></div>
            </div>
        </div>
        """

    grids_html = ""
    for i, result in enumerate(results):
        if result:
            grid, word_positions, title = result
            stats = calculate_statistics(grid, word_positions, input_stats["words"])
            grids_html += generate_grid_html(grid, title, stats, i + 1)
        else:
            grids_html += generate_grid_html(None, "Failed", {}, i + 1)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crossword Variations</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            padding: 40px 20px;
            color: #e8e8e8;
        }}
        h1 {{
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 40px; }}
        .input-stats {{
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }}
        .input-stats span {{ margin: 0 20px; color: #aaa; }}
        .input-stats strong {{ color: #00d9ff; }}
        .grid-wrapper {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        .crossword-container {{
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .crossword-container h2 {{
            font-size: 1rem;
            margin-bottom: 15px;
            color: #fff;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .grid {{
            display: grid;
            gap: 1px;
            background: #333;
            padding: 1px;
            border-radius: 4px;
            margin-bottom: 15px;
            aspect-ratio: 1;
        }}
        .cell {{
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: clamp(6px, 1.2vw, 11px);
            text-transform: uppercase;
            position: relative;
        }}
        .cell.white {{ background: #fff; color: #1a1a2e; }}
        .cell.black {{ background: #1a1a2e; }}
        .word-number {{
            position: absolute;
            top: 1px;
            left: 2px;
            font-size: clamp(4px, 0.6vw, 7px);
            font-weight: 400;
            color: #555;
        }}
        .stats {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }}
        .stat {{
            background: rgba(255,255,255,0.05);
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
        }}
        .stat .label {{ color: #888; }}
        .stat .value {{ color: #00ff88; font-weight: 600; float: right; }}
        .failed {{ text-align: center; padding: 50px; color: #ff6b6b; }}
    </style>
</head>
<body>
    <h1>Crossword Variations</h1>
    <p class="subtitle">{len(results)} variations generated{' with ' + ', '.join(flags) if flags else ''}</p>
    <div class="input-stats">
        <span>Words: <strong>{input_stats['total_words']}</strong></span>
        <span>Avg length: <strong>{input_stats['avg_length']:.2f}</strong></span>
    </div>
    <div class="grid-wrapper">
        {grids_html}
    </div>
</body>
</html>
"""


def main():
    """Generate multiple crossword variations and output to HTML."""
    parser = argparse.ArgumentParser(
        description="Generate crossword variations with custom backtracking algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python algos/custom_backtracking.py words.txt
  python algos/custom_backtracking.py words.txt -n 20
  python algos/custom_backtracking.py words.txt --short-fill
  python algos/custom_backtracking.py words.txt --connectivity
  python algos/custom_backtracking.py words.txt --short-fill --connectivity
        """,
    )
    parser.add_argument("words_file", help="Path to newline-separated words file")
    parser.add_argument(
        "-n",
        "--num-variations",
        type=int,
        default=10,
        help="Number of variations to generate (default: 10)",
    )
    parser.add_argument(
        "--short-fill",
        action="store_true",
        help="Prioritize short words (3-6 chars) when black%% > 30%% to fill gaps",
    )
    parser.add_argument(
        "--connectivity",
        action="store_true",
        help="Sort words by connectivity potential (common letters E,T,A,O,I,N,S)",
    )
    parser.add_argument(
        "--full-shuffle",
        action="store_true",
        help="Fully randomize word order (ignore length sorting)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="crossword_variations.html",
        help="Output HTML file (default: crossword_variations.html)",
    )

    args = parser.parse_args()

    print(f"Reading words from {args.words_file}...")
    words = read_words_from_file(args.words_file)
    print(f"Loaded {len(words)} words")

    # Print active flags
    active_flags = []
    if args.full_shuffle:
        active_flags.append("full-shuffle")
    if args.short_fill:
        active_flags.append("short-fill")
    if args.connectivity:
        active_flags.append("connectivity")
    if active_flags:
        print(f"Active flags: {', '.join(active_flags)}")

    input_avg_length = sum(len(w) for w in words) / len(words) if words else 0
    input_stats = {
        "total_words": len(words),
        "avg_length": input_avg_length,
        "words": words,
    }

    print(f"\nGenerating {args.num_variations} variations...")
    results = []
    for i in range(args.num_variations):
        print(f"  Variation {i + 1}/{args.num_variations}...", end=" ")
        result = generate(
            words,
            GRID_SIZE,
            seed=None,
            use_short_fill=args.short_fill,
            use_connectivity=args.connectivity,
            use_full_shuffle=args.full_shuffle,
        )
        if result:
            grid, word_positions, title = result
            stats = calculate_statistics(grid, word_positions, words)
            print(
                f"Words: {stats['num_words']}, Black: {stats['black_square_percentage']:.1f}%"
            )
        else:
            print("FAILED")
        results.append(result)

    print("\nGenerating HTML...")
    html_content = generate_variations_html(
        results, input_stats, GRID_SIZE, active_flags
    )

    with open(args.output, "w") as f:
        f.write(html_content)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
