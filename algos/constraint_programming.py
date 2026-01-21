"""
Constraint Programming Crossword Generator using OR-Tools CP-SAT

Based on: https://pedtsr.ca/2023/generating-crossword-grids-using-constraint-programming.html

This approach models the crossword as a constraint satisfaction problem:
- Variables L[r][c] for letters (0=black, 1-26=A-Z)
- Boolean variables B[r][c] for tracking black squares
- Table constraints for valid words
- Constraints for black square placement

Run standalone:
    python algos/constraint_programming.py <words_file.txt> [options]
"""

import argparse
import sys
import time
from typing import List, Tuple, Optional, Dict

from ortools.sat.python import cp_model


# =============================================================================
# CONFIGURATION
# =============================================================================

GRID_SIZE = 15
MIN_WORD_LENGTH = 3
MAX_BLACK_RATIO = 0.25  # Max 25% black squares
TIME_LIMIT_SECONDS = 300  # 5 minute time limit per solve


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def word_to_numbers(word: str) -> List[int]:
    """Convert a word to list of numbers (A=1, B=2, ..., Z=26)."""
    return [ord(c.upper()) - ord("A") + 1 for c in word]


def numbers_to_word(numbers: List[int]) -> str:
    """Convert list of numbers back to a word."""
    return "".join(chr(n + ord("A") - 1) for n in numbers if n > 0)


def load_words(path: str) -> Dict[int, List[List[int]]]:
    """
    Load words from file and return a wordlist organized by length.

    Returns:
        {length: [[letter_nums], ...], ...}
        e.g., {3: [[1,20,9], ...], 5: [[12,1,7,5,18], ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip().upper() for line in f if line.strip()]

    # Filter valid words (only letters, within grid size)
    valid_words = []
    for word in words:
        if word.isalpha() and MIN_WORD_LENGTH <= len(word) <= GRID_SIZE:
            valid_words.append(word)

    wordlist = {}
    for word in valid_words:
        word_length = len(word)
        word_nums = word_to_numbers(word)
        if word_length in wordlist:
            wordlist[word_length].append(word_nums)
        else:
            wordlist[word_length] = [word_nums]

    return wordlist


def load_words_raw(path: str) -> List[str]:
    """Load words as strings for statistics."""
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip().upper() for line in f if line.strip()]
    return [w for w in words if w.isalpha() and MIN_WORD_LENGTH <= len(w) <= GRID_SIZE]


# =============================================================================
# CROSSWORD SOLVER CLASS
# =============================================================================


class CrosswordSolver:
    """Constraint programming crossword generator using CP-SAT."""

    def __init__(
        self,
        wordlist: Dict[int, List[List[int]]],
        rows: int = GRID_SIZE,
        cols: int = GRID_SIZE,
    ):
        self.wordlist = wordlist
        self.rows = rows
        self.cols = cols
        self.model = cp_model.CpModel()

        # Decision variables
        self.L = None  # Letter variables (0=black, 1-26=letters)
        self.B = None  # Boolean black square variables

        # Word slot tracking
        self.word_slots = []  # List of (start_row, start_col, length, direction)

    def build_model(self):
        """Build the complete CP-SAT model."""
        print("Building constraint programming model...")

        # Create letter variables: L[r][c] = 0 (black) or 1-26 (A-Z)
        self.L = [
            [self.model.NewIntVar(0, 26, f"L[{r}][{c}]") for c in range(self.cols)]
            for r in range(self.rows)
        ]

        # Create black square boolean variables
        self.B = [
            [self.model.NewBoolVar(f"B[{r}][{c}]") for c in range(self.cols)]
            for r in range(self.rows)
        ]

        # Link L and B: L[r][c] == 0 iff B[r][c] == 1
        print("  Adding L-B linking constraints...")
        for r in range(self.rows):
            for c in range(self.cols):
                self.model.Add(self.L[r][c] == 0).OnlyEnforceIf(self.B[r][c])
                self.model.Add(self.L[r][c] != 0).OnlyEnforceIf(self.B[r][c].Not())

        # Black square constraints: max 3 black squares in any 3x3 subgrid
        # This prevents isolated regions while allowing more flexibility
        print("  Adding black square distribution constraints...")
        if self.rows >= 3 and self.cols >= 3:
            for r in range(self.rows - 2):
                for c in range(self.cols - 2):
                    subgrid_blacks = [
                        self.B[r + i][c + j] for i in range(3) for j in range(3)
                    ]
                    self.model.Add(sum(subgrid_blacks) <= 3)

        # Limit total black squares
        print("  Adding black square ratio constraint...")
        max_blacks = int(self.rows * self.cols * MAX_BLACK_RATIO)
        all_blacks = [self.B[r][c] for r in range(self.rows) for c in range(self.cols)]
        self.model.Add(sum(all_blacks) <= max_blacks)

        # Build word slots and add word constraints
        self._add_word_constraints()

        print(f"  Model built with {len(self.word_slots)} potential word slots")

    def _add_word_constraints(self):
        """Add constraints for words across and down."""
        print("  Building word slot constraints...")

        # For each possible slot, we create an activation variable that is TRUE
        # if and only if that exact slot exists (bounded by blacks/edges, all white inside)

        # Helper to create reified "all true" constraint
        def create_slot_activation(conditions, slot_name):
            """Create slot_active variable that is true iff ALL conditions are true."""
            slot_active = self.model.NewBoolVar(slot_name)

            # Use proper reification: slot_active <=> (c1 AND c2 AND ... AND cn)
            # In CP-SAT, we can use AddBoolAnd with reification for the forward direction
            # and AddBoolOr for the reverse

            # Forward: slot_active => all conditions true
            # If slot_active is true, each condition must be true
            for cond in conditions:
                self.model.AddImplication(slot_active, cond)

            # Reverse: all conditions true => slot_active
            # Equivalent to: (NOT c1) OR (NOT c2) OR ... OR slot_active
            # For each condition, we need its negation
            negated_conds = []
            for cond in conditions:
                # cond might be B[r][c] (a BoolVar) or B[r][c].Not() (a negated literal)
                # In both cases, .Not() gives us the negation
                negated_conds.append(cond.Not())

            # At least one of the negated conditions OR slot_active must be true
            # This means: if ALL conditions are true, slot_active MUST be true
            self.model.AddBoolOr(negated_conds + [slot_active])

            return slot_active

        # ACROSS slots
        for r in range(self.rows):
            for start_c in range(self.cols):
                for length in range(MIN_WORD_LENGTH, self.cols - start_c + 1):
                    if length not in self.wordlist:
                        continue
                    end_c = start_c + length - 1

                    # Build conditions list (all must be true for slot to exist)
                    conditions = []

                    # Before start must be black or edge
                    if start_c > 0:
                        conditions.append(self.B[r][start_c - 1])

                    # After end must be black or edge
                    if end_c < self.cols - 1:
                        conditions.append(self.B[r][end_c + 1])

                    # All cells in slot must be white
                    for i in range(length):
                        conditions.append(self.B[r][start_c + i].Not())

                    # Create activation with bi-directional constraint
                    slot_active = create_slot_activation(
                        conditions, f"slot_across_{r}_{start_c}_{length}"
                    )

                    # If slot is active, letters must form valid word
                    if self.wordlist[length]:
                        slot_vars = [self.L[r][start_c + i] for i in range(length)]
                        self.model.AddAllowedAssignments(
                            slot_vars, self.wordlist[length]
                        ).OnlyEnforceIf(slot_active)

                    self.word_slots.append((r, start_c, length, "across", slot_active))

        # DOWN slots
        for c in range(self.cols):
            for start_r in range(self.rows):
                for length in range(MIN_WORD_LENGTH, self.rows - start_r + 1):
                    if length not in self.wordlist:
                        continue
                    end_r = start_r + length - 1

                    # Build conditions list
                    conditions = []

                    # Before start must be black or edge
                    if start_r > 0:
                        conditions.append(self.B[start_r - 1][c])

                    # After end must be black or edge
                    if end_r < self.rows - 1:
                        conditions.append(self.B[end_r + 1][c])

                    # All cells in slot must be white
                    for i in range(length):
                        conditions.append(self.B[start_r + i][c].Not())

                    # Create activation with bi-directional constraint
                    slot_active = create_slot_activation(
                        conditions, f"slot_down_{start_r}_{c}_{length}"
                    )

                    # If slot is active, letters must form valid word
                    if self.wordlist[length]:
                        slot_vars = [self.L[start_r + i][c] for i in range(length)]
                        self.model.AddAllowedAssignments(
                            slot_vars, self.wordlist[length]
                        ).OnlyEnforceIf(slot_active)

                    self.word_slots.append((start_r, c, length, "down", slot_active))

        # Constraint: single/double letter sequences must not exist
        # Every white cell must be part of at least one word of length >= MIN_WORD_LENGTH
        # Add constraints to forbid white sequences of unsupported lengths
        print("  Forbidding unsupported word lengths...")
        self._forbid_unsupported_lengths()

        print("  Adding minimum word length constraints...")
        self._add_min_word_length_constraints()

    def _forbid_unsupported_lengths(self):
        """Forbid white sequences of lengths we don't have words for."""
        available_lengths = set(self.wordlist.keys())
        max_available = max(available_lengths) if available_lengths else MIN_WORD_LENGTH

        # For each length NOT in our wordlist (from MIN_WORD_LENGTH to grid size)
        # Note: lengths 1-2 are handled by _add_min_word_length_constraints
        for length in range(MIN_WORD_LENGTH, max(self.rows, self.cols) + 1):
            if length in available_lengths:
                continue

            # Forbid ACROSS sequences of this length
            for r in range(self.rows):
                for start_c in range(self.cols - length + 1):
                    end_c = start_c + length - 1

                    # Conditions for this slot to exist (we want to FORBID this)
                    # Before: must be black or edge
                    # After: must be black or edge
                    # All cells: must be white

                    # If all these conditions are met, it's invalid
                    # So we add: NOT(before_ok AND after_ok AND all_white)
                    # = NOT before_ok OR NOT after_ok OR NOT all_white

                    literals = []

                    # Before must NOT be (black or edge) = must be white if not edge
                    if start_c > 0:
                        # If cell before is black, that's one condition met
                        # We want at least one condition to NOT be met
                        literals.append(self.B[r][start_c - 1].Not())  # before is white

                    # After must NOT be (black or edge) = must be white if not edge
                    if end_c < self.cols - 1:
                        literals.append(self.B[r][end_c + 1].Not())  # after is white

                    # At least one cell must be black
                    for i in range(length):
                        literals.append(self.B[r][start_c + i])  # cell is black

                    # At least one of these must be true (breaking the forbidden pattern)
                    if literals:
                        self.model.AddBoolOr(literals)

            # Forbid DOWN sequences of this length
            for c in range(self.cols):
                for start_r in range(self.rows - length + 1):
                    end_r = start_r + length - 1

                    literals = []

                    if start_r > 0:
                        literals.append(self.B[start_r - 1][c].Not())

                    if end_r < self.rows - 1:
                        literals.append(self.B[end_r + 1][c].Not())

                    for i in range(length):
                        literals.append(self.B[start_r + i][c])

                    if literals:
                        self.model.AddBoolOr(literals)

    def _add_min_word_length_constraints(self):
        """
        Forbid ALL isolated 1-2 letter sequences (middle AND edges).
        Uses the blog's linear constraint approach.
        """
        # === MIDDLE CONSTRAINTS ===

        # Forbid 1-letter in middle ACROSS: black-white-black
        for r in range(self.rows):
            for c in range(1, self.cols - 1):
                self.model.Add(self.B[r][c - 1] + self.B[r][c + 1] <= 1 + self.B[r][c])

        # Forbid 1-letter in middle DOWN
        for c in range(self.cols):
            for r in range(1, self.rows - 1):
                self.model.Add(self.B[r - 1][c] + self.B[r + 1][c] <= 1 + self.B[r][c])

        # Forbid 2-letter in middle ACROSS: black-white-white-black
        for r in range(self.rows):
            for c in range(1, self.cols - 2):
                self.model.Add(
                    self.B[r][c - 1] + self.B[r][c + 2]
                    <= 1 + self.B[r][c] + self.B[r][c + 1]
                )

        # Forbid 2-letter in middle DOWN
        for c in range(self.cols):
            for r in range(1, self.rows - 2):
                self.model.Add(
                    self.B[r - 1][c] + self.B[r + 2][c]
                    <= 1 + self.B[r][c] + self.B[r + 1][c]
                )

        # === EDGE CONSTRAINTS ===

        # Forbid 1-letter at LEFT edge: edge-white-black
        # If B[0]=white AND B[1]=black, that's forbidden
        # Constraint: B[1] <= B[0] (if B[0]=0, B[1] must be 0)
        for r in range(self.rows):
            self.model.Add(self.B[r][1] <= self.B[r][0])

        # Forbid 1-letter at RIGHT edge
        for r in range(self.rows):
            self.model.Add(self.B[r][self.cols - 2] <= self.B[r][self.cols - 1])

        # Forbid 1-letter at TOP edge
        for c in range(self.cols):
            self.model.Add(self.B[1][c] <= self.B[0][c])

        # Forbid 1-letter at BOTTOM edge
        for c in range(self.cols):
            self.model.Add(self.B[self.rows - 2][c] <= self.B[self.rows - 1][c])

        # Forbid 2-letter at LEFT edge: edge-white-white-black
        # If B[0]=white AND B[1]=white AND B[2]=black, forbidden
        # Constraint: B[2] <= B[0] + B[1]
        for r in range(self.rows):
            self.model.Add(self.B[r][2] <= self.B[r][0] + self.B[r][1])

        # Forbid 2-letter at RIGHT edge
        for r in range(self.rows):
            self.model.Add(
                self.B[r][self.cols - 3]
                <= self.B[r][self.cols - 1] + self.B[r][self.cols - 2]
            )

        # Forbid 2-letter at TOP edge
        for c in range(self.cols):
            self.model.Add(self.B[2][c] <= self.B[0][c] + self.B[1][c])

        # Forbid 2-letter at BOTTOM edge
        for c in range(self.cols):
            self.model.Add(
                self.B[self.rows - 3][c]
                <= self.B[self.rows - 1][c] + self.B[self.rows - 2][c]
            )

    def solve(self, time_limit: int = TIME_LIMIT_SECONDS) -> Optional[Tuple]:
        """
        Solve the model and return the grid.

        Returns:
            Tuple of (grid, word_positions, algorithm_name) or None if failed
        """
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 8  # Use multiple cores

        print(f"Solving (time limit: {time_limit}s)...")
        start_time = time.time()
        status = solver.Solve(self.model)
        solve_time = time.time() - start_time

        status_name = solver.StatusName(status)
        print(f"  Status: {status_name} (solved in {solve_time:.2f}s)")

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print("  No solution found!")
            return None

        # Extract solution
        grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                val = solver.Value(self.L[r][c])
                if val == 0:
                    row.append("#")
                else:
                    row.append(chr(val + ord("A") - 1))
            grid.append(row)

        # Extract word positions
        word_positions = []
        for r, c, length, direction, slot_active in self.word_slots:
            if solver.Value(slot_active):
                if direction == "across":
                    word = "".join(grid[r][c + i] for i in range(length))
                else:
                    word = "".join(grid[r + i][c] for i in range(length))
                word_positions.append(
                    {"word": word, "direction": direction, "row": r, "col": c}
                )

        return grid, word_positions, "Constraint Programming (CP-SAT)"


# =============================================================================
# SIMPLIFIED SOLVER - Pre-filled black squares
# =============================================================================


class SimplifiedCrosswordSolver:
    """
    Simplified solver that works with a pre-defined black square pattern.
    Much faster than the full solver as it doesn't need to decide black placement.
    """

    def __init__(
        self,
        wordlist: Dict[int, List[List[int]]],
        black_pattern: List[List[bool]] = None,
        rows: int = GRID_SIZE,
        cols: int = GRID_SIZE,
    ):
        self.wordlist = wordlist
        self.rows = rows
        self.cols = cols
        self.model = cp_model.CpModel()
        self.L = None
        self.word_slots = []

        # Generate or use provided black pattern
        if black_pattern is None:
            self.black_pattern = self._generate_symmetric_pattern()
        else:
            self.black_pattern = black_pattern

    def _generate_symmetric_pattern(self) -> List[List[bool]]:
        """Generate a symmetric black square pattern that creates word slots matching our wordlist."""
        import random

        available_lengths = set(self.wordlist.keys())
        word_counts = {k: len(v) for k, v in self.wordlist.items()}

        # Find max length we can actually use (need at least 2 words of that length for safety)
        max_usable_length = max(
            (l for l, c in word_counts.items() if c >= 2), default=7
        )

        best_pattern = None
        best_score = -float("inf")

        # Try multiple patterns and pick the best one
        for attempt in range(100):
            pattern = [[False] * self.cols for _ in range(self.rows)]

            # FIRST: ensure every row and column has at least one black
            # For a square grid, both rows and cols use the same word length pool
            # So if we have 1 word of length 15, we can only have 1 total full-span slot
            full_slots_allowed = min(
                word_counts.get(self.cols, 0),  # Words for row spans
                word_counts.get(self.rows, 0),  # Words for col spans (same for square)
            )

            # For simplicity with limited long words, break ALL rows and columns
            # This ensures no full-span slots are required
            if full_slots_allowed < 2:
                full_slots_allowed = 0  # Don't risk it with so few long words

            full_slots_used = 0

            # PHASE 1: Place blacks to break rows/cols while avoiding short slots
            # Use the _would_create_short_slot check for safety

            min_gap = MIN_WORD_LENGTH  # 3

            # First pass: place blacks in rows that need them
            for r in range(self.rows):
                if any(pattern[r][c] for c in range(self.cols)):
                    continue  # Row already has a black

                # Try all valid positions (away from edges)
                positions = list(range(min_gap, self.cols - min_gap))
                random.shuffle(positions)

                placed = False
                for c in positions:
                    if pattern[r][c]:
                        continue
                    if not self._would_create_short_slot(pattern, r, c):
                        pattern[r][c] = True
                        # Try symmetric placement
                        sym_r = self.rows - 1 - r
                        sym_c = self.cols - 1 - c
                        if sym_r != r and not pattern[sym_r][sym_c]:
                            if not self._would_create_short_slot(pattern, sym_r, sym_c):
                                pattern[sym_r][sym_c] = True
                        placed = True
                        break

                # If no valid position found, try near edges (but still >= min_gap from edge)
                if not placed:
                    for c in [min_gap, self.cols - min_gap - 1]:
                        if not pattern[r][c] and not self._would_create_short_slot(
                            pattern, r, c
                        ):
                            pattern[r][c] = True
                            sym_r = self.rows - 1 - r
                            sym_c = self.cols - 1 - c
                            if sym_r != r and not pattern[sym_r][sym_c]:
                                if not self._would_create_short_slot(
                                    pattern, sym_r, sym_c
                                ):
                                    pattern[sym_r][sym_c] = True
                            break

            # Second pass: columns
            for c in range(self.cols):
                if any(pattern[r][c] for r in range(self.rows)):
                    continue

                positions = list(range(min_gap, self.rows - min_gap))
                random.shuffle(positions)

                for r in positions:
                    if pattern[r][c]:
                        continue
                    if not self._would_create_short_slot(pattern, r, c):
                        pattern[r][c] = True
                        sym_r = self.rows - 1 - r
                        sym_c = self.cols - 1 - c
                        if sym_c != c and not pattern[sym_r][sym_c]:
                            if not self._would_create_short_slot(pattern, sym_r, sym_c):
                                pattern[sym_r][sym_c] = True
                        break

            # Now add more blacks to reach target density and create good slot lengths
            target_blacks = int(self.rows * self.cols * random.uniform(0.15, 0.22))
            current_blacks = sum(sum(row) for row in pattern)

            inner_attempts = 0
            while current_blacks < target_blacks and inner_attempts < 500:
                r = random.randint(0, self.rows - 1)
                c = random.randint(0, self.cols - 1)

                if pattern[r][c]:
                    inner_attempts += 1
                    continue

                # Check 3x3 constraint, short slots, and slot lengths
                if self._check_3x3_constraint(
                    pattern, r, c
                ) and not self._would_create_short_slot(pattern, r, c):
                    pattern[r][c] = True
                    sym_r = self.rows - 1 - r
                    sym_c = self.cols - 1 - c

                    # Check symmetric position too
                    if not self._would_create_short_slot(pattern, sym_r, sym_c):
                        pattern[sym_r][sym_c] = True

                        # Verify all slots are of usable lengths
                        if self._has_unusable_slots(
                            pattern, max_usable_length, word_counts
                        ):
                            # Undo
                            pattern[r][c] = False
                            pattern[sym_r][sym_c] = False
                        else:
                            current_blacks = sum(sum(row) for row in pattern)
                    else:
                        # Symmetric position would create short slot, undo
                        pattern[r][c] = False

                inner_attempts += 1

            # Score this pattern
            score = self._score_pattern_v2(pattern, word_counts, max_usable_length)
            if score > best_score:
                best_score = score
                best_pattern = [row[:] for row in pattern]

        if best_pattern is None:
            print("  Using fallback dense pattern...")
            return self._create_dense_pattern(max_usable_length)

        return best_pattern

    def _has_unusable_slots(self, pattern, max_length, word_counts) -> bool:
        """Check if pattern has slots we can't fill (either missing length or need more than available)."""
        slot_needs = {}  # length -> count needed

        # Check across
        for r in range(self.rows):
            c = 0
            while c < self.cols:
                if pattern[r][c]:
                    c += 1
                    continue
                start = c
                while c < self.cols and not pattern[r][c]:
                    c += 1
                length = c - start
                if length >= MIN_WORD_LENGTH:
                    if length > max_length or length not in word_counts:
                        return True
                    slot_needs[length] = slot_needs.get(length, 0) + 1

        # Check down
        for c in range(self.cols):
            r = 0
            while r < self.rows:
                if pattern[r][c]:
                    r += 1
                    continue
                start = r
                while r < self.rows and not pattern[r][c]:
                    r += 1
                length = r - start
                if length >= MIN_WORD_LENGTH:
                    if length > max_length or length not in word_counts:
                        return True
                    slot_needs[length] = slot_needs.get(length, 0) + 1

        # Check we have enough words for each length
        for length, need in slot_needs.items():
            if need > word_counts.get(length, 0):
                return True

        return False

    def _score_pattern_v2(self, pattern, word_counts, max_length) -> int:
        """Score pattern - penalize unusable slots heavily, reward good distribution."""
        score = 0
        slot_needs = {}  # length -> count needed

        # Check across
        for r in range(self.rows):
            c = 0
            while c < self.cols:
                if pattern[r][c]:
                    c += 1
                    continue
                start = c
                while c < self.cols and not pattern[r][c]:
                    c += 1
                length = c - start
                if length >= MIN_WORD_LENGTH:
                    if length > max_length or length not in word_counts:
                        score -= 1000  # Heavy penalty
                    else:
                        slot_needs[length] = slot_needs.get(length, 0) + 1

        # Check down
        for c in range(self.cols):
            r = 0
            while r < self.rows:
                if pattern[r][c]:
                    r += 1
                    continue
                start = r
                while r < self.rows and not pattern[r][c]:
                    r += 1
                length = r - start
                if length >= MIN_WORD_LENGTH:
                    if length > max_length or length not in word_counts:
                        score -= 1000
                    else:
                        slot_needs[length] = slot_needs.get(length, 0) + 1

        # Check if we have enough words
        for length, need in slot_needs.items():
            available = word_counts.get(length, 0)
            if need > available:
                score -= (need - available) * 100
            else:
                score += available - need  # Reward having slack

        return score

    def _create_dense_pattern(self, max_length) -> List[List[bool]]:
        """Create a pattern with many blacks to ensure short slots."""
        pattern = [[False] * self.cols for _ in range(self.rows)]

        # Place blacks every max_length cells to ensure slots are short enough
        spacing = max(3, max_length - 1)
        for r in range(self.rows):
            for c in range(spacing - 1, self.cols, spacing):
                if c < self.cols:
                    pattern[r][c] = True
        for c in range(self.cols):
            for r in range(spacing - 1, self.rows, spacing):
                if r < self.rows:
                    pattern[r][c] = True

        return pattern

    def _count_bad_slots(
        self, pattern: List[List[bool]], available_lengths: set
    ) -> int:
        """Count slots that don't match available word lengths."""
        bad = 0

        # Check across
        for r in range(self.rows):
            c = 0
            while c < self.cols:
                if pattern[r][c]:
                    c += 1
                    continue
                start = c
                while c < self.cols and not pattern[r][c]:
                    c += 1
                length = c - start
                if length >= MIN_WORD_LENGTH and length not in available_lengths:
                    bad += 1

        # Check down
        for c in range(self.cols):
            r = 0
            while r < self.rows:
                if pattern[r][c]:
                    r += 1
                    continue
                start = r
                while r < self.rows and not pattern[r][c]:
                    r += 1
                length = r - start
                if length >= MIN_WORD_LENGTH and length not in available_lengths:
                    bad += 1

        return bad

    def _score_pattern(self, pattern: List[List[bool]], available_lengths: set) -> int:
        """Score a pattern by how many valid word slots it creates."""
        score = 0

        # Check across
        for r in range(self.rows):
            c = 0
            while c < self.cols:
                if pattern[r][c]:
                    c += 1
                    continue
                start = c
                while c < self.cols and not pattern[r][c]:
                    c += 1
                length = c - start
                if length in available_lengths:
                    # Bonus for having many words of this length
                    score += len(self.wordlist.get(length, []))
                elif length >= MIN_WORD_LENGTH:
                    score -= 10  # Penalty for unusable slots

        # Check down
        for c in range(self.cols):
            r = 0
            while r < self.rows:
                if pattern[r][c]:
                    r += 1
                    continue
                start = r
                while r < self.rows and not pattern[r][c]:
                    r += 1
                length = r - start
                if length in available_lengths:
                    score += len(self.wordlist.get(length, []))
                elif length >= MIN_WORD_LENGTH:
                    score -= 10

        return score

    def _check_3x3_constraint(self, pattern: List[List[bool]], r: int, c: int) -> bool:
        """Check if placing a black at (r,c) violates 3x3 constraint."""
        # Check all 3x3 subgrids that would include this cell
        for start_r in range(max(0, r - 2), min(self.rows - 2, r + 1)):
            for start_c in range(max(0, c - 2), min(self.cols - 2, c + 1)):
                count = sum(
                    1
                    for i in range(3)
                    for j in range(3)
                    if pattern[start_r + i][start_c + j]
                )
                # Adding this cell would make it 3 (too many)
                if count >= 2:
                    return False
        return True

    def _would_create_short_slot(
        self, pattern: List[List[bool]], r: int, c: int
    ) -> bool:
        """Check if placing a black at (r,c) would create a slot of length 1-2."""
        # Temporarily place the black
        pattern[r][c] = True

        # Check horizontal slots around this position
        # Check left
        left_len = 0
        for i in range(c - 1, -1, -1):
            if pattern[r][i]:
                break
            left_len += 1

        # Check right
        right_len = 0
        for i in range(c + 1, self.cols):
            if pattern[r][i]:
                break
            right_len += 1

        # If placing black creates a short slot on left or right
        if 0 < left_len < MIN_WORD_LENGTH or 0 < right_len < MIN_WORD_LENGTH:
            pattern[r][c] = False
            return True

        # Check vertical slots around this position
        # Check up
        up_len = 0
        for i in range(r - 1, -1, -1):
            if pattern[i][c]:
                break
            up_len += 1

        # Check down
        down_len = 0
        for i in range(r + 1, self.rows):
            if pattern[i][c]:
                break
            down_len += 1

        # If placing black creates a short slot above or below
        if 0 < up_len < MIN_WORD_LENGTH or 0 < down_len < MIN_WORD_LENGTH:
            pattern[r][c] = False
            return True

        pattern[r][c] = False
        return False

    def build_model(self):
        """Build model with fixed black pattern."""
        print("Building simplified model with fixed black pattern...")

        # Create letter variables only for white cells
        self.L = [[None] * self.cols for _ in range(self.rows)]

        for r in range(self.rows):
            for c in range(self.cols):
                if not self.black_pattern[r][c]:
                    self.L[r][c] = self.model.NewIntVar(1, 26, f"L[{r}][{c}]")

        # Find and constrain word slots
        self._find_and_constrain_slots()

        print(f"  Model built with {len(self.word_slots)} word slots")

    def _find_and_constrain_slots(self):
        """Find all word slots and add table constraints."""
        slot_lengths = {}  # Track slot length distribution
        missing_lengths = []  # Track slots with no matching words

        # ACROSS slots
        for r in range(self.rows):
            c = 0
            while c < self.cols:
                if self.black_pattern[r][c]:
                    c += 1
                    continue

                # Find end of white sequence
                start_c = c
                while c < self.cols and not self.black_pattern[r][c]:
                    c += 1
                end_c = c - 1
                length = end_c - start_c + 1

                if length >= MIN_WORD_LENGTH:
                    slot_lengths[length] = slot_lengths.get(length, 0) + 1
                    if length in self.wordlist:
                        slot_vars = [self.L[r][start_c + i] for i in range(length)]
                        self.model.AddAllowedAssignments(
                            slot_vars, self.wordlist[length]
                        )
                        self.word_slots.append((r, start_c, length, "across"))
                    else:
                        missing_lengths.append(length)

        # DOWN slots
        for c in range(self.cols):
            r = 0
            while r < self.rows:
                if self.black_pattern[r][c]:
                    r += 1
                    continue

                # Find end of white sequence
                start_r = r
                while r < self.rows and not self.black_pattern[r][c]:
                    r += 1
                end_r = r - 1
                length = end_r - start_r + 1

                if length >= MIN_WORD_LENGTH:
                    slot_lengths[length] = slot_lengths.get(length, 0) + 1
                    if length in self.wordlist:
                        slot_vars = [self.L[start_r + i][c] for i in range(length)]
                        self.model.AddAllowedAssignments(
                            slot_vars, self.wordlist[length]
                        )
                        self.word_slots.append((start_r, c, length, "down"))
                    else:
                        missing_lengths.append(length)

        print(f"  Slot lengths needed: {dict(sorted(slot_lengths.items()))}")
        print(
            f"  Words available: {dict((k, len(v)) for k, v in sorted(self.wordlist.items()))}"
        )
        if missing_lengths:
            print(f"  ⚠ Missing words for lengths: {set(missing_lengths)}")

    def solve(self, time_limit: int = TIME_LIMIT_SECONDS) -> Optional[Tuple]:
        """Solve and return solution."""
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = 8

        print(f"Solving (time limit: {time_limit}s)...")
        start_time = time.time()
        status = solver.Solve(self.model)
        solve_time = time.time() - start_time

        status_name = solver.StatusName(status)
        print(f"  Status: {status_name} (solved in {solve_time:.2f}s)")

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return None

        # Extract grid
        grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if self.black_pattern[r][c]:
                    row.append("#")
                else:
                    val = solver.Value(self.L[r][c])
                    row.append(chr(val + ord("A") - 1))
            grid.append(row)

        # Extract words
        word_positions = []
        for r, c, length, direction in self.word_slots:
            if direction == "across":
                word = "".join(grid[r][c + i] for i in range(length))
            else:
                word = "".join(grid[r + i][c] for i in range(length))
            word_positions.append(
                {"word": word, "direction": direction, "row": r, "col": c}
            )

        return grid, word_positions, "Constraint Programming (Simplified)"


# =============================================================================
# GENERATE FUNCTION (compatible with other algorithms)
# =============================================================================


def generate(
    words: List[str],
    grid_size: int = GRID_SIZE,
    seed: int = None,
    use_simplified: bool = True,
    time_limit: int = TIME_LIMIT_SECONDS,
    max_attempts: int = 5,
) -> Optional[Tuple]:
    """
    Generate a crossword using constraint programming.

    Args:
        words: List of words to use
        grid_size: Size of the grid (default 15x15)
        seed: Random seed for pattern generation
        use_simplified: Use simplified solver with pre-generated pattern
        time_limit: Solving time limit in seconds
        max_attempts: Maximum number of pattern attempts for simplified solver

    Returns:
        Tuple of (grid, word_positions, algorithm_name) or None
    """
    import random

    if seed is not None:
        random.seed(seed)

    # Build wordlist
    wordlist = {}
    for word in words:
        word = word.upper()
        if word.isalpha() and MIN_WORD_LENGTH <= len(word) <= grid_size:
            length = len(word)
            word_nums = word_to_numbers(word)
            if length in wordlist:
                wordlist[length].append(word_nums)
            else:
                wordlist[length] = [word_nums]

    if not use_simplified:
        solver = CrosswordSolver(wordlist, rows=grid_size, cols=grid_size)
        solver.build_model()
        return solver.solve(time_limit=time_limit)

    # For simplified solver, try multiple patterns
    per_attempt_time = max(10, time_limit // max_attempts)

    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt + 1}/{max_attempts}...")
        solver = SimplifiedCrosswordSolver(wordlist, rows=grid_size, cols=grid_size)
        solver.build_model()
        result = solver.solve(time_limit=per_attempt_time)
        if result:
            return result

    return None


# =============================================================================
# STATISTICS AND HTML OUTPUT
# =============================================================================


def calculate_statistics(grid, word_positions, input_words: List[str]) -> dict:
    """Calculate statistics for a generated crossword."""
    stats = {
        "avg_word_length_ratio": 0,
        "black_square_percentage": 0,
        "vertical_horizontal_ratio": 0,
        "num_words": 0,
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
                if direction in ["down", "vertical"]:
                    vertical_count += 1
                else:
                    horizontal_count += 1

    stats["num_words"] = len(placed_words)

    if placed_words and input_avg_length > 0:
        placed_avg_length = sum(len(w) for w in placed_words) / len(placed_words)
        stats["avg_word_length_ratio"] = placed_avg_length / input_avg_length

    total_cells = grid_size * grid_size
    black_cells = sum(1 for row in grid for cell in row if cell == "#")
    stats["black_square_percentage"] = (black_cells / total_cells) * 100

    if horizontal_count > 0:
        stats["vertical_horizontal_ratio"] = vertical_count / horizontal_count

    return stats


def generate_html(grid, word_positions, stats: dict, grid_size: int) -> str:
    """Generate HTML visualization of the crossword."""
    if grid is None:
        return "<p>Failed to generate crossword</p>"

    # Build word numbers
    word_numbers = {}
    current_number = 1
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] == "#":
                continue
            starts_horiz = (j == 0 or grid[i][j - 1] == "#") and (
                j < grid_size - 1 and grid[i][j + 1] != "#"
            )
            starts_vert = (i == 0 or grid[i - 1][j] == "#") and (
                i < grid_size - 1 and grid[i + 1][j] != "#"
            )
            if starts_horiz or starts_vert:
                word_numbers[(i, j)] = current_number
                current_number += 1

    grid_html = (
        f'<div class="grid" style="grid-template-columns: repeat({grid_size}, 1fr);">'
    )
    for i in range(grid_size):
        for j in range(grid_size):
            cell = grid[i][j]
            if cell != "#":
                num = word_numbers.get((i, j), "")
                num_html = f'<span class="word-number">{num}</span>' if num else ""
                grid_html += f'<div class="cell white">{num_html}<span class="letter">{cell}</span></div>'
            else:
                grid_html += '<div class="cell black"></div>'
    grid_html += "</div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CP Crossword</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: linear-gradient(145deg, #0d0d0d 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            padding: 40px 20px;
            color: #e0e0e0;
        }}
        h1 {{
            text-align: center;
            font-size: 2rem;
            margin-bottom: 30px;
            color: #00ff9f;
            text-shadow: 0 0 20px rgba(0, 255, 159, 0.3);
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
        }}
        .grid {{
            display: grid;
            gap: 2px;
            background: #333;
            padding: 2px;
            border-radius: 8px;
            margin-bottom: 30px;
            aspect-ratio: 1;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }}
        .cell {{
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: clamp(10px, 2vw, 16px);
            text-transform: uppercase;
            position: relative;
        }}
        .cell.white {{
            background: #fafafa;
            color: #1a1a2e;
        }}
        .cell.black {{
            background: #1a1a2e;
        }}
        .word-number {{
            position: absolute;
            top: 2px;
            left: 3px;
            font-size: clamp(6px, 1vw, 9px);
            font-weight: 400;
            color: #666;
        }}
        .stats {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
        }}
        .stats h2 {{
            font-size: 1rem;
            margin-bottom: 15px;
            color: #00ff9f;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .stat-row:last-child {{ border-bottom: none; }}
        .stat-label {{ color: #888; }}
        .stat-value {{ color: #00d9ff; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Constraint Programming Crossword</h1>
        {grid_html}
        <div class="stats">
            <h2>Statistics</h2>
            <div class="stat-row">
                <span class="stat-label">Words placed</span>
                <span class="stat-value">{stats.get('num_words', 0)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Black squares</span>
                <span class="stat-value">{stats.get('black_square_percentage', 0):.1f}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">V/H ratio</span>
                <span class="stat-value">{stats.get('vertical_horizontal_ratio', 0):.2f}</span>
            </div>
        </div>
    </div>
</body>
</html>"""


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate crossword using OR-Tools CP-SAT constraint programming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python algos/constraint_programming.py sample_words.txt
  python algos/constraint_programming.py sample_words.txt --full
  python algos/constraint_programming.py sample_words.txt -t 60 -o my_crossword.html
        """,
    )
    parser.add_argument("words_file", help="Path to words file (one word per line)")
    parser.add_argument(
        "-o",
        "--output",
        default="constraint_variation.html",
        help="Output HTML file (default: constraint_variation.html)",
    )
    parser.add_argument(
        "-t",
        "--time-limit",
        type=int,
        default=TIME_LIMIT_SECONDS,
        help=f"Solving time limit in seconds (default: {TIME_LIMIT_SECONDS})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full solver (slower, determines black pattern)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-g",
        "--grid-size",
        type=int,
        default=GRID_SIZE,
        help=f"Grid size (default: {GRID_SIZE})",
    )

    args = parser.parse_args()

    print(f"Loading words from {args.words_file}...")
    words = load_words_raw(args.words_file)
    print(f"Loaded {len(words)} valid words")

    # Show word length distribution
    length_dist = {}
    for w in words:
        l = len(w)
        length_dist[l] = length_dist.get(l, 0) + 1
    print(f"Word lengths: {dict(sorted(length_dist.items()))}")

    print(f"\nGenerating {args.grid_size}x{args.grid_size} crossword...")
    result = generate(
        words,
        grid_size=args.grid_size,
        seed=args.seed,
        use_simplified=not args.full,
        time_limit=args.time_limit,
    )

    if result:
        grid, word_positions, algo_name = result
        stats = calculate_statistics(grid, word_positions, words)

        print(f"\n✓ Generated crossword:")
        print(f"  Words: {stats['num_words']}")
        print(f"  Black: {stats['black_square_percentage']:.1f}%")

        # Print grid to console
        print("\nGrid:")
        for row in grid:
            print("  " + " ".join(c if c != "#" else "█" for c in row))

        # Generate HTML
        html = generate_html(grid, word_positions, stats, args.grid_size)
        with open(args.output, "w") as f:
            f.write(html)
        print(f"\nSaved to {args.output}")
    else:
        print("\n✗ Failed to generate crossword")
        print("  Try:")
        print("  - Adding more words to the word list")
        print("  - Increasing time limit with -t")
        print("  - Using simplified solver (default)")
        sys.exit(1)


if __name__ == "__main__":
    main()
