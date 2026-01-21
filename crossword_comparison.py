#!/usr/bin/env python3
"""
Compare different crossword generation algorithms.
Takes a text file with newline-separated words and generates crosswords using 4 different approaches.
"""

import sys
import random
from typing import List, Tuple, Optional

from html_generator import generate_html_page
from algos import custom_backtracking

GRID_SIZE = 15


def read_words_from_file(filename: str) -> List[str]:
    """Read words from a newline-separated text file."""
    with open(filename, "r") as f:
        words = [line.strip().upper() for line in f if line.strip()]
    # Filter to words that fit in grid
    words = [w for w in words if len(w) <= GRID_SIZE]
    return words


def center_grid_with_black_border(grid, target_size, target_black_percentage=20):
    """Center the content of a grid within target_size, filling edges with black squares."""
    if not grid:
        return [["#" for _ in range(target_size)] for _ in range(target_size)]

    # Convert to list if needed
    if hasattr(grid, "tolist"):
        grid = grid.tolist()

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Find bounding box of actual content (letters)
    min_row, max_row = rows, -1
    min_col, max_col = cols, -1

    for i in range(rows):
        for j in range(cols):
            cell = grid[i][j]
            if cell and cell not in ["#", " ", ".", None, "", 0]:
                min_row = min(min_row, i)
                max_row = max(max_row, i)
                min_col = min(min_col, j)
                max_col = max(max_col, j)

    if max_row < 0:  # No content found
        return [["#" for _ in range(target_size)] for _ in range(target_size)]

    # Calculate content dimensions
    content_height = max_row - min_row + 1
    content_width = max_col - min_col + 1

    # Calculate offset to center content
    offset_row = (target_size - content_height) // 2
    offset_col = (target_size - content_width) // 2

    # Create new grid with black background
    result = [["#" for _ in range(target_size)] for _ in range(target_size)]

    # Copy content to centered position
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            cell = grid[i][j]
            new_i = offset_row + (i - min_row)
            new_j = offset_col + (j - min_col)
            if 0 <= new_i < target_size and 0 <= new_j < target_size:
                if cell and cell not in ["#", " ", ".", None, "", 0]:
                    result[new_i][new_j] = str(cell)

    return result


def generate_with_puzzlecreator(words: List[str]) -> Optional[Tuple]:
    """Generate crossword using puzzlecreator library (Greedy + Scoring)."""
    try:
        from puzzlecreator import crossword

        # Generate crossword with time limit
        result = crossword.repeated_word_placement(words, max_time=10)

        if result:
            # Get word positions
            positions_across, positions_down = (
                crossword.get_ordered_word_positions_with_numbers(result, words)
            )

            # Combine positions into a format we can use
            word_positions = []
            for item in positions_across:
                word_positions.append({"word": item[0], "direction": "across"})
            for item in positions_down:
                word_positions.append({"word": item[0], "direction": "down"})

            # Center grid with black border to achieve ~20% black squares (NYT typical)
            grid = center_grid_with_black_border(result, GRID_SIZE)

            return grid, word_positions, "PuzzleCreator (Greedy + Scoring)"
        return None
    except Exception as e:
        print(f"PuzzleCreator error: {e}")
        return None


def generate_with_crossword_generator(words: List[str]) -> Optional[Tuple]:
    """Generate crossword using crossword-generator library (MCTS)."""
    try:
        from crossword_generator import generate_crossword
        import tempfile
        import os
        import csv

        # Create word list CSV with 'answer' column
        words_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        words_file.write("answer,clue\n")
        for word in words:
            words_file.write(f"{word},clue\n")
        words_file.close()

        # Create output file path (just get a unique path, don't create the file)
        out_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        out_path = out_file.name
        out_file.close()
        os.unlink(out_path)  # Delete it so the library can create it

        # Generate crossword
        generate_crossword(
            num_rows=GRID_SIZE,
            num_cols=GRID_SIZE,
            path_to_words=words_file.name,
            output_path=out_path,
            max_mcts_iterations=1000,
        )

        # Read the output file to get the grid
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            grid = []
            word_positions = []
            with open(out_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        grid.append([c if c != "_" else "#" for c in row])

            # Ensure grid is 15x15
            grid = pad_grid_to_size(grid, GRID_SIZE)

            # Extract word positions from grid
            word_positions = extract_words_from_grid(grid)

            # Cleanup temp files
            os.unlink(words_file.name)
            os.unlink(out_path)

            return grid, word_positions, "Crossword-Generator (MCTS)"

        # Cleanup
        os.unlink(words_file.name)
        if os.path.exists(out_path):
            os.unlink(out_path)
        return None
    except Exception as e:
        print(f"Crossword-Generator error: {e}")
        return None


def generate_with_blacksquare(words: List[str]) -> Optional[Tuple]:
    """Generate crossword using blacksquare library (Beam Search)."""
    try:
        import blacksquare as bsq
        import numpy as np

        # Create a grid pattern with black squares (typical crossword pattern)
        # Start with empty grid, add some black squares for structure
        grid_pattern = np.full((GRID_SIZE, GRID_SIZE), " ", dtype=object)

        # Add black squares in a symmetric pattern (like real crosswords)
        black_positions = [
            (0, 4),
            (0, 10),
            (1, 4),
            (1, 10),
            (2, 4),
            (2, 10),
            (3, 0),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 14),
            (4, 0),
            (4, 1),
            (4, 5),
            (4, 9),
            (4, 13),
            (4, 14),
            (5, 5),
            (5, 9),
            (6, 3),
            (6, 11),
            (7, 3),
            (7, 7),
            (7, 11),
            (8, 3),
            (8, 11),
            (9, 5),
            (9, 9),
            (10, 0),
            (10, 1),
            (10, 5),
            (10, 9),
            (10, 13),
            (10, 14),
            (11, 0),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 14),
            (12, 4),
            (12, 10),
            (13, 4),
            (13, 10),
            (14, 4),
            (14, 10),
        ]

        for r, c in black_positions:
            if r < GRID_SIZE and c < GRID_SIZE:
                grid_pattern[r, c] = bsq.BLACK

        # Create crossword from pattern
        xw = bsq.Crossword(grid=grid_pattern)

        # Create a WordList from our words
        wordlist = bsq.WordList(words)

        # Try to fill the grid with our words only
        filled = xw.fill(word_list=wordlist, timeout=30.0)

        if filled is not None:
            xw = filled

        # Extract grid
        grid = []
        for i in range(GRID_SIZE):
            row = []
            for j in range(GRID_SIZE):
                cell = xw[i, j]
                if cell == bsq.BLACK:
                    row.append("#")
                elif cell == bsq.EMPTY:
                    row.append("#")
                else:
                    # Get the actual letter value
                    val = cell.value
                    if val == bsq.EMPTY or val == bsq.BLACK or val is None:
                        row.append("#")
                    else:
                        row.append(str(val))
            grid.append(row)

        # Get word positions
        word_positions = []
        for word in xw.iterwords():
            val = word.value.strip() if word.value else ""
            if val:
                word_positions.append(
                    {
                        "word": val,
                        "direction": (
                            "across" if word.direction == bsq.ACROSS else "down"
                        ),
                    }
                )

        return grid, word_positions, "Blacksquare (Beam Search)"
    except Exception as e:
        print(f"Blacksquare error: {e}")
        return None


def generate_with_backtracking(words: List[str]) -> Optional[Tuple]:
    """Generate crossword using custom backtracking algorithm."""
    return custom_backtracking.generate(words, GRID_SIZE)


def pad_grid_to_size(grid, size):
    """Pad or trim grid to specified size."""
    if not grid:
        return [["#" for _ in range(size)] for _ in range(size)]

    # Convert to list of lists if needed
    if hasattr(grid, "tolist"):
        grid = grid.tolist()

    result = []
    for i in range(size):
        row = []
        for j in range(size):
            if i < len(grid) and j < len(grid[i]):
                cell = grid[i][j]
                if cell in [None, "", " ", ".", 0]:
                    row.append("#")
                else:
                    row.append(str(cell))
            else:
                row.append("#")
        result.append(row)
    return result


def extract_words_from_grid(grid):
    """Extract word positions from a filled grid."""
    words = []
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Find horizontal words
    for i in range(rows):
        j = 0
        while j < cols:
            if grid[i][j] != "#":
                word = ""
                start_j = j
                while j < cols and grid[i][j] != "#":
                    word += grid[i][j]
                    j += 1
                if len(word) > 1:
                    words.append({"word": word, "direction": "across"})
            else:
                j += 1

    # Find vertical words
    for j in range(cols):
        i = 0
        while i < rows:
            if grid[i][j] != "#":
                word = ""
                start_i = i
                while i < rows and grid[i][j] != "#":
                    word += grid[i][j]
                    i += 1
                if len(word) > 1:
                    words.append({"word": word, "direction": "down"})
            else:
                i += 1

    return words


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

    # Calculate input words average length
    input_avg_length = (
        sum(len(w) for w in input_words) / len(input_words) if input_words else 0
    )

    # Count vertical and horizontal words
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

    # Calculate average word length ratio
    if placed_words and input_avg_length > 0:
        placed_avg_length = sum(len(w) for w in placed_words) / len(placed_words)
        stats["avg_word_length_ratio"] = placed_avg_length / input_avg_length

    # Calculate black square percentage
    total_cells = GRID_SIZE * GRID_SIZE
    black_cells = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            cell = grid[i][j]
            if cell in ["#", " ", ".", None, "", 0]:
                black_cells += 1

    stats["black_square_percentage"] = (black_cells / total_cells) * 100

    # Calculate vertical to horizontal ratio
    if horizontal_count > 0:
        stats["vertical_horizontal_ratio"] = vertical_count / horizontal_count
    elif vertical_count > 0:
        stats["vertical_horizontal_ratio"] = float("inf")

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python crossword_comparison.py <words_file.txt>")
        sys.exit(1)

    words_file = sys.argv[1]

    print(f"Reading words from {words_file}...")
    words = read_words_from_file(words_file)
    print(
        f"Loaded {len(words)} words: {', '.join(words[:5])}{'...' if len(words) > 5 else ''}"
    )

    # Generate crosswords with all 4 algorithms
    print("\nGenerating crosswords...")

    print("1. Trying PuzzleCreator...")
    result1 = generate_with_puzzlecreator(words)

    print("2. Trying Crossword-Generator (MCTS)...")
    result2 = generate_with_crossword_generator(words)

    print("3. Trying Blacksquare...")
    result3 = generate_with_blacksquare(words)

    print("4. Trying Custom Backtracking...")
    result4 = generate_with_backtracking(words)

    # Calculate and print statistics
    print("\n" + "=" * 80)
    print("CROSSWORD GENERATION STATISTICS")
    print("=" * 80)

    input_avg_length = sum(len(w) for w in words) / len(words) if words else 0
    print(f"\nInput words average length: {input_avg_length:.2f}")
    print(f"Total input words: {len(words)}")
    print("\n" + "-" * 80)

    results = [result1, result2, result3, result4]

    for idx, result in enumerate(results):
        if result:
            grid, word_positions, title = result
            stats = calculate_statistics(grid, word_positions, words)

            print(f"\n{title}:")
            print(f"  Words placed: {stats['num_words']}")
            print(
                f"  1. Avg word length ratio: {stats['avg_word_length_ratio']:.2f}x (vs input avg)"
            )
            print(f"  2. Black squares: {stats['black_square_percentage']:.1f}%")
            if stats["vertical_horizontal_ratio"] == float("inf"):
                print(f"  3. Vertical/Horizontal ratio: ∞ (no horizontal words)")
            else:
                print(
                    f"  3. Vertical/Horizontal ratio: {stats['vertical_horizontal_ratio']:.2f}"
                )
        else:
            print(f"\nAlgorithm {idx+1}: FAILED")

    print("\n" + "=" * 80)

    # Generate HTML visualization
    print("\nGenerating HTML visualization...")

    input_stats = {
        "total_words": len(words),
        "avg_length": input_avg_length,
        "words": words,
    }

    html_content = generate_html_page(
        results, input_stats, GRID_SIZE, calculate_statistics
    )

    output_file = "crossword_comparison.html"
    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"\nSaved visualization to {output_file}")
    print(f"Open in browser: file://{sys.path[0]}/{output_file}")


if __name__ == "__main__":
    main()
