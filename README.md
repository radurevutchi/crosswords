# Crossword Generation Algorithm Comparison

A tool to compare different crossword generation algorithms side-by-side using the same word list and grid constraints.

## Overview

This project generates crossword puzzles using **4 different algorithms** and visualizes the results in an HTML page, allowing you to compare and contrast each approach's strengths and weaknesses.

---

## The 4 Algorithms Compared

### 1. PuzzleCreator — Greedy Repeated Placement with Scoring

**Library:** `puzzlecreator`

**Algorithm Type:** Iterative Greedy Placement with Multi-Attempt Optimization

**How it works:**
1. Takes a word list and attempts to place words on a grid
2. For each word, finds all possible intersection points with already-placed words (matching letters)
3. Scores each potential placement based on:
   - Number of intersections with existing words
   - Grid compactness (prefers smaller grids)
   - Proximity to existing word clusters
4. Places the word at the highest-scoring position
5. Repeats the entire process multiple times within a time limit (`max_time`)
6. Returns the best-scoring crossword from all attempts
7. Trims empty rows/columns to compact the final grid

**Characteristics:**
- **Speed:** Very Fast (milliseconds to seconds)
- **Quality:** Fair to Good (depends on time allowed and word list)
- **Determinism:** Non-deterministic (randomized starting placements)
- **Backtracking:** No explicit backtracking; relies on multiple attempts

**Trade-offs:**
- ✅ Simple and fast
- ✅ Works with any word list
- ❌ Greedy decisions can't be reversed
- ❌ May leave many words unplaced
- ❌ No guaranteed grid size

---

### 2. Crossword-Generator — Monte Carlo Tree Search (MCTS)

**Library:** `crossword-generator` by Jonas Schumacher  
**Source:** [github.com/jonas-schumacher/crossword-generator](https://github.com/jonas-schumacher/crossword-generator)

**Algorithm Type:** Monte Carlo Tree Search (MCTS)

**How it works:**
1. Models crossword generation as a tree search problem
2. Each node in the tree represents a partial crossword state
3. Uses the MCTS cycle:
   - **Selection:** Choose most promising node using UCB1 (Upper Confidence Bound)
   - **Expansion:** Add new child nodes representing word placements
   - **Simulation:** Random playout to terminal state (filled grid or failure)
   - **Backpropagation:** Update node statistics based on simulation result
4. Balances **exploration** (trying new placements) vs **exploitation** (refining good solutions)
5. After many iterations, selects the best path through the tree

**Characteristics:**
- **Speed:** Slow to Medium (seconds to minutes depending on iterations)
- **Quality:** Excellent (near-optimal solutions with enough time)
- **Determinism:** Non-deterministic (randomized simulations)
- **Backtracking:** Implicit through tree structure

**Trade-offs:**
- ✅ Finds high-quality solutions
- ✅ Handles complex constraints well
- ✅ Works with predefined grid layouts
- ❌ Computationally expensive
- ❌ Requires tuning of exploration parameters
- ❌ No guaranteed solution

---

### 3. Blacksquare — Word List Matching with Beam Search

**Library:** `blacksquare` by pmaher86  
**Source:** [github.com/pmaher86/blacksquare](https://github.com/pmaher86/blacksquare)

**Algorithm Type:** Constraint-Based Fill with Heuristic Search

**How it works:**
1. Creates a grid structure with defined word slots (across and down)
2. For each empty slot, finds all words from the word list that could fit
3. Uses **beam search** to explore multiple partial solutions simultaneously:
   - Maintains a "beam" of the k best partial solutions
   - Expands each solution by trying words in unfilled slots
   - Prunes to keep only the top k solutions at each step
4. Applies constraint propagation: when a word is placed, eliminates incompatible words from crossing slots
5. Continues until grid is filled or no valid words remain

**Characteristics:**
- **Speed:** Medium (seconds)
- **Quality:** Good to Excellent (depends on word list coverage)
- **Determinism:** Mostly deterministic (beam search is systematic)
- **Backtracking:** Implicit through beam pruning

**Trade-offs:**
- ✅ Works well with comprehensive word lists
- ✅ Good Jupyter notebook integration
- ✅ Exports to .puz format
- ❌ Requires pre-defined grid structure
- ❌ May fail if word list doesn't cover constraints
- ❌ Less flexible about word selection

---

### 4. Custom Backtracking — Greedy Intersection-Based Placement

**Algorithm Type:** Greedy Placement with Intersection Prioritization

**How it works:**
1. Sorts words by length (longest first)
2. Places the first word in the center of the grid horizontally
3. For each subsequent word:
   - Scans the entire grid for valid placement positions
   - Checks constraints: word must fit, no letter conflicts
   - Counts intersections with already-placed words
   - Only places words that have **at least one intersection**
   - Chooses the placement with the **maximum intersections**
4. Continues until all words are attempted or word limit reached

**Characteristics:**
- **Speed:** Fast (milliseconds)
- **Quality:** Fair (simple heuristic)
- **Determinism:** Deterministic (same input → same output)
- **Backtracking:** None (pure greedy)

**Trade-offs:**
- ✅ Very fast
- ✅ Predictable behavior
- ✅ Easy to understand and modify
- ❌ No backtracking means early mistakes persist
- ❌ May produce sparse grids
- ❌ Quality depends heavily on word order

---

## Algorithm Comparison Table

| Algorithm | Speed | Quality | Deterministic | Handles Constraints | Best Use Case |
|-----------|-------|---------|---------------|---------------------|---------------|
| **PuzzleCreator** | ⚡⚡⚡ Very Fast | ⭐⭐ Fair | No | Basic | Quick prototyping |
| **MCTS** | ⚡ Slow | ⭐⭐⭐⭐ Excellent | No | Complex | Publication-quality |
| **Blacksquare** | ⚡⚡ Medium | ⭐⭐⭐ Good | Yes | Medium | Word list matching |
| **Backtracking** | ⚡⚡⚡ Very Fast | ⭐⭐ Fair | Yes | Basic | Learning/baseline |

---

## Key Algorithmic Differences

### Search Strategy
- **Greedy (PuzzleCreator, Backtracking):** Makes locally optimal choices, no lookahead
- **MCTS:** Probabilistically explores many futures, learns from simulations
- **Beam Search (Blacksquare):** Maintains multiple candidates, prunes systematically

### Constraint Handling
- **Greedy:** Only checks conflicts at placement time
- **MCTS:** Evaluates constraint satisfaction through simulation outcomes
- **Beam Search:** Propagates constraints to eliminate invalid options early

### Optimality Guarantees
- **Greedy:** No guarantee, first valid solution accepted
- **MCTS:** Statistical convergence to near-optimal with enough iterations
- **Beam Search:** Finds good solutions within beam width, may miss global optimum

### Scalability
- **Greedy:** O(n²) - scales well
- **MCTS:** O(iterations × simulation_depth) - tunable
- **Beam Search:** O(beam_width × slots × words) - depends on word list size

---

## Metrics Reported

For each algorithm, the comparison reports:

1. **Average Word Length Ratio** — How the placed words' average length compares to input words (1.0 = same, >1.0 = longer words placed)

2. **Black Square Percentage** — Percentage of grid cells that are empty/black (lower = denser crossword)

3. **Vertical/Horizontal Ratio** — Balance between down and across words (1.0 = perfectly balanced)

---

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- `puzzlecreator` — Greedy placement algorithm
- `crossword-generator` — MCTS algorithm
- `blacksquare` — Beam search fill algorithm

---

## Usage

```bash
python crossword_comparison.py <words_file.txt>
```

Example:
```bash
python crossword_comparison.py sample_words.txt
```

The script will:
1. Read words from the input file (one word per line)
2. Generate crosswords using all 4 algorithms
3. Print statistics to the terminal
4. Save an HTML visualization to `crossword_comparison.html`

---

## Input Format

Words file should contain one word per line:
```
PYTHON
CODE
ALGORITHM
DATA
COMPUTER
```

---

## Output

- **Terminal:** Statistics for each algorithm (words placed, ratios, percentages)
- **HTML:** Visual comparison at `crossword_comparison.html`

---

## Project Structure

```
crossword/
├── crossword_comparison.py  # Main script with algorithm implementations
├── html_generator.py        # HTML visualization generator
├── sample_words.txt         # Example word list
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## References

- [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [Constraint Satisfaction Problems](https://en.wikipedia.org/wiki/Constraint_satisfaction_problem)
- [Beam Search](https://en.wikipedia.org/wiki/Beam_search)
- [crossword-generator GitHub](https://github.com/jonas-schumacher/crossword-generator)
- [blacksquare GitHub](https://github.com/pmaher86/blacksquare)
- [CS50 AI Crossword Project](https://cs50.harvard.edu/ai/projects/3/crossword/)

---

## License

MIT
