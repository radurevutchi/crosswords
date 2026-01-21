"""
HTML generator for crossword visualization.
"""

from typing import List


def is_white_cell(grid, row, col, rows, cols):
    """Check if a cell is a white (letter) cell."""
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return False
    cell = grid[row][col]
    return cell and cell not in ["#", " ", ".", None, 0, ""]


def get_word_numbers(grid):
    """Calculate word numbers for a grid like NYT crosswords.
    
    A cell gets a number if it's a white cell and:
    - Starts a horizontal word (left is black/edge AND right is white)
    - OR starts a vertical word (above is black/edge AND below is white)
    
    Returns a dict mapping (row, col) -> number
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    word_numbers = {}
    current_number = 1
    
    for i in range(rows):
        for j in range(cols):
            if not is_white_cell(grid, i, j, rows, cols):
                continue
            
            # Check if this starts a horizontal word
            starts_horizontal = (
                not is_white_cell(grid, i, j - 1, rows, cols) and  # left is black/edge
                is_white_cell(grid, i, j + 1, rows, cols)  # right is white
            )
            
            # Check if this starts a vertical word
            starts_vertical = (
                not is_white_cell(grid, i - 1, j, rows, cols) and  # above is black/edge
                is_white_cell(grid, i + 1, j, rows, cols)  # below is white
            )
            
            if starts_horizontal or starts_vertical:
                word_numbers[(i, j)] = current_number
                current_number += 1
    
    return word_numbers


def generate_html_grid(grid, title: str, stats: dict, grid_size: int) -> str:
    """Generate HTML for a single crossword grid."""
    if grid is None:
        return f'''
        <div class="crossword-container">
            <h2>{title}</h2>
            <div class="failed">Failed to generate</div>
        </div>
        '''
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    if rows == 0 or cols == 0:
        return f'''
        <div class="crossword-container">
            <h2>{title}</h2>
            <div class="failed">Empty grid</div>
        </div>
        '''
    
    # Get word numbers for this grid
    word_numbers = get_word_numbers(grid)
    
    # Build grid HTML
    grid_html = f'<div class="grid" style="grid-template-columns: repeat({grid_size}, 1fr);">'
    for i in range(rows):
        for j in range(cols):
            cell = grid[i][j]
            if cell and cell not in ["#", " ", ".", None, 0, ""]:
                number = word_numbers.get((i, j), "")
                number_html = f'<span class="word-number">{number}</span>' if number else ''
                grid_html += f'<div class="cell white">{number_html}<span class="letter">{cell}</span></div>'
            else:
                grid_html += '<div class="cell black"></div>'
    grid_html += '</div>'
    
    # Stats HTML
    stats_html = f'''
    <div class="stats">
        <div class="stat"><span class="label">Words placed:</span> <span class="value">{stats.get('num_words', 0)}</span></div>
        <div class="stat"><span class="label">Avg word length ratio:</span> <span class="value">{stats.get('avg_word_length_ratio', 0):.2f}x</span></div>
        <div class="stat"><span class="label">Black squares:</span> <span class="value">{stats.get('black_square_percentage', 0):.1f}%</span></div>
        <div class="stat"><span class="label">Vertical/Horizontal:</span> <span class="value">{stats.get('vertical_horizontal_ratio', 0):.2f}</span></div>
    </div>
    '''
    
    return f'''
    <div class="crossword-container">
        <h2>{title}</h2>
        {grid_html}
        {stats_html}
    </div>
    '''


def generate_html_page(results: list, input_stats: dict, grid_size: int, calculate_stats_fn) -> str:
    """Generate full HTML page with all crosswords.
    
    Args:
        results: List of (grid, word_positions, title) tuples or None
        input_stats: Dict with 'total_words', 'avg_length', 'words' keys
        grid_size: Size of the crossword grid
        calculate_stats_fn: Function to calculate statistics for a crossword
    """
    
    grids_html = ""
    for result in results:
        if result:
            grid, word_positions, title = result
            stats = calculate_stats_fn(grid, word_positions, input_stats['words'])
            grids_html += generate_html_grid(grid, title, stats, grid_size)
        else:
            grids_html += generate_html_grid(None, "Algorithm Failed", {}, grid_size)
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crossword Algorithm Comparison</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 40px;
            font-size: 1.1rem;
        }}
        
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
        
        .input-stats span {{
            margin: 0 20px;
            color: #aaa;
        }}
        
        .input-stats strong {{
            color: #00d9ff;
        }}
        
        .grid-wrapper {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .crossword-container {{
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .crossword-container h2 {{
            font-size: 1.2rem;
            margin-bottom: 20px;
            color: #fff;
            font-weight: 600;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .grid {{
            display: grid;
            gap: 1px;
            background: #333;
            padding: 1px;
            border-radius: 4px;
            margin-bottom: 20px;
            aspect-ratio: 1;
        }}
        
        .cell {{
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: clamp(8px, 1.5vw, 14px);
            text-transform: uppercase;
            position: relative;
        }}
        
        .cell.white {{
            background: #fff;
            color: #1a1a2e;
        }}
        
        .cell.black {{
            background: #1a1a2e;
        }}
        
        .word-number {{
            position: absolute;
            top: 1px;
            left: 2px;
            font-size: clamp(5px, 0.7vw, 8px);
            font-weight: 400;
            color: #555;
            line-height: 1;
        }}
        
        .letter {{
            position: relative;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        
        .stat {{
            background: rgba(255,255,255,0.05);
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
        }}
        
        .stat .label {{
            color: #888;
        }}
        
        .stat .value {{
            color: #00ff88;
            font-weight: 600;
            float: right;
        }}
        
        .failed {{
            text-align: center;
            padding: 100px 20px;
            color: #ff6b6b;
            font-size: 1.2rem;
        }}
        
        @media (max-width: 900px) {{
            .grid-wrapper {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <h1>Crossword Algorithm Comparison</h1>
    <p class="subtitle">Comparing 4 different crossword generation algorithms</p>
    
    <div class="input-stats">
        <span>Total words: <strong>{input_stats['total_words']}</strong></span>
        <span>Avg length: <strong>{input_stats['avg_length']:.2f}</strong></span>
    </div>
    
    <div class="grid-wrapper">
        {grids_html}
    </div>
</body>
</html>
'''

