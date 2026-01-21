#!/usr/bin/env python3
"""
CLI wrapper for generate_themed_words.py that accepts custom person descriptions.

Usage:
    python generate_from_input.py "Your person description here"
    
Or interactively:
    python generate_from_input.py
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from generate_themed_words import generate_all_words, save_words_to_file


async def main():
    """Main entry point for CLI wrapper."""
    # Load environment variables from .env file
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        print("Error: OPENAI_API_KEY not configured.")
        print("Please copy .env.example to .env and add your API key.")
        print("Get your key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    # Get person description from command line or interactively
    if len(sys.argv) > 1:
        person_description = " ".join(sys.argv[1:])
    else:
        print("=" * 80)
        print("CROSSWORD WORD GENERATOR FROM PERSON DESCRIPTION")
        print("=" * 80)
        print("\nEnter a paragraph describing a person.")
        print("Include interests, profession, hobbies, background, etc.")
        print("(Press Ctrl+D or Ctrl+Z when done)\n")
        
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        person_description = "\n".join(lines).strip()
        
        if not person_description:
            print("\nError: No description provided.")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print(f"Person Description:\n{person_description}")
    print("=" * 80 + "\n")
    
    # Generate all words
    print("Generating themed crossword words...\n")
    word_lists = await generate_all_words(person_description, api_key)
    
    # Save to file
    print("\nSaving words to file...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = save_words_to_file(word_lists, output_dir)
    
    print(f"\n✓ Successfully saved words to: {filepath}")
    print(f"✓ Generated {sum(len(wl.words) for wl in word_lists)} total words across {len(word_lists)} themes")


if __name__ == "__main__":
    asyncio.run(main())

