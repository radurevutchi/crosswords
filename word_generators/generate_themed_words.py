#!/usr/bin/env python3
"""
Script to generate themed crossword words from a person description.

Takes a paragraph description, decomposes it into themes using structured LLM output,
then generates crossword words for each theme in parallel.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv


# Pydantic models for structured outputs
class Theme(BaseModel):
    """A thematic category extracted from the person description."""

    name: str = Field(description="Short name for the theme (2-4 words)")
    description: str = Field(description="Brief description of this theme")


class ThemeDecomposition(BaseModel):
    """Collection of themes extracted from a person description."""

    themes: List[Theme] = Field(
        description="List of thematic categories, typically 5-10 themes"
    )


class CrosswordWord(BaseModel):
    """A word suitable for crossword puzzles."""

    word: str = Field(description="The crossword word in uppercase")
    length: int = Field(description="Number of letters in the word")
    reason: str = Field(
        description="Brief explanation of why this word fits the theme and works well in crosswords"
    )


class CrosswordWordList(BaseModel):
    """Collection of crossword words for a theme."""

    theme_name: str = Field(description="The theme these words relate to")
    words: List[CrosswordWord] = Field(description="List of 100 crossword words")


async def decompose_into_themes(
    person_description: str, api_key: str
) -> ThemeDecomposition:
    """
    Use LLM with structured output to decompose person description into themes.

    Args:
        person_description: Paragraph describing a person
        api_key: OpenAI API key

    Returns:
        ThemeDecomposition containing multiple themes
    """
    client = openai.AsyncOpenAI(api_key=api_key)

    completion = await client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at analyzing person descriptions and extracting "
                    "thematic categories that would be useful for generating crossword puzzle words. "
                    "Extract 5-10 diverse themes that capture different aspects of the person's "
                    "interests, background, profession, hobbies, personality traits, etc."
                ),
            },
            {
                "role": "user",
                "content": f"Analyze this person description and extract thematic categories:\n\n{person_description}",
            },
        ],
        response_format=ThemeDecomposition,
    )

    return completion.choices[0].message.parsed


async def generate_words_for_theme(theme: Theme, api_key: str) -> CrosswordWordList:
    """
    Generate 100 crossword words for a specific theme.

    Args:
        theme: Theme to generate words for
        api_key: OpenAI API key

    Returns:
        CrosswordWordList with 100 words
    """
    client = openai.AsyncOpenAI(api_key=api_key)

    completion = await client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert crossword puzzle constructor. Generate exactly 100 words "
                    "related to the given theme. Words should:\n"
                    "- Range from 3 to 15 letters in length\n"
                    "- Have most words in the 3-8 letter range (easier to fit in puzzles)\n"
                    "- Be interesting, clear, and unambiguous\n"
                    "- Avoid obscure or overly specialized terms\n"
                    "- Be written in UPPERCASE\n"
                    "- Include a brief reason why each word is good for crosswords\n"
                    "Distribute word lengths with roughly: 40% (3-5 letters), 40% (6-8 letters), "
                    "15% (9-11 letters), 5% (12-15 letters)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate 100 crossword words for this theme:\n\n"
                    f"Theme: {theme.name}\n"
                    f"Description: {theme.description}"
                ),
            },
        ],
        response_format=CrosswordWordList,
    )

    return completion.choices[0].message.parsed


async def generate_all_words(
    person_description: str, api_key: str
) -> List[CrosswordWordList]:
    """
    Main async function to decompose themes and generate words for all themes in parallel.

    Args:
        person_description: Paragraph describing a person
        api_key: OpenAI API key

    Returns:
        List of CrosswordWordList objects, one per theme
    """
    print("Step 1: Decomposing person description into themes...")
    theme_decomposition = await decompose_into_themes(person_description, api_key)

    print(f"\nFound {len(theme_decomposition.themes)} themes:")
    for i, theme in enumerate(theme_decomposition.themes, 1):
        print(f"  {i}. {theme.name}: {theme.description}")

    print(f"\nStep 2: Generating 100 words for each theme (in parallel)...")

    # Generate words for all themes in parallel using asyncio.gather
    word_lists = await asyncio.gather(
        *[
            generate_words_for_theme(theme, api_key)
            for theme in theme_decomposition.themes
        ]
    )

    return word_lists


def save_words_to_file(
    word_lists: List[CrosswordWordList], output_dir: str = "."
) -> str:
    """
    Save all generated words to a text file named with the current date.
    Also creates a words-only file with just the words, one per line.

    Args:
        word_lists: List of CrosswordWordList objects
        output_dir: Directory to save the file in

    Returns:
        Path to the saved detailed file
    """
    # Create filename with current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"crossword_words_{current_date}.txt"
    filepath = os.path.join(output_dir, filename)

    # Also create words-only filename
    words_only_filename = f"words_only_{current_date}.txt"
    words_only_filepath = os.path.join(output_dir, words_only_filename)

    # Collect all unique words for words-only file
    all_words = set()

    # Write detailed file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"CROSSWORD WORDS GENERATED ON {current_date}\n")
        f.write("=" * 80 + "\n\n")

        total_words = 0
        for word_list in word_lists:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"THEME: {word_list.theme_name}\n")
            f.write(f"{'=' * 80}\n\n")

            # Sort words by length for better organization
            sorted_words = sorted(word_list.words, key=lambda w: (w.length, w.word))

            for word_obj in sorted_words:
                f.write(f"{word_obj.word} ({word_obj.length} letters)\n")
                f.write(f"  Reason: {word_obj.reason}\n\n")
                total_words += 1
                all_words.add(word_obj.word)

        f.write(f"\n{'=' * 80}\n")
        f.write(f"TOTAL: {total_words} words across {len(word_lists)} themes\n")
        f.write("=" * 80 + "\n")

    # Write words-only file
    with open(words_only_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(all_words)))

    print(f"✓ Also saved {len(all_words)} unique words to: {words_only_filepath}")

    return filepath


async def main():
    """Main entry point for the script."""
    # Load environment variables from .env file
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        raise ValueError(
            "OPENAI_API_KEY not configured. "
            "Please copy .env.example to .env and add your API key. "
            "Get your key from: https://platform.openai.com/api-keys"
        )

    # Example person description (can be replaced with user input)
    # person_description = """
    # Dr. Sarah Chen is a marine biologist who specializes in coral reef ecosystems
    # and climate change impacts on ocean life. She grew up in Hawaii, where she
    # developed a deep love for the ocean through surfing and scuba diving. In her
    # free time, she's an avid photographer, capturing underwater landscapes and
    # wildlife. She's also passionate about science communication, regularly writing
    # blog posts and giving TED talks about ocean conservation. Sarah plays the ukulele
    # and is learning to sail. She's a vegetarian who loves cooking fusion cuisine,
    # blending Hawaiian and Asian flavors. She's fluent in English, Mandarin, and
    # conversational Spanish.
    # """

    person_description = """
    Alex grew up in Moldova during the post-Soviet era, experiencing a unique blend 
    of Eastern European culture and the rapid changes of the 1990s. At age 12, they 
    moved with their family to Washington DC, where they navigated being a third-culture 
    kid while attending American schools. Alex attended Carnegie Mellon University in 
    Qatar (CMU-Q) in Doha, studying computer science while immersed in the cosmopolitan 
    Middle Eastern environment. They developed a passion for startups and technology 
    while participating in hackathons across the Gulf region. After graduating, Alex 
    moved to San Francisco to work in the tech industry. They're fluent in Romanian, 
    Russian, and English, with conversational Arabic. Alex loves exploring SF's diverse 
    food scene, particularly Eastern European and Middle Eastern restaurants that remind 
    them of home. They're an avid runner who trains for half-marathons along the 
    Embarcadero, and enjoys weekend trips to Tahoe for hiking. Alex is deeply interested 
    in immigration policy, cross-cultural communication, and building products for 
    global audiences. They maintain connections to all three regions they've called 
    home through online communities and annual visits.
    """

    print("=" * 80)
    print("CROSSWORD WORD GENERATOR FROM PERSON DESCRIPTION")
    print("=" * 80)
    print(f"\nPerson Description:\n{person_description.strip()}\n")

    # Generate all words
    word_lists = await generate_all_words(person_description, api_key)

    # Save to file
    print("\nStep 3: Saving words to file...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = save_words_to_file(word_lists, output_dir)

    print(f"\n✓ Successfully saved words to: {filepath}")
    print(
        f"✓ Generated {sum(len(wl.words) for wl in word_lists)} total words across {len(word_lists)} themes"
    )


if __name__ == "__main__":
    asyncio.run(main())
