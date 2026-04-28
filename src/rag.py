"""
RAG pipeline for VibeFinder 2.0.

Retrieve: recommend_songs() scores and ranks songs (from recommender.py)
Augment:  format retrieved songs + user profile into structured prompt context
Generate: Claude API produces natural-language recommendation explanations
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import anthropic

from src.recommender import recommend_songs

SYSTEM_PROMPT = """You are VibeFinder AI, a music recommendation expert. You help users discover songs from a curated catalog based on their listening preferences.

RULES:
1. Only recommend songs from the RETRIEVED SONGS section below. Never invent songs.
2. Reference specific audio features (energy, valence, danceability, etc.) when explaining why a song matches.
3. Be conversational but concise.
4. If the top matches have low scores (below 0.75), honestly note that the catalog may not perfectly match the user's taste and explain why.
5. Suggest which song to try first and why.
6. Keep your response under 200 words."""


PARSE_SYSTEM_PROMPT = """You extract music preferences from natural language. Given a user's description of what they want to listen to, return a JSON object with these fields:

- genre: one of [ambient, classical, country, electronic, folk, hip-hop, indie pop, jazz, latin, lofi, metal, pop, r&b, rock, synthwave]
- mood: one of [chill, energetic, focused, happy, intense, moody, relaxed, romantic, sad]
- energy: float 0.0-1.0 (how energetic the music should feel)
- valence: float 0.0-1.0 (how happy/positive the music should feel)
- danceability: float 0.0-1.0 (how danceable)
- acousticness: float 0.0-1.0 (acoustic vs electronic)
- instrumentalness: float 0.0-1.0 (instrumental vs vocal)
- loudness_norm: float 0.0-1.0 (quiet vs loud)

Respond with ONLY valid JSON, no other text."""


def parse_natural_input(user_text: str) -> Dict:
    """Use Claude to extract structured preferences from natural language.

    Example: "something chill for a rainy coding session" ->
    {"genre": "lofi", "mood": "chill", "energy": 0.35, ...}
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=PARSE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_text}],
    )
    text = response.content[0].text.strip()
    # Strip markdown code fences if the model wraps its JSON in them
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove opening ```json line
        text = text.rsplit("```", 1)[0]  # remove closing ```
        text = text.strip()
    return json.loads(text)


def format_user_profile(user_prefs: Dict) -> str:
    """Convert a user preferences dict into readable text for the prompt."""
    lines = [
        f"- Genre: {user_prefs['genre']}",
        f"- Mood: {user_prefs['mood']}",
        f"- Energy: {user_prefs['energy']}",
        f"- Valence: {user_prefs['valence']}",
        f"- Danceability: {user_prefs['danceability']}",
        f"- Acousticness: {user_prefs['acousticness']}",
        f"- Instrumentalness: {user_prefs['instrumentalness']}",
        f"- Loudness: {user_prefs['loudness_norm']}",
    ]
    return "\n".join(lines)


def format_song_context(recommendations: List[Tuple[Dict, float, List[str]]]) -> str:
    """Format the output of recommend_songs() into structured text for the prompt."""
    blocks = []
    for rank, (song, score, reasons) in enumerate(recommendations, 1):
        block = (
            f"#{rank} — {song['title']} by {song['artist']}\n"
            f"  Genre: {song['genre']} | Mood: {song['mood']}\n"
            f"  Energy: {song['energy']} | Valence: {song['valence']} | "
            f"Danceability: {song['danceability']} | Acousticness: {song['acousticness']}\n"
            f"  Match Score: {score:.3f}\n"
            f"  Scoring Reasons: {'; '.join(reasons)}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def build_rag_prompt(user_profile_text: str, song_context_text: str) -> Tuple[str, List[Dict]]:
    """Build the system prompt and messages list for the Anthropic API.

    Returns (system_prompt, messages).
    """
    user_message = (
        f"## User Profile\n{user_profile_text}\n\n"
        f"## Retrieved Songs (ranked by match score)\n{song_context_text}\n\n"
        "Based on this user's preferences and the retrieved songs above, provide a "
        "personalized recommendation summary. Explain why each song matches or doesn't "
        "perfectly match their taste, referencing specific features. Suggest which song "
        "to start with."
    )
    messages = [{"role": "user", "content": user_message}]
    return SYSTEM_PROMPT, messages


def generate_recommendation(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
) -> Tuple[str, List[Tuple[Dict, float, List[str]]]]:
    """Run the full RAG pipeline: retrieve, augment, generate.

    Returns (claude_response_text, recommendations).
    """
    # Retrieve
    recommendations = recommend_songs(user_prefs, songs, k=k)

    # Augment
    user_profile_text = format_user_profile(user_prefs)
    song_context_text = format_song_context(recommendations)
    system_prompt, messages = build_rag_prompt(user_profile_text, song_context_text)

    # Generate
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    ai_text = response.content[0].text

    return ai_text, recommendations
