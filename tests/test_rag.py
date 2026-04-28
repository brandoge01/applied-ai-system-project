"""Tests for the RAG pipeline (src/rag.py)."""

from unittest.mock import MagicMock, patch

from src.recommender import load_songs, recommend_songs
from src.rag import (
    build_rag_prompt,
    format_song_context,
    format_user_profile,
    generate_recommendation,
)

import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "songs.csv")

SAMPLE_PREFS = {
    "genre": "lofi",
    "mood": "chill",
    "energy": 0.40,
    "valence": 0.58,
    "danceability": 0.60,
    "acousticness": 0.75,
    "instrumentalness": 0.85,
    "loudness_norm": 0.30,
}


def test_format_user_profile_includes_preferences():
    text = format_user_profile(SAMPLE_PREFS)
    assert "lofi" in text
    assert "chill" in text
    assert "0.4" in text


def test_format_song_context_includes_all_songs():
    songs = load_songs(DATA_PATH)
    recs = recommend_songs(SAMPLE_PREFS, songs, k=3)
    text = format_song_context(recs)
    for song, _score, _reasons in recs:
        assert song["title"] in text
        assert song["artist"] in text


def test_build_rag_prompt_structure():
    songs = load_songs(DATA_PATH)
    recs = recommend_songs(SAMPLE_PREFS, songs, k=3)
    profile_text = format_user_profile(SAMPLE_PREFS)
    context_text = format_song_context(recs)
    system, messages = build_rag_prompt(profile_text, context_text)

    assert isinstance(system, str)
    assert "RETRIEVED SONGS" in system
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "lofi" in messages[0]["content"]
    for song, _score, _reasons in recs:
        assert song["title"] in messages[0]["content"]


@patch("src.rag.anthropic.Anthropic")
def test_retrieval_feeds_into_prompt(mock_anthropic_cls):
    """Verify that retrieved songs appear in the prompt sent to Claude."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Great picks!")]
    mock_client.messages.create.return_value = mock_response

    songs = load_songs(DATA_PATH)
    ai_text, recs = generate_recommendation(SAMPLE_PREFS, songs, k=3)

    assert ai_text == "Great picks!"
    assert len(recs) == 3

    call_kwargs = mock_client.messages.create.call_args[1]
    prompt_content = call_kwargs["messages"][0]["content"]
    for song, _score, _reasons in recs:
        assert song["title"] in prompt_content, (
            f"Retrieved song '{song['title']}' not found in prompt sent to Claude"
        )
