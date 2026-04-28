"""
Streamlit web UI for VibeFinder 2.0.

Lets users set taste preferences via sliders/dropdowns, then shows:
1. Retrieved songs (scored by the recommender engine)
2. AI analysis (Claude's RAG-generated explanation)
3. How It Works (raw prompt context for transparency)
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import load_songs
from src.rag import (
    format_song_context,
    format_user_profile,
    generate_recommendation,
    parse_natural_input,
    recommend_songs,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "songs.csv")


@st.cache_data
def get_songs():
    return load_songs(DATA_PATH)


def main():
    st.set_page_config(page_title="VibeFinder 2.0", layout="wide")
    st.title("VibeFinder 2.0")
    st.caption("RAG-enhanced music recommendations powered by Claude")

    songs = get_songs()
    genres = sorted(set(s["genre"] for s in songs))
    moods = sorted(set(s["mood"] for s in songs))

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # --- Sidebar: input mode + preferences ---
    with st.sidebar:
        st.header("Your Taste Profile")
        mode = st.radio("Input mode", ["Describe your vibe", "Use sliders"], index=0)

        if mode == "Describe your vibe":
            user_text = st.text_area(
                "What do you want to listen to?",
                placeholder="e.g. something chill for a rainy coding session",
            )
            k = st.slider("Number of results", 1, 10, 5)
            run = st.button("Get Recommendations")
        else:
            user_text = None
            genre = st.selectbox("Genre", genres)
            mood = st.selectbox("Mood", moods)
            energy = st.slider("Energy", 0.0, 1.0, 0.50, 0.05)
            valence = st.slider("Valence (happiness)", 0.0, 1.0, 0.60, 0.05)
            danceability = st.slider("Danceability", 0.0, 1.0, 0.65, 0.05)
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.50, 0.05)
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.30, 0.05)
            loudness_norm = st.slider("Loudness", 0.0, 1.0, 0.50, 0.05)
            k = st.slider("Number of results", 1, 10, 5)
            run = st.button("Get Recommendations")

    if not run:
        st.info("Set your preferences in the sidebar and click **Get Recommendations**.")
        return

    # --- Build user_prefs from either input mode ---
    if mode == "Describe your vibe":
        if not user_text or not user_text.strip():
            st.warning("Please describe what you want to listen to.")
            return
        if not api_key:
            st.error("Natural language input requires an API key. Switch to slider mode or set ANTHROPIC_API_KEY.")
            return
        with st.spinner("Understanding your vibe..."):
            try:
                user_prefs = parse_natural_input(user_text)
                st.success(f"Interpreted as: **{user_prefs['genre']}** / **{user_prefs['mood']}** "
                           f"(energy {user_prefs['energy']}, valence {user_prefs['valence']})")
            except Exception as e:
                st.error(f"Could not parse your input: {e}")
                return
    else:
        user_prefs = {
            "genre": genre,
            "mood": mood,
            "energy": energy,
            "valence": valence,
            "danceability": danceability,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "loudness_norm": loudness_norm,
        }

    # --- Retrieval step (always runs) ---
    recommendations = recommend_songs(user_prefs, songs, k=k)

    st.subheader("Retrieved Songs")
    rows = []
    for rank, (song, score, _reasons) in enumerate(recommendations, 1):
        rows.append({
            "Rank": rank,
            "Title": song["title"],
            "Artist": song["artist"],
            "Genre": song["genre"],
            "Mood": song["mood"],
            "Score": f"{score:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Generation step (needs API key) ---
    st.subheader("AI Analysis")

    if not api_key:
        st.warning(
            "Set the `ANTHROPIC_API_KEY` environment variable to enable AI analysis. "
            "Showing scored results only."
        )
    else:
        with st.spinner("Generating AI analysis..."):
            try:
                ai_text, _ = generate_recommendation(user_prefs, songs, k=k)
                st.markdown(ai_text)
            except Exception as e:
                st.error(f"Claude API error: {e}")

    # --- Transparency expander ---
    with st.expander("How It Works"):
        st.markdown("**User profile sent to Claude:**")
        st.code(format_user_profile(user_prefs))
        st.markdown("**Retrieved song context sent to Claude:**")
        st.code(format_song_context(recommendations))


if __name__ == "__main__":
    main()
