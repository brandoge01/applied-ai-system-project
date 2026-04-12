"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import os
from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "songs.csv"))

    # User taste profile — target values the recommender scores against
    user_prefs = {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.40,
        "valence": 0.58,
        "danceability": 0.60,
        "acousticness": 0.75,
        "instrumentalness": 0.85,
        "loudness_norm": 0.30,
        "likes_acoustic": True,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 50)
    print("  YOUR TOP 5 RECOMMENDATIONS")
    print("=" * 50)

    for rank, (song, score, reasons) in enumerate(recommendations, 1):
        print(f"\n  #{rank}  {song['title']} by {song['artist']}")
        print(f"       Genre: {song['genre']}  |  Mood: {song['mood']}")
        print(f"       Score: {score:.3f}")
        print("       Why:")
        for reason in reasons:
            print(f"         - {reason}")
        print("  " + "-" * 46)

    print()


if __name__ == "__main__":
    main()
