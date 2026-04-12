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

    # --- Standard profiles ---

    profiles = {
        "Chill Lofi Listener": {
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.40,
            "valence": 0.58,
            "danceability": 0.60,
            "acousticness": 0.75,
            "instrumentalness": 0.85,
            "loudness_norm": 0.30,
        },
        "High-Energy Pop Fan": {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.90,
            "valence": 0.82,
            "danceability": 0.85,
            "acousticness": 0.10,
            "instrumentalness": 0.03,
            "loudness_norm": 0.88,
        },
        "Deep Intense Rock": {
            "genre": "rock",
            "mood": "intense",
            "energy": 0.92,
            "valence": 0.40,
            "danceability": 0.60,
            "acousticness": 0.08,
            "instrumentalness": 0.25,
            "loudness_norm": 0.90,
        },

        # --- Adversarial / edge case profiles ---

        "Sad but High-Energy": {
            # Conflict: high energy usually pairs with happy/intense, not sad
            "genre": "folk",
            "mood": "sad",
            "energy": 0.90,
            "valence": 0.30,
            "danceability": 0.80,
            "acousticness": 0.50,
            "instrumentalness": 0.10,
            "loudness_norm": 0.85,
        },
        "Genre Doesn't Exist": {
            # Edge case: no song in the catalog matches this genre
            "genre": "reggae",
            "mood": "chill",
            "energy": 0.50,
            "valence": 0.70,
            "danceability": 0.75,
            "acousticness": 0.60,
            "instrumentalness": 0.20,
            "loudness_norm": 0.45,
        },
        "All Extremes": {
            # Edge case: every numeric value maxed out — tests ceiling behavior
            "genre": "electronic",
            "mood": "energetic",
            "energy": 1.00,
            "valence": 1.00,
            "danceability": 1.00,
            "acousticness": 1.00,
            "instrumentalness": 1.00,
            "loudness_norm": 1.00,
        },
    }

    for profile_name, user_prefs in profiles.items():
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print("\n" + "=" * 50)
        print(f"  {profile_name.upper()}")
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
