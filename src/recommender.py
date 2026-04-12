import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    target_valence: float
    target_danceability: float
    target_acousticness: float
    target_instrumentalness: float
    target_loudness_norm: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Read a CSV file and return a list of song dictionaries with numeric fields converted."""
    numeric_fields = {
        "id": int,
        "tempo_bpm": int,
        "energy": float,
        "valence": float,
        "danceability": float,
        "acousticness": float,
        "instrumentalness": float,
        "loudness_norm": float,
    }

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field, convert in numeric_fields.items():
                row[field] = convert(row[field])
            songs.append(row)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score a single song against user preferences and return (score, reasons)."""
    weights = {
        "genre": 2.0,
        "mood": 1.5,
        "energy": 2.0,
        "valence": 1.0,
        "danceability": 1.0,
        "acousticness": 1.5,
        "instrumentalness": 1.0,
        "loudness_norm": 0.5,
    }

    scores = {}
    reasons = []

    # Categorical scoring — exact match = 1.0, mismatch = 0.0
    if song["genre"] == user_prefs["genre"]:
        scores["genre"] = 1.0
        reasons.append(f"genre match (+{weights['genre']:.1f})")
    else:
        scores["genre"] = 0.0
        reasons.append(f"genre mismatch (+0.0)")

    if song["mood"] == user_prefs["mood"]:
        scores["mood"] = 1.0
        reasons.append(f"mood match (+{weights['mood']:.1f})")
    else:
        scores["mood"] = 0.0
        reasons.append(f"mood mismatch (+0.0)")

    # Numeric proximity scoring — 1 - |song_value - user_preference|
    numeric_features = ["energy", "valence", "danceability",
                        "acousticness", "instrumentalness", "loudness_norm"]
    for feature in numeric_features:
        distance = abs(song[feature] - user_prefs[feature])
        feature_score = 1.0 - distance
        scores[feature] = feature_score
        weighted = feature_score * weights[feature]
        reasons.append(f"{feature}: {feature_score:.2f} (weighted +{weighted:.2f})")

    # Weighted average
    total = sum(scores[f] * weights[f] for f in weights)
    total_weight = sum(weights.values())
    final_score = total / total_weight

    return (final_score, reasons)

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score all songs, sort by score descending, and return the top k with reasons."""
    # Score every song and pair it with its result
    scored = [(song, *score_song(user_prefs, song)) for song in songs]

    # Sort by score descending
    scored.sort(key=lambda item: item[1], reverse=True)

    # Return top k as (song, score, reasons list)
    return [(song, score, reasons) for song, score, reasons in scored[:k]]
