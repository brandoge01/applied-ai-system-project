"""
Evaluation harness for VibeFinder 2.0.

Runs predefined user profiles through the full RAG pipeline and checks:
1. Grounding -Does Claude only mention songs from the retrieved set?
2. Feature references -Does Claude reference actual audio features?
3. Low-score honesty -Does Claude flag weak matches when scores are low?
4. Consistency -Do repeated runs produce stable retrieval rankings?

Usage:
    python -m src.evaluate
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.recommender import load_songs, recommend_songs
from src.rag import format_song_context, format_user_profile, generate_recommendation

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "songs.csv")

TEST_PROFILES = {
    "Chill Lofi Listener": {
        "genre": "lofi", "mood": "chill",
        "energy": 0.40, "valence": 0.58, "danceability": 0.60,
        "acousticness": 0.75, "instrumentalness": 0.85, "loudness_norm": 0.30,
    },
    "High-Energy Pop Fan": {
        "genre": "pop", "mood": "happy",
        "energy": 0.90, "valence": 0.82, "danceability": 0.85,
        "acousticness": 0.10, "instrumentalness": 0.03, "loudness_norm": 0.88,
    },
    "Sad but High-Energy (Adversarial)": {
        "genre": "folk", "mood": "sad",
        "energy": 0.90, "valence": 0.30, "danceability": 0.80,
        "acousticness": 0.50, "instrumentalness": 0.10, "loudness_norm": 0.85,
    },
    "Genre Doesn't Exist (Edge Case)": {
        "genre": "reggae", "mood": "chill",
        "energy": 0.50, "valence": 0.70, "danceability": 0.75,
        "acousticness": 0.60, "instrumentalness": 0.20, "loudness_norm": 0.45,
    },
}

FEATURE_KEYWORDS = ["energy", "valence", "danceability", "acousticness", "genre", "mood"]


def check_grounding(ai_text: str, recommendations: list, all_songs: list) -> tuple:
    """Check that Claude only mentions song titles from the retrieved set.

    Returns (passed: bool, details: str).
    """
    retrieved_titles = {song["title"].lower() for song, _, _ in recommendations}
    all_titles = {song["title"].lower() for song in all_songs}
    non_retrieved = all_titles - retrieved_titles

    hallucinated = []
    for title in non_retrieved:
        if title.lower() in ai_text.lower():
            hallucinated.append(title)

    if hallucinated:
        return False, f"Mentioned non-retrieved songs: {', '.join(hallucinated)}"
    return True, "Only referenced retrieved songs"


def check_feature_references(ai_text: str) -> tuple:
    """Check that Claude references at least 2 audio features.

    Returns (passed: bool, details: str).
    """
    found = [kw for kw in FEATURE_KEYWORDS if kw in ai_text.lower()]
    if len(found) >= 2:
        return True, f"Referenced features: {', '.join(found)}"
    return False, f"Only referenced {len(found)} feature(s): {', '.join(found)}"


def check_low_score_honesty(ai_text: str, recommendations: list) -> tuple:
    """If top score is below 0.75, check that Claude acknowledges weak matches.

    Returns (passed: bool, details: str).
    """
    top_score = recommendations[0][1]
    if top_score >= 0.75:
        return True, f"Top score {top_score:.3f} is above threshold, check not applicable"

    honesty_signals = ["low", "weak", "tough", "doesn't perfectly", "not ideal",
                       "no ", "none of these", "limited", "struggle", "gap", "miss"]
    text_lower = ai_text.lower()
    found = any(signal in text_lower for signal in honesty_signals)
    if found:
        return True, f"Top score {top_score:.3f} -Claude acknowledged weak matches"
    return False, f"Top score {top_score:.3f} -Claude did not flag weak matches"


def check_retrieval_consistency(user_prefs: dict, songs: list) -> tuple:
    """Run retrieval twice and verify rankings are identical (deterministic).

    Returns (passed: bool, details: str).
    """
    run1 = [(s["title"], score) for s, score, _ in recommend_songs(user_prefs, songs, k=5)]
    run2 = [(s["title"], score) for s, score, _ in recommend_songs(user_prefs, songs, k=5)]
    if run1 == run2:
        return True, "Rankings are deterministic across runs"
    return False, "Rankings differ between runs"


def run_evaluation():
    songs = load_songs(DATA_PATH)

    print("=" * 60)
    print("  VibeFinder 2.0 - Evaluation Harness")
    print("=" * 60)

    total_checks = 0
    passed_checks = 0
    results_by_profile = {}

    for name, prefs in TEST_PROFILES.items():
        print(f"\n{'-' * 60}")
        print(f"  Profile: {name}")
        print(f"{'-' * 60}")

        profile_results = []

        # Check 1: Retrieval consistency (no API needed)
        total_checks += 1
        passed, detail = check_retrieval_consistency(prefs, songs)
        passed_checks += passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Retrieval consistency -{detail}")
        profile_results.append(("Retrieval consistency", passed))

        # Checks 2-4 need the Claude API
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("  [SKIP] Grounding, features, honesty -no ANTHROPIC_API_KEY set")
            results_by_profile[name] = profile_results
            continue

        try:
            ai_text, recs = generate_recommendation(prefs, songs, k=5)
        except Exception as e:
            print(f"  [ERROR] API call failed: {e}")
            results_by_profile[name] = profile_results
            continue

        # Check 2: Grounding
        total_checks += 1
        passed, detail = check_grounding(ai_text, recs, songs)
        passed_checks += passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Grounding -{detail}")
        profile_results.append(("Grounding", passed))

        # Check 3: Feature references
        total_checks += 1
        passed, detail = check_feature_references(ai_text)
        passed_checks += passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Feature references -{detail}")
        profile_results.append(("Feature references", passed))

        # Check 4: Low-score honesty
        total_checks += 1
        passed, detail = check_low_score_honesty(ai_text, recs)
        passed_checks += passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Low-score honesty -{detail}")
        profile_results.append(("Low-score honesty", passed))

        results_by_profile[name] = profile_results

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {passed_checks}/{total_checks} checks passed")
    print(f"{'=' * 60}")
    for name, results in results_by_profile.items():
        statuses = " ".join("PASS" if p else "FAIL" for _, p in results)
        print(f"  {name}: {statuses}")

    return passed_checks, total_checks


if __name__ == "__main__":
    run_evaluation()
