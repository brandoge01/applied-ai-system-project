# VibeFinder 2.0: RAG-Enhanced Music Recommender

Github: https://github.com/brandoge01/applied-ai-system-project
Demo: https://www.loom.com/share/ee6dc33c0ee74732b5e2b74d7ce6d8bb

## Original Project

VibeFinder 1.0 (Modules 1-3) was a content-based music recommender that scored 20 songs against a user's taste profile using weighted proximity matching on genre, mood, energy, and other audio features. It returned the top 5 matches with transparent scoring breakdowns. We tested it with 6 user profiles, ran weight experiments to expose scoring flaws, and documented the biases we found in the model card.

## Summary

VibeFinder 2.0 adds Retrieval-Augmented Generation (RAG) using the Anthropic Claude API and a Streamlit web UI. The original scoring engine retrieves the best-matching songs, then Claude generates natural-language explanations grounded in the retrieved data, so there are no hallucinated songs or invented features. Users interact through sliders, dropdowns, or plain English descriptions in the browser. This matters because recommendation systems shape what people discover, and making the reasoning visible keeps the user in control.

## Architecture Overview

```
User (Streamlit UI)
  │
  ├─ "Describe your vibe" ──► NLP Parser (Claude API)
  │                              │  parses natural language into
  │                              │  structured preferences (genre,
  │                              │  mood, energy, etc.)
  │                              ▼
  ├─ Sliders / dropdowns ────► User Preferences
  │                              │
  ▼                              ▼
Retriever ── recommend_songs() ◄── songs.csv
  │  top-k songs + scores + reasons
  ▼
Augmenter ── format context for Claude prompt
  │  structured song data + user profile
  ▼
Generator ── Claude API
  │  natural-language recommendation narrative
  ▼
Display ── retrieved songs table + AI analysis
  │
Human Review ── "How It Works" expander shows raw prompt
Automated Tests ── verify grounding + prompt structure
```

| Stage | What Happens | Code |
|-------|-------------|------|
| **Parse (vibe mode only)** | Claude converts natural language like "something chill for a rainy coding session" into structured preferences (genre, mood, energy, etc.) | `src/rag.py` `parse_natural_input()` |
| **Retrieve** | Scores all 20 songs via weighted proximity matching, returns top-k with reasons | `src/recommender.py` |
| **Augment** | Formats retrieved songs + user profile into structured prompt context with a grounding constraint | `src/rag.py` |
| **Generate** | Claude produces conversational explanations referencing specific audio features | `src/rag.py` via Anthropic API |

The existing scorer serves as the retriever directly. With only 20 songs, there's no need for a vector database.

## Setup Instructions

Prerequisites: Python 3.9+ and an [Anthropic API key](https://console.anthropic.com/)

```bash
# 1. Clone and enter the repo
git clone https://github.com/your-username/applied-ai-system-final.git
cd applied-ai-system-final

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
export ANTHROPIC_API_KEY="your-key-here"       # Mac/Linux
set ANTHROPIC_API_KEY=your-key-here             # Windows CMD
$env:ANTHROPIC_API_KEY="your-key-here"          # PowerShell
```

Run the Streamlit app:
```bash
streamlit run src/app.py
```

Run the original CLI (scoring only, no AI):
```bash
python -m src.main
```

Run tests:
```bash
pytest
```

## Sample Interactions

### Example 1: Chill Lofi Listener

Input: Genre = lofi, Mood = chill, Energy = 0.40, Acousticness = 0.75

| # | Song | Artist | Score |
|---|------|--------|-------|
| 1 | Library Rain | Paper Lanterns | 0.895 |
| 2 | Midnight Coding | LoRoom | 0.871 |
| 3 | Focus Flow | LoRoom | 0.860 |

AI Analysis:
> Your top picks are a lofi dream lineup. Library Rain is your best match with low energy (0.35) and high acousticness (0.86), almost exactly what you're after. Once you leave the lofi bubble, Spacewalk Thoughts (ambient) nails the acoustic feel but drops to 0.68 because the genre label doesn't match. That's the filter bubble in action. Start with Library Rain.

### Example 2: Sad but High-Energy (Adversarial)

Input: Genre = folk, Mood = sad, Energy = 0.90, Loudness = 0.85

| # | Song | Artist | Score |
|---|------|--------|-------|
| 1 | Autumn Letters | Maren Cole | 0.628 |
| 2 | Iron Tide | Shredlock | 0.606 |
| 3 | Gym Hero | Max Pulse | 0.602 |

AI Analysis:
> Tough profile. You want sad and acoustic but also high-energy and loud. Your top match is only 0.628, well below the usual 0.90+. Autumn Letters takes #1 because it's the only folk song, but its energy is 0.31 when you want 0.90. The genre weight (2.0) is carrying it past Iron Tide and Storm Runner, which actually sound like what you described. None of these are a great fit, and the system is exposing a real limitation. Start with Iron Tide for energy, Autumn Letters for the folk vibe.

### Example 3: Genre Doesn't Exist (Edge Case)

Input: Genre = reggae, Mood = chill, Energy = 0.50, Danceability = 0.75

| # | Song | Artist | Score |
|---|------|--------|-------|
| 1 | Velvet Groove | SoulCraft | 0.684 |
| 2 | Cumbia del Sol | Luna Roja | 0.682 |
| 3 | Dust Road Radio | Hank Wilder | 0.673 |

AI Analysis:
> No reggae in the catalog, so every song takes a zero on genre matching, dragging scores to the 0.65-0.68 range. The system falls back on numeric features. Velvet Groove (R&B) and Cumbia del Sol (Latin) have the mid-energy, danceable feel a reggae listener might enjoy, but the system can't account for offbeat guitar and bass-heavy production that define the genre. Start with Cumbia del Sol for the closest rhythmic vibe.

## Design Decisions

**Why RAG?** The recommender already retrieves and ranks songs, so adding Claude as a generation layer was the most natural integration. Fine-tuning would require training data we don't have.

**Why no vector database?** With 20 songs, brute-force scoring is instant. The existing scorer already produces ranked results with explanations.

**Why show retrieval separately from AI output?** Transparency. Users can compare scores to Claude's narrative and catch any misrepresentation.

**Why Streamlit?** It was already in the dependencies and gives us an interactive UI with minimal code.

**Natural language input.** Users can type "something chill for a rainy coding session" instead of fiddling with sliders. A separate Claude call parses that into structured preferences (genre, mood, energy, etc.), which keeps the retrieval layer deterministic while the input feels conversational. If there's no API key, the app falls back to slider-only mode.

**Trade-offs.** The small catalog limits recommendation quality. API calls add roughly 2-3 seconds of latency. The grounding constraint prevents Claude from suggesting real-world songs outside the catalog, which keeps it honest but limits usefulness compared to a production system with a large library.

## Testing Summary

### Automated Tests (pytest)

6 unit tests cover the scoring engine and RAG pipeline:

```bash
pytest
# 6 passed: format functions, prompt structure, grounding verification (mocked API)
```

### Evaluation Harness

`src/evaluate.py` runs 4 predefined profiles (including adversarial cases) through the full RAG pipeline and checks:

| Check | What It Verifies |
|-------|-----------------|
| Retrieval consistency | Same input always produces the same ranking (deterministic) |
| Grounding | Claude only mentions songs from the retrieved set, never hallucinated titles |
| Feature references | Claude references at least 2 audio features (energy, valence, etc.) |
| Low-score honesty | When top scores fall below 0.75, Claude flags weak matches instead of overselling |

```bash
python -m src.evaluate
```

Results: 16/16 checks passed across all 4 profiles. The reggae edge case (no genre match in catalog) triggered the low-score honesty check, and Claude correctly noted that scores were weak because no genre matched. The adversarial "sad but high-energy" profile passed grounding despite conflicting preferences. Claude stayed anchored to retrieved songs and acknowledged the trade-offs.

### Guardrails

The system prompt tells Claude "Only recommend songs from the RETRIEVED SONGS section. Never invent songs," which prevents hallucinated titles. Without an API key, the app still shows scored results since the retrieval layer works independently. API failures are caught and displayed in the UI instead of crashing.

### What Needed Fixing

Genre dominance (weight 2.0) still causes quiet folk ballads to outrank high-energy tracks in edge cases. Claude acknowledges it but can't fix the underlying ranking. Early prompt versions also made Claude over-explain. Adding "Keep your response under 200 words" and "suggest which song to start with" focused the output.

## Reflection

Building VibeFinder 2.0 taught me that AI features amplify the engineering underneath them, not replace it. The hardest part wasn't the API integration. It was making sure the retrieval step returned songs worth explaining. If the scoring logic is flawed, Claude just writes a more convincing description of a bad recommendation.

### Limitations and Biases

The 20-song catalog skews toward Western genres. There's no K-pop, Afrobeats, or Bollywood, so users with those tastes get poor matches and the system can't tell them why beyond "no genre match." The fixed feature weights (genre at 2.0x) encode my assumptions about what matters most in music taste, and those assumptions quietly shape every recommendation before the AI even runs. A user who cares more about tempo than genre will get worse results, and the UI doesn't explain that trade-off. Content-based filtering also creates a filter bubble by design. It only recommends what already matches your taste profile and will never surface a surprising cross-genre discovery the way collaborative filtering might.

### Misuse Prevention

The main risk is trust inflation. A polished AI narrative can make mediocre recommendations sound authoritative. Someone could wrap this pattern around a larger catalog and present AI-generated "personalized" picks as expert curation to push specific artists or sell ads. I addressed this by showing raw scores alongside AI text and exposing the full prompt in a "How It Works" expander, so users can see exactly what the AI was told and judge for themselves. The grounding constraint prevents Claude from inventing songs that could link to misleading content. For a production system, I'd add input validation to guard against prompt injection through the natural language input field.

### Testing Surprises

The biggest surprise was how confidently Claude would recommend non-existent songs before I added the grounding rule. It would seamlessly invent titles that sounded like they belonged in the catalog, complete with plausible artist names and feature descriptions. The evaluation harness caught this reliably, but it was unsettling how natural the hallucinations looked. I also didn't expect the adversarial "sad but high-energy" profile to be as revealing as it was. The retrieval scores (0.60-0.63) exposed that the scoring engine has no good answer for conflicting preferences. It just averages out the mismatch rather than flagging the contradiction. I was glad the system surfaced this honestly instead of hiding it behind a confident narrative.

### AI Collaboration

I used Claude throughout development to scaffold the RAG pipeline, draft prompts, and build the Streamlit UI. One helpful suggestion was structuring the system prompt with an explicit grounding constraint ("only recommend from the retrieved songs section"). This immediately prevented hallucinated titles and became the project's core guardrail. I wouldn't have thought to be that directive on the first try. One flawed suggestion: Claude's initial system prompt didn't include a word limit, producing 400+ word responses that buried the actual recommendation under unnecessary context. The output read well in isolation, so the AI didn't flag it as a problem. I had to recognize myself that conciseness mattered more than thoroughness for a recommendation tool, then manually add the "under 200 words" constraint after reviewing the output.

The weight experiment from v1.0 carried forward into this reflection. Claude can describe the genre dominance problem eloquently, but it can't fix it. The ranking is set before Claude sees the data. The decisions that matter most in an AI system are often made before the model runs.

## Portfolio

GitHub: [github.com/brandoge01/applied-ai-system-project](https://github.com/brandoge01/applied-ai-system-project)

This project reflects how I think about AI engineering. It's not just about wiring up an API, but understanding what happens before and after the model runs. I built a retrieval layer with known biases, wrote an evaluation harness to catch hallucinations, and designed the UI to show its work instead of hiding behind a polished answer. What I care about is building AI systems that are transparent about their limits, testable under adversarial conditions, and honest with the people who use them. That's the kind of engineer I want to be.

## Project Structure

```
applied-ai-system-final/
├── data/songs.csv              # 20-song catalog with audio features
├── src/
│   ├── recommender.py          # Scoring engine (retriever)
│   ├── rag.py                  # RAG pipeline (augment + generate)
│   ├── evaluate.py             # Evaluation harness (grounding, honesty checks)
│   ├── main.py                 # Original CLI runner
│   └── app.py                  # Streamlit web UI
├── tests/
│   ├── test_recommender.py     # Scoring engine tests
│   └── test_rag.py             # RAG pipeline tests
├── model_card.md               # Model card with bias documentation
├── requirements.txt            # pandas, pytest, streamlit, anthropic
└── README.md
```

