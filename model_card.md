# Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 2.0** (RAG-Enhanced)

---

## 2. Intended Use

VibeFinder picks 5 songs from a small catalog based on your genre, mood, and vibe preferences. It assumes you can describe your taste as numbers, which most people can't actually do.

---

## 3. How the Model Works

VibeFinder 2.0 uses a three-stage RAG (Retrieval-Augmented Generation) pipeline:

1. **Retrieve** — The scoring engine matches song attributes to a user's taste profile. Each song gets scored from 0 to 1. Genre and mood are yes-or-no checks (match = 1.0, mismatch = 0.0). Six numeric features get proximity scored: `score = 1 - |song_value - user_preference|`. Each feature is weighted (genre and energy at 2.0, mood and acousticness at 1.5, the rest at 1.0 or below). The top-k songs are returned with scores and reasons.

2. **Augment** — The retrieved songs and user profile are formatted into structured prompt context. A grounding constraint tells the AI to only reference songs from the retrieved set.

3. **Generate** — The Anthropic Claude API reads the retrieved context and produces a conversational recommendation narrative that explains why each song matches, references specific audio features, and suggests which song to try first.

---

## 4. Data

20 songs in `data/songs.csv`. Started with 10, added 10 more to fill genre gaps. Covers 15 genres (lofi, pop, rock, electronic, folk, etc.) and 9 moods (chill, happy, intense, sad, etc.). We added two columns - instrumentalness and loudness_norm. The catalog is small and uneven. Lofi has 3 songs while hip-hop, classical, and country get 1 each.

---

## 5. Strengths

Works well when your taste fits one genre cleanly. The lofi listener got Midnight Coding and Library Rain at the top. The rock fan got Storm Runner at #1. Easy cases where genre, mood, and energy all agree. It's also transparent as every recommendation shows exactly which features matched and which didn't. You can trace why a song ranked where it did. Handles missing genres without crashing too by falling back on numeric similarity with lower-confidence results.

---

## 6. Limitations and Bias

Genre dominance is the biggest issue. With a weight of 2.0, the system ranks a mediocre genre match above a song that nails the user's energy and mood. Our "Sad but High-Energy" test proved this since a quiet folk ballad (energy 0.31) beat a high-energy rock track (0.91) just because the genre label matched. Content-based filtering also means a built-in filter bubble. It only recommends songs similar to what you already like, so no cross-genre discoveries. The catalog is biased too. Hip-hop and classical fans get 1 song to pick from while lofi fans get 3. The proximity formula also treats all distances equally - a 0.1 gap in energy probably matters more than a 0.1 gap in valence, but the system doesn't know that.

---

## 7. Evaluation

Tested 6 profiles: 3 standard (Lofi Listener, Pop Fan, Rock Fan) and 3 adversarial (Sad but High-Energy, Reggae, All Extremes). Checked if the top 5 felt like songs a real person would enjoy. The ranking worked perfectly for normal profiles. The surprise was the Sad but High-Energy profile as a quiet folk ballad ranked #1 over high-energy tracks because labels outweighed sound. We doubled energy weight and halved genre weight, which flipped the ranking. The problem was the weights, not the formula. The reggae test showed a major silent weak point. No genre bonus meant scores topped out around 0.68 instead of the usual 0.95+.

---

## 8. Future Work

Collaborative filtering - Track what multiple users like so we can say "people like you also enjoyed this" and break the filter bubble.
Adaptive weights - Let the system learn which features matter most per user instead of using the same fixed numbers for everyone.
Diversity controls - Force at least one unexpected song into the top 5 so it doesn't feel like the same playlist every time.
Multi-turn conversation - Let users refine recommendations by chatting back and forth with Claude instead of re-adjusting sliders.
Evaluation harness - Compare Claude's explanations against human judgments to measure whether the AI analysis actually helps users find songs they like.

---

## 9. Personal Reflection

My biggest learning moment was the weighting experiment. I changed one genre from 2.0 to 1.0 and the entire ranking flipped for edge-case users. That's when it hit me that the formula doesn't really decide anything. The weights do. The formula is just math, but the weights are opinions baked into code. Someone chose that genre matters twice as much as valence, and that one decision shapes every recommendation the system makes.

AI tools helped a lot with the grunt work by scaffolding functions, expanding the dataset, formatting output. I had to double-check the scoring logic myself. When I first looked at the results for the "Sad but High-Energy" profile, the AI-generated code ran fine but the recommendations felt wrong. A quiet folk ballad beating a high-energy rock track wasn't a code bug, it was a design flaw. The AI or tool couldn't flag that for me. I had to look at the output, think about whether a real person would actually want these songs, and trace the problem back to the weights.

What surprised me most is how little math it takes to make something that feels like a real recommender. It's just subtraction and averaging, but when you see "Storm Runner by Voltline" show up for a rock fan, it feels like the system actually understands their taste. I feel like that gap between what the system does and what it feels like it does is exactly why people trust algorithms more than they should.

If I extended this, I'd add collaborative filtering so the system could say "people like you also liked this" instead of only matching on features. I'd also want adaptive weights that learn for each user so someone who cares about energy more than genre shouldn't get the same fixed formula as everyone else. In addition, I'd add a diversity rule so the top 5 isn't just five songs that sound the same.
