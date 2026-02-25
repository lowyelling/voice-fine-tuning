# Voice Evaluation Rubric

Rate each dimension 1-5 for each model output. Compare against the actual essay (Canary A/C) or your gut sense of voice (Canary B).

## Sentence-Level Voice (all models)

### 1. Rhythm and Length Variation
Does the output alternate between short punchy sentences and longer complex ones? Lily's pattern: assertion, then expansion, then compression. Short sentences do the most work.

1 = monotone length | 5 = rhythm matches

### 2. Punctuation as Voice
Does it use em-dashes, parentheticals, colons the way Lily does? Em-dashes for interruption/clarification. Parentheticals for asides that undercut the main claim. Colons to set up a payoff.

1 = generic punctuation | 5 = punctuation matches

### 3. Register Mixing
Does it move between analytical/academic and visceral/blunt within the same paragraph? "The convergence of English and code" next to "rip things up, start over."

1 = single register throughout | 5 = register mixing matches

### 4. Assertion vs Hedging
Does the output assert or hedge? Lily asserts. She qualifies with precision ("I suspect," "perhaps") but does not hedge with filler ("it could be argued that," "in some ways," "I think it's fair to say").

1 = over-hedged / filler-laden | 5 = assertion pattern matches

### 5. Compression
Does every sentence do work? Lily's Notes are "pure compressed voice." No filler sentences that could be cut without loss. No throat-clearing. No restating what was just said.

1 = padded / repetitive | 5 = compressed

## Structure-Level Voice (Llama only, Canary C)

### 6. Opening Moves
Does the opening create tension or a question, not state a thesis? Lily often opens with a concrete scene or a paradox. Not "In this essay I will discuss..."

1 = generic thesis opening | 5 = opening pattern matches

### 7. Structural Architecture
Do later sections pay off earlier ones? Callbacks, recontextualizations, endings that reshape the beginning. The essay builds rather than listing.

1 = linear / additive | 5 = architectural

### 8. Omission and Negative Space
Does the output know what to leave out? Over-explanation is a tell for AI writing. Lily's essays are "shaped by what they leave out." Trusts the reader.

1 = over-explains everything | 5 = trusts the reader

## Overall

### 9. Authorship
Not "is this good writing" but "does this sound like it came from the same person?" Could this plausibly be a Lily draft?

1 = completely generic | 5 = hard to distinguish

## Scoring Sheet

Copy this table for each canary prompt, each model:

| Dimension | Base GPT-2 | FT GPT-2 | Base Llama | FT Llama |
|-----------|-----------|----------|-----------|---------|
| 1. Rhythm | | | | |
| 2. Punctuation | | | | |
| 3. Register | | | | |
| 4. Assertion | | | | |
| 5. Compression | | | | |
| 6. Opening | — | — | | |
| 7. Architecture | — | — | | |
| 8. Omission | — | — | | |
| 9. Authorship | | | | |
| **Total** | | | | |
