# Text Mining / NLP Task

## Goal
Collect a text corpus (≥500 sentences) and run a set of NLP tasks on it, comparing traditional (NLTK) vs. model-based (Hugging Face) approaches.

## Data
Collect ≥500 sentences from one of:
- Manually scraped articles (BBC, CNN, NYT, etc.)
- A crawler (extend the one from HW1) targeting a specific site
- An existing open dataset (e.g. speech transcripts)

## Tasks
1. **Sentiment analysis** (per sentence) — NLTK vs. HF model, compare.
2. **Named Entity Recognition** — NLTK vs. HF model, compare.
3. **Entity classification** — NLTK vs. HF model, compare.
4. **Zero-shot classification** — pick categories relevant to your data (e.g. politics, sports, technology, entertainment), classify each sentence with an HF zero-shot model.
5. **Text summarization** — summarize the full text with an HF summarization model.

## Deliverable
Code/notebook implementing the above, plus a short comparison of results across approaches for each task.
