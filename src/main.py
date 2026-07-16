import os
import re
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

from transformers import pipeline
import mlflow

# ----------------------------
# Path setup
# ----------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "../results/"


# ----------------------------
# NLTK downloads
# ----------------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker")
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")
nltk.download("vader_lexicon")
nltk.download("inaugural")


# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def normalize_hf_sentiment(label: str) -> str:
    label = label.lower()
    if "pos" in label:
        return "positive"
    if "neg" in label:
        return "negative"
    if "neu" in label:
        return "neutral"
    return label


def chunk_text(text: str, max_words: int = 350):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


# ----------------------------
# Dataset
# ----------------------------
def load_speech_dataset(min_sentence_len: int = 20, max_sentences: int = 600) -> tuple[pd.DataFrame, str]:
    from nltk.corpus import inaugural

    texts = []
    for fileid in inaugural.fileids():
        texts.append(inaugural.raw(fileid))

    full_text = "\n".join(texts)

    with open(DATA_DIR / "full_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    sentences = sent_tokenize(full_text)
    sentences = [clean_sentence(s) for s in sentences]
    sentences = [s for s in sentences if len(s) >= min_sentence_len]
    sentences = sentences[:max_sentences]  # Limit to required minimum

    df = pd.DataFrame({
        "sentence_id": range(1, len(sentences) + 1),
        "sentence": sentences
    })

    df.to_csv(DATA_DIR / "sentences.csv", index=False, encoding="utf-8")
    return df, full_text


# ----------------------------
# Sentiment analysis
# ----------------------------
def run_nltk_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    rows = []

    for _, row in df.iterrows():
        text = row["sentence"]
        scores = sia.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        rows.append({
            "sentence_id": row["sentence_id"],
            "sentence": text,
            "nltk_label": label,
            "nltk_neg": scores["neg"],
            "nltk_neu": scores["neu"],
            "nltk_pos": scores["pos"],
            "nltk_compound": compound
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "sentiment_nltk.csv", index=False, encoding="utf-8")
    return out


def run_hf_sentiment(
    df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
) -> pd.DataFrame:
    clf = pipeline("sentiment-analysis", model=model_name)
    rows = []

    for _, row in df.iterrows():
        text = row["sentence"][:512]
        pred = clf(text)[0]
        label = normalize_hf_sentiment(pred["label"])

        rows.append({
            "sentence_id": row["sentence_id"],
            "sentence": row["sentence"],
            "hf_label": label,
            "hf_score": float(pred["score"])
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "sentiment_hf.csv", index=False, encoding="utf-8")
    return out


# ----------------------------
# NER / entity classification
# ----------------------------
def extract_nltk_entities(text: str):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    chunked = ne_chunk(tagged)

    entities = []
    for node in chunked:
        if isinstance(node, Tree):
            entity_text = " ".join(word for word, _ in node.leaves())
            entity_label = node.label()
            entities.append((entity_text, entity_label))
    return entities


def run_nltk_ner(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        entities = extract_nltk_entities(row["sentence"])
        for entity_text, entity_label in entities:
            rows.append({
                "sentence_id": row["sentence_id"],
                "sentence": row["sentence"],
                "entity_text": entity_text,
                "entity_label": entity_label
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "ner_nltk.csv", index=False, encoding="utf-8")
    return out


def run_hf_ner(
    df: pd.DataFrame,
    model_name: str = "dslim/bert-base-NER"
) -> pd.DataFrame:
    ner_pipe = pipeline("ner", model=model_name, aggregation_strategy="simple")
    rows = []

    for _, row in df.iterrows():
        preds = ner_pipe(row["sentence"][:512])
        for pred in preds:
            rows.append({
                "sentence_id": row["sentence_id"],
                "sentence": row["sentence"],
                "entity_text": pred["word"],
                "entity_label": pred["entity_group"],
                "score": float(pred["score"])
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "ner_hf.csv", index=False, encoding="utf-8")
    return out


def run_entity_classification_comparison(nltk_ner_df: pd.DataFrame, hf_ner_df: pd.DataFrame) -> pd.DataFrame:
    nltk_counts = nltk_ner_df["entity_label"].value_counts().to_dict() if not nltk_ner_df.empty else {}
    hf_counts = hf_ner_df["entity_label"].value_counts().to_dict() if not hf_ner_df.empty else {}

    labels = sorted(set(nltk_counts) | set(hf_counts))
    rows = []
    for label in labels:
        rows.append({
            "entity_label": label,
            "nltk_count": nltk_counts.get(label, 0),
            "hf_count": hf_counts.get(label, 0)
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "entity_classification_comparison.csv", index=False, encoding="utf-8")
    return out


# ----------------------------
# Zero-shot classification
# ----------------------------
def run_zero_shot(
    df: pd.DataFrame,
    labels=None,
    model_name: str = "facebook/bart-large-mnli"
) -> pd.DataFrame:
    if labels is None:
        labels = ["politics", "governance", "economy", "war/security", "society"]

    classifier = pipeline("zero-shot-classification", model=model_name)
    rows = []

    for _, row in df.iterrows():
        pred = classifier(row["sentence"][:512], candidate_labels=labels)
        rows.append({
            "sentence_id": row["sentence_id"],
            "sentence": row["sentence"],
            "top_label": pred["labels"][0],
            "top_score": float(pred["scores"][0]),
            "all_labels": json.dumps(pred["labels"]),
            "all_scores": json.dumps([float(x) for x in pred["scores"]])
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "zero_shot.csv", index=False, encoding="utf-8")
    return out


# ----------------------------
# Summarization
# ----------------------------
def run_summarization(
    text: str,
    model_name: str = "facebook/bart-large-cnn"
) -> tuple[str, float]:
    summarizer = pipeline("summarization", model=model_name)

    chunks = chunk_text(text, max_words=350)
    partial_summaries = []

    for chunk in chunks[:10]:
        summary = summarizer(chunk, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
        partial_summaries.append(summary)

    combined_summary = " ".join(partial_summaries)

    if len(combined_summary.split()) > 300:
        final_summary = summarizer(
            combined_summary,
            max_length=150,
            min_length=60,
            do_sample=False
        )[0]["summary_text"]
    else:
        final_summary = combined_summary

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(final_summary)

    original_words = max(len(text.split()), 1)
    summary_words = len(final_summary.split())
    compression_ratio = summary_words / original_words

    return final_summary, compression_ratio


# ----------------------------
# Metrics helpers
# ----------------------------
def safe_mean(series):
    return float(series.mean()) if len(series) > 0 else 0.0


def sentiment_agreement(nltk_sent_df: pd.DataFrame, hf_sent_df: pd.DataFrame) -> float:
    merged = nltk_sent_df.merge(hf_sent_df, on=["sentence_id", "sentence"], how="inner")
    if merged.empty:
        return 0.0
    return float((merged["nltk_label"] == merged["hf_label"]).mean())


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dirs()

    # Use local file-based tracking 
    mlflow.set_tracking_uri(str(SCRIPT_DIR / "mlruns"))
    mlflow.set_experiment("hw-02-nlp")

    with mlflow.start_run():
        # ---------------- Dataset ----------------
        df, full_text = load_speech_dataset()
        dataset_name = "nltk_inaugural_addresses"
        data_source = "nltk inaugural corpus (open speech dataset)"
        num_sentences = len(df)

        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("num_sentences", num_sentences)
        mlflow.log_param("data_source", data_source)

        # ---------------- Sentiment ----------------
        nltk_sent_df = run_nltk_sentiment(df)
        hf_sent_df = run_hf_sentiment(df)

        nltk_sent_counts = nltk_sent_df["nltk_label"].value_counts().to_dict()
        hf_sent_counts = hf_sent_df["hf_label"].value_counts().to_dict()

        mlflow.log_metric("sentiment_nltk_positive", nltk_sent_counts.get("positive", 0))
        mlflow.log_metric("sentiment_nltk_neutral", nltk_sent_counts.get("neutral", 0))
        mlflow.log_metric("sentiment_nltk_negative", nltk_sent_counts.get("negative", 0))

        mlflow.log_metric("sentiment_hf_positive", hf_sent_counts.get("positive", 0))
        mlflow.log_metric("sentiment_hf_neutral", hf_sent_counts.get("neutral", 0))
        mlflow.log_metric("sentiment_hf_negative", hf_sent_counts.get("negative", 0))

        mlflow.log_metric("sentiment_method_agreement", sentiment_agreement(nltk_sent_df, hf_sent_df))
        mlflow.log_metric("sentiment_nltk_compound_mean", safe_mean(nltk_sent_df["nltk_compound"]))
        mlflow.log_metric("sentiment_hf_score_mean", safe_mean(hf_sent_df["hf_score"]))

        # ---------------- NER ----------------
        nltk_ner_df = run_nltk_ner(df)
        hf_ner_df = run_hf_ner(df)

        mlflow.log_metric("ner_nltk_total_entities", len(nltk_ner_df))
        mlflow.log_metric("ner_hf_total_entities", len(hf_ner_df))
        mlflow.log_metric("ner_nltk_unique_entities", int(nltk_ner_df["entity_text"].nunique()) if not nltk_ner_df.empty else 0)
        mlflow.log_metric("ner_hf_unique_entities", int(hf_ner_df["entity_text"].nunique()) if not hf_ner_df.empty else 0)

        # ---------------- Entity classification ----------------
        ent_cmp_df = run_entity_classification_comparison(nltk_ner_df, hf_ner_df)

        mlflow.log_metric("entity_class_nltk_unique_types", int(nltk_ner_df["entity_label"].nunique()) if not nltk_ner_df.empty else 0)
        mlflow.log_metric("entity_class_hf_unique_types", int(hf_ner_df["entity_label"].nunique()) if not hf_ner_df.empty else 0)

        for label, count in nltk_ner_df["entity_label"].value_counts().to_dict().items():
            mlflow.log_metric(f"entity_class_nltk_count_{label}", count)

        for label, count in hf_ner_df["entity_label"].value_counts().to_dict().items():
            mlflow.log_metric(f"entity_class_hf_count_{label}", count)

        # ---------------- Zero-shot classification ----------------
        zero_labels = ["politics", "governance", "economy", "war/security", "society"]
        zero_df = run_zero_shot(df, labels=zero_labels)

        zero_counts = zero_df["top_label"].value_counts().to_dict()
        for label in zero_labels:
            metric_name = label.replace("/", "_")
            mlflow.log_metric(f"zeroshot_count_{metric_name}", zero_counts.get(label, 0))

        mlflow.log_metric("zeroshot_top_score_mean", safe_mean(zero_df["top_score"]))

        # ---------------- Summarization ----------------
        long_text_for_summary = " ".join(df["sentence"].head(200).tolist())
        summary, compression_ratio = run_summarization(long_text_for_summary)

        mlflow.log_metric("summarization_input_word_count", len(long_text_for_summary.split()))
        mlflow.log_metric("summarization_output_word_count", len(summary.split()))
        mlflow.log_metric("summarization_compression_ratio_mean", compression_ratio)

        # ---------------- Artifacts ----------------
        mlflow.log_artifact(str(DATA_DIR / "sentences.csv"))
        mlflow.log_artifact(str(OUTPUT_DIR / "sentiment_nltk.csv"))
        mlflow.log_artifact(str(OUTPUT_DIR / "sentiment_hf.csv"))
        mlflow.log_artifact(str(OUTPUT_DIR / "ner_nltk.csv"))
        mlflow.log_artifact(str(OUTPUT_DIR / "ner_hf.csv"))
        mlflow.log_artifact(str(OUTPUT_DIR / "entity_classification_comparison.csv"))
        mlflow.log_artifact(str(OUTPUT_DIR / "zero_shot.csv"))
        mlflow.log_artifact(str(OUTPUT_DIR / "summary.txt"))

        print(f"Logged NLP results: {num_sentences} sentences processed")


if __name__ == "__main__":
    main()