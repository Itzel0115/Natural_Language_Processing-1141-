# -*- coding: utf-8 -*-
"""
Assignment 1 - Word Embeddings
"""

from __future__ import annotations

import argparse
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE

import gensim.downloader
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.utils import simple_preprocess


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[1]          # .../assignments1
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# =========================
# External resources
# =========================
ANALOGY_URL = "https://download.tensorflow.org/data/questions-words.txt"
ANALOGY_TXT = DATA_DIR / "questions-words.txt"
ANALOGY_CSV = DATA_DIR / "questions-words.csv"

MODEL_NAME = "glove-wiki-gigaword-100"

# TA-provided, publicly listed (per your statement).
WIKI_IDS: Dict[str, str] = {
    "wiki_texts_part_0.txt.gz": "1jiu9E1NalT2Y8EIuWNa1xf2Tw1f1XuGd",
    "wiki_texts_part_1.txt.gz": "1ABblLRd9HXdXvaNv8H9fFq984bhnowoG",
    "wiki_texts_part_2.txt.gz": "1z2VFNhpPvCejTP5zyejzKj5YjI_Bn42M",
    "wiki_texts_part_3.txt.gz": "1VKjded9BxADRhIoCzXy_W8uzVOTWIf0g",
    "wiki_texts_part_4.txt.gz": "16mBeG26m9LzHXdPe8UrijUIc6sHxhknz",
    "wiki_texts_part_5.txt.gz": "17JFvxOH-kc-VmvGkhG7p3iSZSpsWdgJI",
    "wiki_texts_part_6.txt.gz": "19IvB2vOJRGlrYulnTXlZECR8zT5v550P",
    "wiki_texts_part_7.txt.gz": "1sjwO8A2SDOKruv6-8NEq7pEIuQ50ygVV",
    "wiki_texts_part_8.txt.gz": "1s7xKWJmyk98Jbq6Fi1scrHy7fr_ellUX",
    "wiki_texts_part_9.txt.gz": "17eQXcrvY1cfpKelLbP2BhQKrljnFNykr",
    "wiki_texts_part_10.txt.gz": "1J5TAN6bNBiSgTIYiPwzmABvGhAF58h62",
}

WIKI_COMBINED = DATA_DIR / "wiki_texts_combined.txt"
WIKI_SAMPLE_20 = DATA_DIR / "wiki_sample_20.txt"

W2V_MODEL_PATH = OUT_DIR / "wiki_w2v_20_final.model"
W2V_KV_PATH = OUT_DIR / "wiki_w2v_20_final.kv"

FIG_GLOVE = OUT_DIR / "word_relationships.png"
FIG_TRAINED = OUT_DIR / "word_relationships_trained.png"

SUB_CATEGORY = ": family"


# =========================
# Helpers: download / prepare
# =========================
def download_file(url: str, out_path: Path) -> None:
    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved to: {out_path}")


def ensure_analogy_txt() -> None:
    if ANALOGY_TXT.exists():
        print(f"[skip] {ANALOGY_TXT.name} already exists.")
        return
    download_file(ANALOGY_URL, ANALOGY_TXT)


def build_questions_csv() -> None:
    """
    Build questions-words.csv with columns:
      - Question (A B C D)
      - Category (semantic/syntactic)
      - SubCategory (e.g., ': family')
    """
    if ANALOGY_CSV.exists():
        print(f"[skip] {ANALOGY_CSV.name} already exists.")
        return

    print("Building questions-words.csv ...")
    with open(ANALOGY_TXT, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    questions: List[str] = []
    categories: List[str] = []
    sub_categories: List[str] = []

    current_subcat: Optional[str] = None
    subcat_count = 0

    for line in lines:
        if not line.strip():
            continue
        if line.startswith(":"):
            current_subcat = line.strip()
            subcat_count += 1
            continue

        toks = line.strip().split()
        if len(toks) == 4 and current_subcat is not None:
            questions.append(" ".join(toks))
            sub_categories.append(current_subcat)
            categories.append("semantic" if subcat_count <= 5 else "syntactic")

    df = pd.DataFrame(
        {"Question": questions, "Category": categories, "SubCategory": sub_categories}
    )
    df.to_csv(ANALOGY_CSV, index=False)
    print(f"Saved: {ANALOGY_CSV} (rows={len(df):,})")


def ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: gdown\n"
            "Please install it:\n"
            "  pip install gdown\n"
        ) from e


def download_wiki_parts() -> None:
    """
    Download wiki_texts_part_*.txt.gz to data/ using gdown.
    """
    ensure_gdown()
    import gdown

    print("Downloading wiki parts (TA-provided ids) ...")
    for fname, fid in WIKI_IDS.items():
        out_path = DATA_DIR / fname
        if out_path.exists():
            print(f"[skip] {fname}")
            continue
        url = f"https://drive.google.com/uc?id={fid}"
        print(f"  - {fname}")
        gdown.download(url, str(out_path), quiet=False)


def build_combined_wiki() -> None:
    """
    Combine wiki gz parts into a single text file.
    This avoids creating many .txt extracted files.
    """
    if WIKI_COMBINED.exists():
        print(f"[skip] {WIKI_COMBINED.name} already exists.")
        return

    import gzip
    import shutil

    print("Building wiki_texts_combined.txt ...")
    with open(WIKI_COMBINED, "wb") as w:
        for i in range(0, 11):
            gz_path = DATA_DIR / f"wiki_texts_part_{i}.txt.gz"
            if not gz_path.exists():
                raise FileNotFoundError(f"Missing: {gz_path}. Run download step first.")
            with gzip.open(gz_path, "rb") as f:
                shutil.copyfileobj(f, w)
    print(f"Saved: {WIKI_COMBINED}")


def sample_wiki(input_path: Path, output_path: Path, ratio: float = 0.20, seed: int = 42) -> None:
    if output_path.exists():
        print(f"[skip] {output_path.name} already exists.")
        return

    print(f"Sampling wiki: ratio={ratio:.0%}, seed={seed}")
    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if random.random() < ratio:
                f_out.write(line)

    print(f"Saved sample: {output_path}")


# =========================
# Evaluation / plotting
# =========================
def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    """
    Accuracy ignoring None/NaN predictions.
    """
    gold = np.asarray(gold, dtype=object)
    pred = np.asarray(pred, dtype=object)

    valid = (~pd.isna(gold)) & (pred != None)  # noqa: E711
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(gold[valid] == pred[valid]))


def eval_analogy(model, df: pd.DataFrame, desc: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """
    Return aligned gold/pred arrays with length == len(df).
    """
    golds = np.empty(len(df), dtype=object)
    preds = np.empty(len(df), dtype=object)

    for i, analogy in enumerate(tqdm(df["Question"].tolist(), desc=desc)):
        toks = str(analogy).strip().split()
        if len(toks) != 4:
            golds[i] = np.nan
            preds[i] = None
            continue

        a, b, c, d = toks[0], toks[1], toks[2], toks[3]
        golds[i] = d

        try:
            # Only require a,b,c in vocab for vector arithmetic
            if all(w in model for w in (a, b, c)):
                result = model.most_similar(positive=[b, c], negative=[a], topn=10)
                pred_word = None
                for w, _score in result:
                    if w not in {a, b, c}:
                        pred_word = w
                        break
                preds[i] = pred_word if pred_word else result[0][0]
            else:
                preds[i] = None
        except Exception:
            preds[i] = None

    return golds, preds


def print_eval_by_group(df: pd.DataFrame, golds: np.ndarray, preds: np.ndarray) -> None:
    print("\n=== Category Evaluation ===")
    for category in df["Category"].unique():
        mask = (df["Category"] == category).to_numpy()
        acc = calculate_accuracy(golds[mask], preds[mask])
        print(f"Category: {category}, Accuracy: {acc * 100:.2f}%")

    print("\n=== Sub-Category Evaluation ===")
    for subcat in df["SubCategory"].unique():
        mask = (df["SubCategory"] == subcat).to_numpy()
        acc = calculate_accuracy(golds[mask], preds[mask])
        print(f"{subcat}: {acc * 100:.2f}%")


def plot_tsne_for_subcategory(df: pd.DataFrame, kv, sub_category: str, out_path: Path, title: str) -> None:
    """
    Plot t-SNE for unique words appearing in a sub-category.
    """
    sub_df = df[df["SubCategory"] == sub_category]
    words_set = set()
    for q in sub_df["Question"].tolist():
        toks = str(q).strip().lower().split()
        if len(toks) == 4:
            words_set.update(toks)

    words_list = [w for w in words_set if w in kv]
    if len(words_list) < 5:
        print(f"Too few words in vocab for t-SNE ({len(words_list)}). Skip plotting.")
        return

    vectors = np.array([kv[w] for w in words_list])
    perplexity = min(30, max(5, len(words_list) - 1))

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000, init="pca")
    emb2d = tsne.fit_transform(vectors)

    plt.figure(figsize=(14, 10))
    plt.scatter(emb2d[:, 0], emb2d[:, 1], alpha=0.6, s=150, edgecolors="black", linewidth=0.5)

    for i, w in enumerate(words_list):
        plt.annotate(
            w,
            xy=(emb2d[i, 0], emb2d[i, 1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
        )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved figure: {out_path}")


# =========================
# Train Word2Vec
# =========================
def get_stopwords() -> set:
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        # Fallback minimal list
        return set((
            "a","an","the","and","or","but","if","while","is","am","are","was","were","be","been","being",
            "of","to","in","for","on","at","by","with","from","as","that","this","it","its","into","about",
            "over","after","before","between","under","above","out","up","down","not","no","so","than","too","very"
        ))


class WikiSentences:
    def __init__(self, filepath: Path, stopwords: set):
        self.filepath = filepath
        self.stopwords = stopwords
        self.alpha_re = re.compile(r"[^a-zA-Z\s]+")

    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.lower()
                line = self.alpha_re.sub(" ", line)
                tokens = simple_preprocess(line, deacc=True, min_len=2, max_len=30)
                tokens = [t for t in tokens if t not in self.stopwords]
                if tokens:
                    yield tokens


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.t0 = None

    def on_epoch_begin(self, model):
        self.t0 = time.perf_counter()
        print(f"→ epoch {self.epoch + 1} start")

    def on_epoch_end(self, model):
        dt = time.perf_counter() - self.t0
        print(f"✓ epoch {self.epoch + 1} done in {dt:.1f}s")
        self.epoch += 1


def train_word2vec(sample_path: Path) -> Word2Vec:
    stopwords = get_stopwords()
    sentences = WikiSentences(sample_path, stopwords=stopwords)

    VECTOR_SIZE = 70
    WINDOW = 5
    MIN_COUNT = 10
    NEGATIVE = 5
    EPOCHS = 2

    print("Training Word2Vec (20% corpus) ...")
    epoch_logger = EpochLogger()

    w2v = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,
        workers=os.cpu_count() or 4,
        epochs=EPOCHS,
        negative=NEGATIVE,
        sample=1e-3,
        seed=42,
    )

    # Build vocab
    w2v.build_vocab(corpus_iterable=sentences, progress_per=100_000)
    print(f"vocab size = {len(w2v.wv):,}")

    # Recreate iterable (single pass)
    sentences = WikiSentences(sample_path, stopwords=stopwords)

    t0 = time.perf_counter()
    w2v.train(
        corpus_iterable=sentences,
        total_examples=w2v.corpus_count,
        epochs=w2v.epochs,
        callbacks=[epoch_logger],
    )
    total = time.perf_counter() - t0
    print(f"training done in {total:.1f}s | epochs={EPOCHS}")

    # Save
    w2v.save(W2V_MODEL_PATH)
    w2v.wv.save(W2V_KV_PATH)
    print(f"Saved model: {W2V_MODEL_PATH}")
    print(f"Saved keyed vectors: {W2V_KV_PATH}")

    return w2v


# =========================
# Main workflows
# =========================
def run_glove(df: pd.DataFrame) -> None:
    print("\n==> Loading GloVe model:", MODEL_NAME)
    glove = gensim.downloader.load(MODEL_NAME)
    print("GloVe loaded.")

    golds, preds = eval_analogy(glove, df, desc="GloVe analogy")
    overall = calculate_accuracy(golds, preds)
    print("\n=== GloVe Overall Accuracy ===")
    print(f"Overall Accuracy: {overall * 100:.2f}%")

    print_eval_by_group(df, golds, preds)

    # t-SNE plot
    plot_tsne_for_subcategory(
        df=df,
        kv=glove,
        sub_category=SUB_CATEGORY,
        out_path=FIG_GLOVE,
        title="Word Relationships from Google Analogy Task (GloVe)",
    )


def run_train(df: pd.DataFrame) -> None:
    # Download & build wiki combined
    download_wiki_parts()
    build_combined_wiki()

    # Sample
    sample_wiki(WIKI_COMBINED, WIKI_SAMPLE_20, ratio=0.20, seed=42)

    # Train / load
    if W2V_MODEL_PATH.exists():
        print(f"[skip] model exists: {W2V_MODEL_PATH.name} (loading)")
        my_model = Word2Vec.load(W2V_MODEL_PATH)
    else:
        my_model = train_word2vec(WIKI_SAMPLE_20)

    # Eval
    golds, preds = eval_analogy(my_model.wv, df, desc="Custom W2V analogy")
    overall = calculate_accuracy(golds, preds)
    print("\n=== Custom Model Overall Accuracy ===")
    print(f"Overall Accuracy: {overall * 100:.2f}%")

    print_eval_by_group(df, golds, preds)

    # t-SNE plot
    plot_tsne_for_subcategory(
        df=df,
        kv=my_model.wv,
        sub_category=SUB_CATEGORY,
        out_path=FIG_TRAINED,
        title="Word Relationships from Google Analogy Task (Trained Word2Vec)",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["glove", "train", "all"],
        default="all",
        help="glove: only pretrained eval; train: only custom word2vec; all: both",
    )
    args = parser.parse_args()

    # Prepare analogy dataset + csv
    ensure_analogy_txt()
    build_questions_csv()

    # Load csv once
    df = pd.read_csv(ANALOGY_CSV)

    if args.mode in ("glove", "all"):
        run_glove(df)

    if args.mode in ("train", "all"):
        run_train(df)

    print("\n✅ Done.")
    print(f"- Data dir: {DATA_DIR}")
    print(f"- Outputs : {OUT_DIR}")


if __name__ == "__main__":
    main()