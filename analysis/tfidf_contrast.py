from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_contrast_terms(
    df: pd.DataFrame,
    issue_col: str,
    text_col: str = "text_full_clean",
    sample_per_group: int = 50_000,
    ngram_range: tuple[int, int] = (1, 3),
    min_df: int = 10,
    max_df: float = 0.60,
    top_n: int = 30,
    seed: int = 123,
):
    pos = df.loc[df[issue_col] == 1, text_col].dropna()
    neg = df.loc[df[issue_col] == 0, text_col].dropna()

    if len(pos) > sample_per_group:
        pos = pos.sample(sample_per_group, random_state=seed)
    if len(neg) > sample_per_group:
        neg = neg.sample(sample_per_group, random_state=seed)

    texts = pd.concat([pos, neg], ignore_index=True)
    y = np.array([1] * len(pos) + [0] * len(neg))

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vec.fit_transform(texts)

    pos_mean = X[y == 1].mean(axis=0).A1
    neg_mean = X[y == 0].mean(axis=0).A1
    diff = pos_mean - neg_mean
    vocab = np.array(vec.get_feature_names_out())

    top_pos_idx = np.argsort(diff)[::-1][:top_n]
    top_neg_idx = np.argsort(diff)[:top_n]

    top_pos = pd.DataFrame(
        {
            "term": vocab[top_pos_idx],
            "pos_mean_tfidf": pos_mean[top_pos_idx],
            "neg_mean_tfidf": neg_mean[top_pos_idx],
            "diff": diff[top_pos_idx],
        }
    )

    top_neg = pd.DataFrame(
        {
            "term": vocab[top_neg_idx],
            "pos_mean_tfidf": pos_mean[top_neg_idx],
            "neg_mean_tfidf": neg_mean[top_neg_idx],
            "diff": diff[top_neg_idx],
        }
    )

    meta = {
        "issue_col": issue_col,
        "pos_n": int((y == 1).sum()),
        "neg_n": int((y == 0).sum()),
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_df": max_df,
    }
    return top_pos, top_neg, meta

def run_tfidf_for_issues(
    df: pd.DataFrame,
    issue_cols: list[str],
    out_dir: str | Path = "outputs/tfidf",
    seed: int = 123,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for issue in issue_cols:
        top_pos, top_neg, meta = tfidf_contrast_terms(
            df=df,
            issue_col=issue,
            seed=seed,
        )
        results[issue] = {"top_pos": top_pos, "top_neg": top_neg, "meta": meta}
        top_pos.to_csv(out_dir / f"{issue}_top_terms_issue.csv", index=False)
        top_neg.to_csv(out_dir / f"{issue}_top_terms_nonissue.csv", index=False)

    return results
