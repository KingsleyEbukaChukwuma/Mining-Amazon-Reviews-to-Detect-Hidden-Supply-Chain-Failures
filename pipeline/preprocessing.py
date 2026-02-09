import re
import html
import pandas as pd

def normalize_text(t: str) -> str:
    if not isinstance(t, str):
        t = "" if t is None else str(t)
    t = html.unescape(t)
    t = re.sub(r"<br\s*/?>", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # numeric/time parsing
    df["timestamp"] = pd.to_numeric(df.get("timestamp"), errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    df["rating"] = pd.to_numeric(df.get("rating"), errors="coerce")
    df["verified_purchase"] = df.get("verified_purchase", False)
    df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)

    df["helpful_vote"] = pd.to_numeric(df.get("helpful_vote"), errors="coerce").fillna(0).astype(int)

    # text fields
    df["title"] = df.get("title", "").fillna("")
    df["text"] = df.get("text", "").fillna("")
    df["text_full"] = (df["title"] + " " + df["text"]).str.strip()
    df["text_full_clean"] = df["text_full"].apply(normalize_text)

    # simple features
    df["review_len"] = df["text_full"].str.len()

    # image flag
    if "images" in df.columns:
        df["has_image"] = df["images"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    else:
        df["has_image"] = False

    return df
