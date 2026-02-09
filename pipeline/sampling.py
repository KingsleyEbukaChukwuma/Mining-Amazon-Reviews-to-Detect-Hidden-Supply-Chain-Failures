from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.ingestion import iter_reviews

@dataclass
class ReservoirSamplingConfig:
    n: int = 400_000
    seed: int = 123
    start_year: int = 2019
    end_year: int = 2023

def reservoir_sample_balanced_by_rating(
    path: str | Path,
    cfg: ReservoirSamplingConfig = ReservoirSamplingConfig(),
) -> pd.DataFrame:
    """
    Balanced reservoir sampling by rating (1–5) within a year range.
    Streaming, single pass, no full-load into memory.
    """
    rng = np.random.default_rng(cfg.seed)

    start_ms = int(datetime(cfg.start_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(cfg.end_year + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    per_rating = cfg.n // 5
    ratings = [1, 2, 3, 4, 5]

    buckets: dict[int, list[dict]] = {r: [] for r in ratings}
    counts_seen: dict[int, int] = {r: 0 for r in ratings}

    for row in iter_reviews(path):
        ts = row.get("timestamp")
        if ts is None:
            continue
        try:
            ts = int(float(ts))
        except Exception:
            continue

        if ts < start_ms or ts >= end_ms:
            continue

        r = row.get("rating")
        if r is None:
            continue
        try:
            r_int = int(round(float(r)))
        except Exception:
            continue
        if r_int not in buckets:
            continue

        counts_seen[r_int] += 1
        if len(buckets[r_int]) < per_rating:
            buckets[r_int].append(row)
        else:
            j = rng.integers(0, counts_seen[r_int])
            if j < per_rating:
                buckets[r_int][j] = row

    sample: list[dict] = []
    for r in ratings:
        sample.extend(buckets[r])

    return pd.DataFrame(sample[: cfg.n])

def save_parquet(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
