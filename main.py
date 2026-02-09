from pathlib import Path

from pipeline.sampling import (
    ReservoirSamplingConfig,
    reservoir_sample_balanced_by_rating,
    save_parquet,
    load_parquet,
)
from pipeline.preprocessing import preprocess_reviews
from labeling.rule_engine import label_issues
from analysis.tfidf_contrast import run_tfidf_for_issues
from analysis.risk_index import add_supplychain_risk_index

ISSUE_COLS = [
    "delivery_delay",
    "damaged_packaging",
    "defective_product",
    "wrong_item",
    "missing_accessories",
]

def main():
    data_path = Path("Electronics.jsonl.gz")
    sample_path = Path("data/reservoir_sample_400k.parquet")

    # 1) sample once (streaming)
    if not sample_path.exists():
        cfg = ReservoirSamplingConfig(n=400_000, seed=123, start_year=2019, end_year=2023)
        df_raw = reservoir_sample_balanced_by_rating(data_path, cfg)
        save_parquet(df_raw, sample_path)

    # 2) load + preprocess
    df = load_parquet(sample_path)
    df = preprocess_reviews(df)

    # 3) label issues
    df = label_issues(df)

    # 4) tfidf outputs
    tfidf_results = run_tfidf_for_issues(df, ISSUE_COLS, out_dir="outputs/tfidf", seed=123)

    # 5) risk index
    df = add_supplychain_risk_index(df)

    # optional: save final dataset
    save_parquet(df, "outputs/final_labeled.parquet")

    print("Done. Outputs in outputs/")

if __name__ == "__main__":
    main()
