import numpy as np
import pandas as pd

def add_rating_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rating_bucket"] = pd.cut(
        df["rating"],
        bins=[0, 2, 3, 5],
        labels=["1-2", "3", "4-5"],
        include_lowest=True,
    )
    return df

def monthly_issue_rates(df: pd.DataFrame, issue_cols: list[str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp["year_month"] = pd.to_datetime(tmp["timestamp"], unit="ms").dt.to_period("M")

    monthly_rates = (
        tmp.groupby("year_month")[issue_cols]
        .mean()
        .reset_index()
        .sort_values("year_month")
    )
    return monthly_rates

def rolling_monthly_rates(monthly_rates: pd.DataFrame, issue_cols: list[str], window: int = 3) -> pd.DataFrame:
    out = monthly_rates.sort_values("year_month").copy()
    for col in issue_cols:
        out[col] = out[col].rolling(window, min_periods=1).mean()
    return out

def issue_rates_by_rating_bucket(df: pd.DataFrame, issue_cols: list[str]) -> pd.DataFrame:
    tmp = add_rating_bucket(df)
    bucket_rates = (
        tmp.groupby("rating_bucket", observed=False)[issue_cols]
        .mean()
        .reset_index()
    )
    return bucket_rates

def cooccurrence_matrix(df: pd.DataFrame, issue_cols: list[str]) -> pd.DataFrame:
    mat = pd.DataFrame(
        np.dot(df[issue_cols].T, df[issue_cols]),
        index=issue_cols,
        columns=issue_cols,
    )
    return mat

def cooccurrence_rate(cooccurrence: pd.DataFrame) -> pd.DataFrame:
    # conditional probabilities: divide each row by its diagonal
    return cooccurrence.div(np.diag(cooccurrence), axis=0)
