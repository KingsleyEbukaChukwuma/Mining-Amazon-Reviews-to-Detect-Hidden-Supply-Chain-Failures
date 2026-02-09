import pandas as pd

DEFAULT_WEIGHTS = {
    "defective_product": 0.40,
    "missing_accessories": 0.20,
    "wrong_item": 0.15,
    "damaged_packaging": 0.15,
    "delivery_delay": 0.10,
}

def add_supplychain_risk_index(
    df: pd.DataFrame,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    scale: float = 100.0,
    out_col: str = "supplychain_risk_index",
) -> pd.DataFrame:
    df = df.copy()
    df[out_col] = sum(df[col] * w for col, w in weights.items()) * scale
    return df

def monthly_avg_risk(df: pd.DataFrame, risk_col: str = "supplychain_risk_index") -> pd.DataFrame:
    tmp = df.copy()
    tmp["year_month"] = pd.to_datetime(tmp["timestamp"], unit="ms").dt.to_period("M")
    risk_monthly = (
        tmp.groupby("year_month")[risk_col]
        .mean()
        .reset_index()
        .sort_values("year_month")
    )
    return risk_monthly

def risk_by_rating_bucket(df: pd.DataFrame, risk_col: str = "supplychain_risk_index") -> pd.Series:
    tmp = df.copy()
    tmp["rating_bucket"] = pd.cut(
        tmp["rating"],
        bins=[0, 2, 3, 5],
        labels=["1-2", "3", "4-5"],
        include_lowest=True,
    )
    order = ["1-2", "3", "4-5"]
    return (
        tmp.groupby("rating_bucket", observed=False)[risk_col]
        .mean()
        .reindex(order)
    )
