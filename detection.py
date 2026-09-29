"""
detection.py

Three independent, explainable anomaly rules, each based on a z-score
threshold, plus a combine_flags() function that merges them with OR logic
into one final flagged-accounts table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore_flag(series: pd.Series, k: float = 2.0) -> pd.Series:
    """Flag values more than k standard deviations above the mean."""
    series = series.fillna(0)
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=series.index)
    z = (series - series.mean()) / std
    return z > k


def flag_degree_ratio(features: pd.DataFrame, k: float = 2.0) -> pd.Series:
    """Flag accounts with an unusually high in-degree/out-degree ratio (many senders, few outputs)."""
    return _zscore_flag(features["degree_ratio"], k)


def flag_velocity(features: pd.DataFrame, k: float = 2.0) -> pd.Series:
    """Flag accounts receiving an unusually high number of transactions in a short window."""
    return _zscore_flag(features["max_velocity"], k)


def flag_amount_concentration(features: pd.DataFrame, k: float = 2.0) -> pd.Series:
    """Flag accounts whose inbound volume is unusually concentrated among few senders."""
    return _zscore_flag(features["concentration"], k)


def combine_flags(features: pd.DataFrame, k: float = 2.0) -> pd.DataFrame:
    """
    Apply all three rules and return only the flagged accounts, each annotated
    with which rule(s) triggered, sorted by inbound volume (highest first).
    """
    result = features.copy()
    result["flag_degree_ratio"] = flag_degree_ratio(features, k)
    result["flag_velocity"] = flag_velocity(features, k)
    result["flag_concentration"] = flag_amount_concentration(features, k)
    result["flagged"] = (
        result["flag_degree_ratio"] | result["flag_velocity"] | result["flag_concentration"]
    )

    def _reasons(row) -> str:
        reasons = []
        if row["flag_degree_ratio"]:
            reasons.append("degree ratio")
        if row["flag_velocity"]:
            reasons.append("velocity")
        if row["flag_concentration"]:
            reasons.append("amount concentration")
        return ", ".join(reasons)

    result["triggered_rules"] = result.apply(_reasons, axis=1)

    flagged = result[result["flagged"]].sort_values("inbound_volume", ascending=False)
    return flagged.drop(columns=["flagged"])
