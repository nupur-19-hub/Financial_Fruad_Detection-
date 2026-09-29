"""
data_loader.py

Loads a transactions CSV (from a file path or an uploaded file-like object,
e.g. from Streamlit's file_uploader) into a clean, standardized DataFrame.

Works with different datasets by matching common column-name variants instead
of requiring one exact schema.
"""

from __future__ import annotations

import pandas as pd

# Canonical field -> list of accepted column-name variants (case-insensitive).
COLUMN_ALIASES: dict[str, list[str]] = {
    "sender_id": ["sender_id", "from_account", "sender", "orig", "nameorig", "source"],
    "receiver_id": ["receiver_id", "to_account", "receiver", "dest", "namedest", "target"],
    "amount": ["amount", "amt", "value", "transaction_amount"],
    "timestamp": ["timestamp", "time", "date", "step", "transaction_date"],
}

REQUIRED_FIELDS = ["sender_id", "receiver_id", "amount", "timestamp"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name in df for a list of aliases (case-insensitive)."""
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def load_transactions(file) -> pd.DataFrame:
    """
    Load and clean a transactions CSV.

    Parameters
    ----------
    file : str, Path, or file-like object
        A file path, or an uploaded file object (e.g. from st.file_uploader).

    Returns
    -------
    pd.DataFrame
        Columns standardized to: sender_id, receiver_id, amount, timestamp.

    Raises
    ------
    ValueError
        If a required field cannot be matched to any column in the file.
    """
    df = pd.read_csv(file)

    rename_map = {}
    missing = []
    for field in REQUIRED_FIELDS:
        found = _find_column(df, COLUMN_ALIASES[field])
        if found is None:
            missing.append(field)
        else:
            rename_map[found] = field

    if missing:
        raise ValueError(
            "Could not find required column(s): "
            f"{', '.join(missing)}. "
            f"Found columns: {list(df.columns)}. "
            "Add the actual column name(s) to COLUMN_ALIASES in data_loader.py "
            "if this dataset uses a different naming convention."
        )

    df = df.rename(columns=rename_map)[REQUIRED_FIELDS].copy()

    # Clean
    df = df.drop_duplicates()
    df = df.dropna(subset=REQUIRED_FIELDS)

    # If "timestamp" is actually a plain integer step counter (like PaySim's
    # "step" column), convert it into a synthetic datetime so time-window
    # logic downstream still works.
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="h", origin="2024-01-01")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    df["sender_id"] = df["sender_id"].astype(str)
    df["receiver_id"] = df["receiver_id"].astype(str)

    return df.reset_index(drop=True)
