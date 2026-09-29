"""
graph_features.py

Builds a transaction graph with NetworkX and computes per-account features
used by the detection layer: degree ratio, inbound volume, transaction
velocity, and sender concentration.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed transaction graph.

    Nodes are accounts. Each edge (sender -> receiver) carries the total
    amount transferred and the number of transactions between that pair.
    """
    G = nx.DiGraph()
    for row in df.itertuples(index=False):
        if G.has_edge(row.sender_id, row.receiver_id):
            G[row.sender_id][row.receiver_id]["weight"] += row.amount
            G[row.sender_id][row.receiver_id]["count"] += 1
        else:
            G.add_edge(row.sender_id, row.receiver_id, weight=row.amount, count=1)
    return G


def _max_rolling_count(group: pd.DataFrame, window_hours: int) -> float:
    """Largest number of inbound transactions within any window_hours-long window."""
    if len(group) == 0:
        return 0.0
    g = group.sort_values("timestamp").set_index("timestamp")
    counts = g["amount"].rolling(f"{window_hours}h").count()
    return float(counts.max())


def _concentration(group: pd.DataFrame) -> float:
    """
    Herfindahl-Hirschman style concentration of inbound volume across senders.
    Ranges from ~1/n (spread evenly across n senders) to 1 (all from one sender).
    """
    totals = group.groupby("sender_id")["amount"].sum()
    total = totals.sum()
    if total == 0:
        return 0.0
    shares = totals / total
    return float((shares ** 2).sum())


def compute_account_features(df: pd.DataFrame, G: nx.DiGraph, window_hours: int = 24) -> pd.DataFrame:
    """
    Compute one row of features per account (using each account's role as a
    receiver for inbound-focused features, since that's where laundering
    hubs show up).

    Returns
    -------
    pd.DataFrame indexed by account_id with columns:
        in_degree, out_degree, degree_ratio,
        inbound_volume, unique_senders, tx_count,
        max_velocity, concentration
    """
    all_accounts = pd.unique(pd.concat([df["sender_id"], df["receiver_id"]]))
    features = pd.DataFrame(index=all_accounts)
    features.index.name = "account_id"

    features["in_degree"] = [G.in_degree(n) if n in G else 0 for n in features.index]
    features["out_degree"] = [G.out_degree(n) if n in G else 0 for n in features.index]
    # Avoid division by zero; accounts with no outbound activity get a large
    # finite ratio proportional to their inbound activity instead of inf/NaN.
    features["degree_ratio"] = features["in_degree"] / features["out_degree"].replace(0, 0.5)

    inbound = df.groupby("receiver_id").agg(
        inbound_volume=("amount", "sum"),
        unique_senders=("sender_id", "nunique"),
        tx_count=("sender_id", "count"),
    )
    features = features.join(inbound, how="left")
    features[["inbound_volume", "unique_senders", "tx_count"]] = features[
        ["inbound_volume", "unique_senders", "tx_count"]
    ].fillna(0)

    velocity = df.groupby("receiver_id").apply(lambda g: _max_rolling_count(g, window_hours))
    features["max_velocity"] = velocity.reindex(features.index).fillna(0)

    concentration = df.groupby("receiver_id").apply(_concentration)
    features["concentration"] = concentration.reindex(features.index).fillna(0)

    return features
