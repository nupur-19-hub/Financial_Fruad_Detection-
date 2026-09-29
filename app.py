"""
app.py

Streamlit interface for the fraud/AML detection MVP.
Anyone can upload their own transactions CSV — no hardcoded file path.
Runs fully offline: no external API calls in this part of the project.
"""

import streamlit as st

from data_loader import load_transactions
from graph_features import build_graph, compute_account_features
from detection import combine_flags

st.set_page_config(page_title="Fraud/AML Detection MVP", layout="wide")

st.title("Fraud / AML Detection — MVP")
st.caption(
    "Upload a transactions CSV to detect suspicious accounts using graph "
    "and statistical anomaly rules. No AI agent in this part — pure "
    "statistics and graph analysis, fully offline."
)

uploaded_file = st.file_uploader("Upload a transactions CSV", type=["csv"])

with st.sidebar:
    st.header("Detection settings")
    k = st.slider(
        "Sensitivity (standard deviations above mean)",
        min_value=1.0, max_value=4.0, value=2.0, step=0.5,
        help="Lower = more accounts flagged (more sensitive). Higher = fewer, stronger flags.",
    )
    window_hours = st.number_input(
        "Velocity window (hours)", min_value=1, max_value=168, value=24,
        help="Time window used to measure how fast transactions arrive at an account.",
    )

if uploaded_file is None:
    st.info("Upload a CSV to get started. Expected columns (names can vary): "
            "sender ID, receiver ID, amount, and a timestamp.")
    st.stop()

try:
    df = load_transactions(uploaded_file)
except ValueError as e:
    st.error(f"Could not process this file: {e}")
    st.stop()

st.success(f"Loaded {len(df):,} transactions across "
           f"{df['sender_id'].nunique() + df['receiver_id'].nunique():,} account references.")

with st.spinner("Building transaction graph and computing features..."):
    G = build_graph(df)
    features = compute_account_features(df, G, window_hours=window_hours)
    flagged = combine_flags(features, k=k)

col1, col2, col3 = st.columns(3)
col1.metric("Total accounts", f"{len(features):,}")
col2.metric("Flagged accounts", f"{len(flagged):,}")
col3.metric("Total flagged exposure", f"${flagged['inbound_volume'].sum():,.0f}")

st.subheader("Flagged accounts")
if flagged.empty:
    st.write("No accounts were flagged at this sensitivity level. Try lowering the sensitivity slider.")
else:
    display_cols = [
        "in_degree", "out_degree", "degree_ratio",
        "inbound_volume", "unique_senders", "max_velocity",
        "concentration", "triggered_rules",
    ]
    st.dataframe(
        flagged[display_cols].style.format({
            "degree_ratio": "{:.2f}",
            "inbound_volume": "${:,.0f}",
            "concentration": "{:.2f}",
        }),
        use_container_width=True,
    )
    st.caption(
        "This is the guaranteed core system: every flag above comes from "
        "statistics and graph structure alone, with no API or AI model "
        "involved. The next project phase adds an AI agent that writes a "
        "plain-language investigation report for each flagged account."
    )
