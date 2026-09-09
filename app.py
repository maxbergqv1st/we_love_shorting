"""View: Streamlit dashboard. Run with `streamlit run app.py`."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))  # ponytail: path shim, drop after `pip install -e .`

import streamlit as st

from we_love_shorting import controller

logging.basicConfig(level=logging.INFO)

st.title("we_love_shorting — GDELT tone signal")
st.caption("News tone about a topic → probability the index closes down tomorrow.")

query = st.text_input("GDELT query", "recession")
symbol = st.text_input("Ticker", "SPY")

run = st.cache_data(ttl="1h")(controller.run)  # avoid re-hitting GDELT on every click

if st.button("Run flow"):
    df = run(query, symbol)
    st.line_chart(df.set_index("date")[["tone", "short_prob"]])
    st.dataframe(df.tail(20), use_container_width=True)
