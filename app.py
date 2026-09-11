"""View: Streamlit dashboard. Run with `streamlit run app.py`."""

import logging
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent / "src")
)  # ponytail: path shim, drop after `pip install -e .`

import streamlit as st

from we_love_shorting import controller

logging.basicConfig(level=logging.INFO)

st.title("we_love_shorting — price → news-tone signal")
st.caption(
    "SPY + precious-metal + oil prices → predicted news tone. Low tone = bearish."
)

query = st.text_input("GDELT query", "recession")
spy_symbol = st.text_input("Index ticker", "SPY")
metal_symbol = st.text_input("Precious-metal ticker", "GC=F")
oil_symbol = st.text_input("Oil ticker", "CL=F")

run = st.cache_data(ttl="1h")(controller.run)  # avoid re-hitting GDELT on every click

if st.button("Run flow"):
    df = run(query, spy_symbol, metal_symbol, oil_symbol)
    st.line_chart(df.set_index("date")[["tone", "predicted_tone"]])
    st.dataframe(df.tail(20), use_container_width=True)
