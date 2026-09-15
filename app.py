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

get_data = st.cache_data(ttl="1h")(controller.get_data)  # don't re-hit GDELT per click

if st.button("Fetch data"):
    st.session_state["df"] = get_data(query, spy_symbol, metal_symbol, oil_symbol)

if "df" in st.session_state:
    df = st.session_state["df"]
    # numeric columns are the feature/target menu; drop bookkeeping columns
    candidates = [c for c in df.select_dtypes("number").columns if c != "market_closed"]
    target = st.selectbox(
        "Target (predict this)",
        candidates,
        index=candidates.index("tone") if "tone" in candidates else 0,
    )
    features = st.multiselect(
        "Features (predict from these)", [c for c in candidates if c != target]
    )

    if features:
        out = controller.run(df.copy(), features, target)  # cheap: no re-fetch
        st.line_chart(out.set_index("date")[[target, f"predicted_{target}"]])
        styled = out.style.apply(
            lambda row: (
                ["background-color: #5a1f1f" if row.get("market_closed") else ""]
                * len(row)
            ),
            axis=1,
        )  # market_closed drives the row colour; hidden via column_config
        st.dataframe(
            styled,
            use_container_width=True,
            column_config={"market_closed": None},  # None = hide, Styler still reads it
        )
        st.caption(
            "🟥 Röd rad = börsen stängd (helg/helgdag), föregående close används."
        )
    else:
        st.info("Välj minst en feature.")
