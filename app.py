"""View: Streamlit dashboard. Run with `streamlit run app.py`."""

import logging
import sys
import time
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

# reads the DB (no fetch); cleared after a top-up, 1h TTL bounds CLI-fill staleness
get_data = st.cache_data(ttl="1h")(controller.get_data)


def fill(label: str, fetch, retries: int = 3, cooldown: int = 60) -> None:
    """Run a fetch action. Only rate-limited sources (GDELT's 429) are auto-
    retried after a visible one-minute countdown; other errors surface at once.
    Each source that succeeds is saved, so retries only re-hit what's throttled."""
    with st.spinner(label):
        retry, errors = fetch()
    box = st.empty()
    while retry and retries:
        retries -= 1
        for s in range(cooldown, 0, -1):
            box.info(f"⏳ {', '.join(retry)} rate-limitad — försöker igen om {s}s…")
            time.sleep(1)
        with st.spinner(f"Försöker igen: {', '.join(retry)}…"):
            retry, more = controller._fetch_all(retry)
            errors |= more  # a retry can still surface a non-rate-limit error
    box.empty()
    get_data.clear()  # DB changed -> reload on next "Ladda data"
    if errors:
        st.error("Fel: " + "; ".join(f"{t}: {m}" for t, m in errors.items()))
    if retry:
        st.warning(f"Fortfarande rate-limitad: {', '.join(retry)}. Försök igen strax.")
    if not errors and not retry:
        st.success("Klart. Klicka 'Ladda data' för att träna på den.")


if st.button("Första fyllning (5 år)"):
    fill(
        "Hämtar ~5 års historik…",
        lambda: controller.backfill(query, spy_symbol, metal_symbol, oil_symbol),
    )

if st.button("Fyll på till idag"):
    fill(
        "Hämtar från senaste lagrade datum till idag…",
        lambda: controller.update(query, spy_symbol, metal_symbol, oil_symbol),
    )

if st.button("Ladda data"):
    try:
        st.session_state["df"] = get_data()
    except Exception as e:  # noqa: BLE001 - empty/mismatched DB -> guide the user
        st.error(
            f"Kunde inte bygga feature-tabellen: {e}. Kör 'Första fyllning' först."
        )

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
