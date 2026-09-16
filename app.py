"""View: Streamlit dashboard. Run with `streamlit run app.py`."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent / "src")
)  # ponytail: path shim, drop after `pip install -e .`

import streamlit as st

from we_love_shorting import controller, features

logging.basicConfig(level=logging.INFO)

st.title("we_love_shorting — price → news-tone signal")
st.caption(
    "SPY + precious-metal + oil prices → predicted news tone. Low tone = bearish."
)

with st.sidebar:
    st.header("Inställningar")
    query = st.text_input("GDELT query (news tone)", "recession")  # only free knob left

# Readable labels for the fixed streams; keys are the DataFrame columns.
LABELS = {
    "tone": "News tone",
    "sp500_close": "S&P 500",
    "omx30_close": "OMX Stockholm 30",
    "eurostoxx_close": "EURO STOXX 50",
    "gold_close": "Gold",
    "silver_close": "Silver",
    "copper_close": "Copper",
    "oil_close": "Crude oil",
}

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


with st.sidebar:
    if st.button("Första fyllning (5 år)"):
        fill("Hämtar ~5 års historik…", lambda: controller.backfill(query))

    if st.button("Fyll på till idag"):
        fill(
            "Hämtar från senaste lagrade datum till idag…",
            lambda: controller.update(query),
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

    with st.sidebar:
        st.subheader("Vad ska modellen förutsäga?")
        default_target = (
            features.TARGET if features.TARGET in candidates else candidates[0]
        )
        target = st.selectbox(
            "TARGET (faktiskt värde att förutsäga)",
            candidates,
            index=candidates.index(default_target),
            format_func=lambda c: LABELS.get(c, c),
        )
        feature_opts = [c for c in candidates if c != target]
        feature_cols = st.multiselect(
            "FEATURES (förutsäg från dessa)",
            feature_opts,
            default=feature_opts,  # start with everything else selected
            format_func=lambda c: LABELS.get(c, c),
        )

    if feature_cols:
        name = LABELS.get(target, target)  # e.g. "News tone", not the raw column
        feature_names = ", ".join(LABELS.get(c, c) for c in feature_cols)
        st.info(
            f"🧠 Tränar **Linear Regression** → förutsäger **{name}** från {feature_names}"
        )
        out = controller.run(df.copy(), feature_cols, target)  # cheap: no re-fetch
        # "Faktisk" < "Prediktion" for every target, so the actual/prediction pair
        # keeps a stable order whether Streamlit colours by column or by (sorted)
        # series name — the explicit list then pins actual=blue, prediction=orange.
        actual, pred = f"Faktisk: {name}", f"Prediktion: {name}"
        chart = out.set_index("date")[[target, f"predicted_{target}"]].rename(
            columns={target: actual, f"predicted_{target}": pred}
        )
        st.line_chart(chart, color=["#4c78a8", "#f58518"])
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
