"""View: Streamlit dashboard. Run with `streamlit run app.py`."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent / "src")
)  # ponytail: path shim, drop after `pip install -e .`

import numpy as np
import pandas as pd
import streamlit as st

from we_love_shorting import controller, features

logging.basicConfig(level=logging.INFO)

st.markdown("### 📉 we_love_shorting")
st.title("Price → news-tone signal")
st.caption(
    "SPY + precious-metal + oil prices → predicted news tone. Low tone = bearish."
)

with st.sidebar:
    st.header("Inställningar")
    query = st.text_input("GDELT query (news tone)", "recession")  # only free knob left

# Human names per stream stem (features.TICKERS is the source of the stem set,
# so a new ticker always gets label entries — unnamed here, it falls back to the
# stem). The _close/_ret labels derive from one name so they never drift apart.
STREAM_NAMES = {
    "sp500": "S&P 500",
    "omx30": "OMX Stockholm 30",
    "eurostoxx": "EURO STOXX 50",
    "gold": "Gold",
    "silver": "Silver",
    "copper": "Copper",
    "oil": "Crude oil",
}
LABELS = {"tone": "News tone"}
for _stem in features.TICKERS:
    _name = STREAM_NAMES.get(_stem, _stem)
    LABELS[f"{_stem}_close"] = _name
    LABELS[f"{_stem}_ret"] = f"{_name} (daily % change)"

# Model roadmap: only Linear Regression is implemented (signal_model.py).
# The others are shown so the UI already has a place for them once built.
MODELS = ["Linear Regression", "Ridge Regression 🔒", "Random Forest 🔒"]

# Chart timespan filter: trading days to show, counting back from the latest row.
TIMESPANS = {"Vecka": 5, "Månad": 21, "År": 252, "Allt": None}

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
    if st.button("Första fyllning (10 år)"):
        fill("Hämtar ~10 års historik…", lambda: controller.backfill(query))

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
    st.caption(f"✅ Data laddad: {df['date'].min()} → {df['date'].max()}")
    # Feature/target menu = numeric columns, minus bookkeeping (market_closed) and
    # the non-stationary price LEVELS (_close): the model runs on daily change
    # (_ret) + tone only, for honest residuals. _close stays in df for the chart's
    # reconstructed price line (controller.run).
    candidates = [
        c
        for c in df.select_dtypes("number").columns
        if c != "market_closed" and not c.endswith("_close")
    ]

    with st.sidebar:
        st.subheader("Modell")
        model_choice = st.selectbox("MODELL", MODELS)
        if model_choice != "Linear Regression":
            st.caption("🔒 Kommer snart — kör Linear Regression tills vidare.")

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
        # Exclude the target's whole stream: choosing sp500_close (or _ret) as
        # target drops both sp500_close and sp500_ret from the features.
        target_stream = features.stream_of(target)
        feature_opts = [c for c in candidates if features.stream_of(c) != target_stream]
        # Persist the picks across reruns (key=), starting with everything
        # selected, then prune anything no longer valid — crucially the current
        # target, so choosing a column as target auto-drops it from features
        # while keeping the rest of the selection intact.
        st.session_state.setdefault("feature_cols", feature_opts)
        st.session_state["feature_cols"] = [
            c for c in st.session_state["feature_cols"] if c in feature_opts
        ]
        feature_cols = st.multiselect(
            "FEATURES (förutsäg från dessa)",
            feature_opts,
            key="feature_cols",
            format_func=lambda c: LABELS.get(c, c),
        )

    if feature_cols:
        name = LABELS.get(target, target)  # e.g. "News tone", not the raw column
        feature_names = ", ".join(LABELS.get(c, c) for c in feature_cols)
        st.info(
            f"🧠 Tränar **Linear Regression** → förutsäger **{name}** från {feature_names}"
        )
        out = controller.run(df.copy(), feature_cols, target)  # cheap: no re-fetch
        # The model predicts a _ret; show the reconstructed price line instead of the
        # raw % so the live panel reads in kronor/dollar (controller.run adds it).
        display_col = features.display_column(target)
        display_pred = f"predicted_{display_col}"
        disp_name = LABELS.get(display_col, display_col)
        latest = out.iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Senaste faktiska: {disp_name}", f"{latest[display_col]:.2f}")
        col2.metric(f"Senaste prediktion: {disp_name}", f"{latest[display_pred]:.2f}")
        col3.metric("Differens", f"{latest[display_pred] - latest[display_col]:.2f}")
        # "Faktisk" < "Prediktion" for every target, so the actual/prediction pair
        # keeps a stable order whether Streamlit colours by column or by (sorted)
        # series name — the explicit list then pins actual=blue, prediction=orange.
        actual, pred = f"Faktisk: {disp_name}", f"Prediktion: {disp_name}"
        chart_slot = st.empty()  # reserved above the timespan picker, filled below
        # `or "Allt"` keeps a selection even if the user deselects the control.
        timespan = (
            st.segmented_control("Visa", list(TIMESPANS), default="Allt") or "Allt"
        )
        days = TIMESPANS[timespan]
        windowed = out if days is None else out.tail(days)
        chart = windowed.set_index("date")[[display_col, display_pred]].rename(
            columns={display_col: actual, display_pred: pred}
        )
        chart_slot.line_chart(chart, color=["#4c78a8", "#f58518"])
        if target.endswith("_ret"):
            st.caption(
                "📈 Priset är rekonstruerat från förutsagd dagsförändring "
                "(gårdagens faktiska close × (1 + prediktion))."
            )
        styled = out.style.apply(
            lambda row: (
                ["background-color: #5a1f1f" if row.get("market_closed") else ""]
                * len(row)
            ),
            axis=1,
        )  # market_closed drives the row colour; hidden via column_config
        st.dataframe(
            styled,
            width="stretch",
            column_config={"market_closed": None},  # None = hide, Styler still reads it
        )
        st.caption(
            "🟥 Röd rad = börsen stängd (helg/helgdag), föregående close används."
        )

        st.subheader("Evaluering (train/test)")
        if st.button("Kör evaluering"):
            try:
                st.session_state["eval_result"] = controller.evaluate(
                    df, feature_cols, target
                )
                st.session_state["eval_feature_cols"] = feature_cols
                st.session_state["eval_target"] = target
            except Exception as e:  # noqa: BLE001 - too little data after filtering, etc.
                st.session_state.pop("eval_result", None)
                st.warning(f"Kunde inte evaluera: {e}")

        if "eval_result" in st.session_state:
            eval_result = st.session_state["eval_result"]
            used_target = st.session_state["eval_target"]
            used_feature_cols = st.session_state["eval_feature_cols"]

            # compare as sets: reselecting the same features in a different order
            # doesn't change the model, so it shouldn't flag the eval as stale.
            if used_target != target or set(used_feature_cols) != set(feature_cols):
                st.info(
                    "Valen ovan har ändrats sedan senaste evalueringen — "
                    "resultaten nedan gäller fortfarande föregående val. "
                    "Klicka 'Kör evaluering' för att uppdatera."
                )

            train_df, test_df = eval_result.train_df, eval_result.test_df
            display_cols = [
                "date",
                used_target,
                f"predicted_{used_target}",
                f"baseline_{used_target}",
                "residual",
                "baseline_residual",
            ]
            baseline_label = (
                "medelvärde (avkastning – nära stationär, så historiskt "
                "medelvärde är den naiva prognosen)"
                if eval_result.baseline_kind == "mean"
                else "persistence (nivå – starkt autokorrelerad, så gårdagens "
                "faktiska värde är den naiva prognosen)"
            )
            st.caption(
                f"Träning: {train_df['date'].min()}–{train_df['date'].max()} "
                f"({len(train_df)} rader, marknad stängd exkluderad). "
                f"Test: {test_df['date'].min()}–{test_df['date'].max()} "
                f"({len(test_df)} rader). Baseline: {baseline_label}."
            )
            st.caption(
                "Modellen tränades på: "
                + ", ".join(LABELS.get(c, c) for c in used_feature_cols)
            )

            metrics_df = pd.DataFrame(eval_result.metrics).rename(
                columns={"model": "Modell", "baseline": "Baseline"},
                index={"rmse": "RMSE", "mae": "MAE"},
            )
            st.dataframe(metrics_df, width="stretch")

            st.subheader("Residualanalys (testdata)")
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Residual vs. prediktion")
                st.scatter_chart(test_df, x=f"predicted_{used_target}", y="residual")
            with col2:
                st.caption("Residual över tid")
                st.line_chart(test_df.set_index("date")["residual"])

            st.caption("Histogram över residualer")
            counts, bin_edges = np.histogram(test_df["residual"], bins=20)
            # bin midpoints as a numeric index: always distinct (unlike the
            # rounded left-edge strings, which could collide for tiny residuals).
            mids = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_df = pd.DataFrame(
                {"Antal": counts}, index=pd.Index(mids, name="Residual")
            )
            st.bar_chart(hist_df)

            st.subheader("Testdata")
            st.dataframe(test_df[display_cols], width="stretch")
        else:
            st.info(
                "Klicka 'Kör evaluering' för att träna och utvärdera på test-split."
            )
    else:
        st.info("Välj minst en feature.")
