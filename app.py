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

from we_love_shorting import chatbot, controller, features

logging.basicConfig(level=logging.INFO)

st.markdown("### 📉 we_love_shorting")
st.title("Price → news-tone signal")
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
    "sp500_ret": "S&P 500 (daily % change)",
    "omx30_ret": "OMX Stockholm 30 (daily % change)",
    "eurostoxx_ret": "EURO STOXX 50 (daily % change)",
    "gold_ret": "Gold (daily % change)",
    "silver_ret": "Silver (daily % change)",
    "copper_ret": "Copper (daily % change)",
    "oil_ret": "Crude oil (daily % change)",
}

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
    st.caption(f"✅ Data laddad: {df['date'].min()} → {df['date'].max()}")
    # numeric columns are the feature/target menu; drop bookkeeping columns
    candidates = [c for c in df.select_dtypes("number").columns if c != "market_closed"]

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
        latest = out.iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Senaste faktiska: {name}", f"{latest[target]:.2f}")
        col2.metric(
            f"Senaste prediktion: {name}", f"{latest[f'predicted_{target}']:.2f}"
        )
        col3.metric(
            "Differens", f"{latest[f'predicted_{target}'] - latest[target]:.2f}"
        )
        # "Faktisk" < "Prediktion" for every target, so the actual/prediction pair
        # keeps a stable order whether Streamlit colours by column or by (sorted)
        # series name — the explicit list then pins actual=blue, prediction=orange.
        actual, pred = f"Faktisk: {name}", f"Prediktion: {name}"
        chart_slot = st.empty()  # reserved above the timespan picker, filled below
        timespan = st.radio("Visa", list(TIMESPANS), index=3, horizontal=True)
        days = TIMESPANS[timespan]
        windowed = out if days is None else out.tail(days)
        chart = windowed.set_index("date")[[target, f"predicted_{target}"]].rename(
            columns={target: actual, f"predicted_{target}": pred}
        )
        chart_slot.line_chart(chart, color=["#4c78a8", "#f58518"])
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

            if used_target != target or used_feature_cols != feature_cols:
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
            st.dataframe(metrics_df, use_container_width=True)

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
            hist_df = pd.DataFrame(
                {"Antal": counts},
                index=[f"{bin_edges[i]:.3g}" for i in range(len(counts))],
            )
            st.bar_chart(hist_df)

            st.subheader("Testdata")
            st.dataframe(test_df[display_cols], use_container_width=True)

            st.subheader("💬 Fråga om resultatet")
            st.caption(
                "AI-assistent grundad i evalueringen ovan (OpenRouter, gratis-modell)."
            )

            quick_questions = [
                "Varför presterar modellen bättre/sämre än baseline?",
                "Var i testperioden är felen som störst?",
                "Är modellen tillförlitlig nog att lita på?",
            ]
            clicked_question = None
            for col, q in zip(st.columns(len(quick_questions)), quick_questions):
                if col.button(q, use_container_width=True):
                    clicked_question = q

            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            for role, text in st.session_state["chat_history"]:
                with st.chat_message(role):
                    st.write(text)

            question = clicked_question or st.chat_input(
                "Ställ en fråga om resultatet…"
            )
            if question:
                api_key = st.secrets.get("OPENROUTER_API_KEY")
                st.session_state["chat_history"].append(("user", question))
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    if not api_key:
                        answer = (
                            "Ingen OPENROUTER_API_KEY hittad i .streamlit/secrets.toml."
                        )
                        st.error(answer)
                    else:
                        context = (
                            f"Target: {name}. "
                            f"Features: {feature_names}. "
                            f"Baseline: {baseline_label}. "
                            f"Mätvärden (modell vs baseline): {eval_result.metrics}. "
                            f"Testperiod: {test_df['date'].min()} till "
                            f"{test_df['date'].max()} ({len(test_df)} rader)."
                        )
                        with st.spinner("Tänker…"):
                            try:
                                answer = chatbot.ask(api_key, context, question)
                            except Exception as e:  # noqa: BLE001 - surface any API failure in chat
                                answer = f"Kunde inte nå AI-tjänsten: {e}"
                        st.write(answer)
                st.session_state["chat_history"].append(("assistant", answer))
        else:
            st.info(
                "Klicka 'Kör evaluering' för att träna och utvärdera på test-split."
            )
    else:
        st.info("Välj minst en feature.")
