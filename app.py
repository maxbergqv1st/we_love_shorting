"""View: Streamlit dashboard. Run with `streamlit run app.py`."""

import datetime as dt
import logging
import sys
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent / "src")
)  # ponytail: path shim, drop after `pip install -e .`

import pandas as pd
import streamlit as st

from we_love_shorting import analysis, chatbot, controller, features
from we_love_shorting.sources import yahoo

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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

# live-predictor only: intraday, never persisted, TTL matches the fragment's rerun cadence
get_intraday = st.cache_data(ttl=60)(yahoo.fetch_intraday_price)


def _stems_for(cols: list[str]) -> list[str]:
    """Ticker stems whose `_close` or `_ret` column is among `cols`."""
    wanted = {features.stream_of(c) for c in cols}
    return [s for s in features.TICKERS if s in wanted]


@st.cache_data(ttl=60)
def _fetch_live_closes(stems: tuple[str, ...]) -> dict[str, tuple[float, pd.Timestamp]]:
    """Latest intraday close per ticker stem. A stem whose fetch fails (closed
    market, bad ticker, yfinance hiccup) is left out rather than raised, so one
    bad ticker doesn't blank the whole live row — predict_live then falls back
    to that column's most recent stored value."""
    out = {}
    for stem in stems:
        try:
            row = get_intraday(features.TICKERS[stem])
        except Exception as e:  # noqa: BLE001 - see docstring
            log.warning("live fetch failed for %s: %s", stem, e)
            continue
        out[stem] = (float(row["Close"]), row.name)
    return out


def _live_feature_values(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[dict[str, float], dict[str, pd.Timestamp]]:
    """Live values for the `_close`/`_ret` columns among `feature_cols`, plus
    each contributing stem's quote timestamp. `_ret` is derived against the
    latest stored close (today's move so far), matching the daily `_ret`
    columns in features.build_features. A feature with no live source (e.g.
    `tone`) is simply absent — predict_live carries its stored value forward."""
    stems = _stems_for(feature_cols)
    fetched = _fetch_live_closes(tuple(stems))
    latest = df.iloc[-1]
    values: dict[str, float] = {}
    timestamps: dict[str, pd.Timestamp] = {}
    for stem, (close, ts) in fetched.items():
        if f"{stem}_close" in feature_cols:
            values[f"{stem}_close"] = close
        if f"{stem}_ret" in feature_cols:
            prev_close = latest[f"{stem}_close"]
            values[f"{stem}_ret"] = (close - prev_close) / prev_close
        timestamps[stem] = ts
    return values, timestamps


@st.fragment(run_every="1m")
def live_predictor_panel(
    df: pd.DataFrame, feature_cols: list[str], target: str
) -> None:
    live_values, timestamps = _live_feature_values(df, feature_cols)
    if not live_values:
        st.info("Ingen av de valda featuresen går att hämta live just nu.")
        return
    prediction = controller.predict_live(df, feature_cols, target, live_values)
    newest_local = max(timestamps.values()).to_pydatetime().astimezone()
    today = dt.datetime.now().astimezone().date()
    stale = newest_local.date() != today
    st.metric(
        f"Skattning för IDAG ({today}): {LABELS.get(target, target)}",
        f"{prediction:.4f}",
    )
    st.caption(
        "Detta är ingen prognos för imorgon: modellen är tränad på samma dag "
        "(kurser → värde samma datum), så den skattar vad "
        f"{LABELS.get(target, target)} borde vara *just nu* givet dagens live-kurser."
    )
    st.caption(
        f"Baserad på {len(live_values)}/{len(feature_cols)} valda features hämtade "
        f"live ({', '.join(LABELS.get(c, c) for c in live_values)}) — övriga "
        "features använder senaste lagrade värde. "
        f"Senaste notering: {newest_local:%Y-%m-%d %H:%M:%S} (lokal tid)"
        + (" — marknaden är stängd, visar senaste handelsdata." if stale else "")
    )


def render_panel(panel: analysis.Panel, days: int | None) -> None:
    """Draw one render-agnostic Panel. `days` trims a live time-series line to
    its trailing window; pass None (evaluation panels) to draw it whole."""
    if panel.caption:
        st.caption(panel.caption)
    data = panel.data
    if panel.kind == "text":
        return
    # live time-series panels (line/scatter over date) trim to the window; the
    # rows are chronological so tail() keeps the most recent `days`.
    if days is not None and data is not None and panel.kind in ("line", "scatter"):
        data = data.tail(days)
    if panel.kind == "line":
        st.line_chart(data, color=panel.colors)  # type: ignore[arg-type]  # stub rejects list[str]
    elif panel.kind == "scatter":
        st.scatter_chart(data, x=panel.x, y=panel.y, color=panel.color)
    elif panel.kind == "bar":
        st.bar_chart(data)
    elif panel.kind == "table":
        if panel.gradient and data is not None:
            data = data.style.background_gradient(
                cmap="RdBu", vmin=-1, vmax=1, axis=None
            ).format(precision=2)
        st.dataframe(data, width="stretch")


def render_chatbot(context_extra: str, name: str, feature_names: str) -> None:
    """Grounded AI Q&A about the evaluation. `context_extra` is the mode's own
    summary (metrics, test period); the rest is shared framing."""
    st.subheader("💬 Fråga om resultatet")
    st.caption("AI-assistent grundad i evalueringen ovan (OpenRouter, gratis-modell).")
    quick_questions = [
        "Varför presterar modellen bättre/sämre än baseline?",
        "Var i testperioden är felen som störst?",
        "Är modellen tillförlitlig nog att lita på?",
    ]
    clicked = None
    for col, q in zip(st.columns(len(quick_questions)), quick_questions):
        if col.button(q, use_container_width=True):
            clicked = q
    st.session_state.setdefault("chat_history", [])
    for role, text in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.write(text)
    question = clicked or st.chat_input("Ställ en fråga om resultatet…")
    if question:
        api_key = st.secrets.get("OPENROUTER_API_KEY")
        st.session_state["chat_history"].append(("user", question))
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            if not api_key:
                answer = "Ingen OPENROUTER_API_KEY hittad i .streamlit/secrets.toml."
                st.error(answer)
            else:
                context = f"Target: {name}. Features: {feature_names}. {context_extra}"
                with st.spinner("Tänker…"):
                    try:
                        answer = chatbot.ask(api_key, context, question)
                    except Exception as e:  # noqa: BLE001 - surface API failure in chat
                        answer = f"Kunde inte nå AI-tjänsten: {e}"
                st.write(answer)
        st.session_state["chat_history"].append(("assistant", answer))


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
        # Pick an analysis mode from the registry. Adding a mode in
        # analysis.MODES makes it appear here with no change to this loop.
        mode_key = (
            st.segmented_control("Analys", list(analysis.MODES), default="Regression")
            or "Regression"
        )
        mode = analysis.MODES[mode_key]
        st.info(f"🧠 **{mode.label}** över **{name}** från {feature_names}")

        # `or "Allt"` keeps a selection even if the user deselects the control.
        timespan = (
            st.segmented_control("Visa", list(TIMESPANS), default="Allt") or "Allt"
        )
        days = TIMESPANS[timespan]
        try:
            for panel in mode.live_panels(df, feature_cols, target, name):
                render_panel(panel, days)
        except Exception as e:  # noqa: BLE001 - degenerate fit (e.g. all-flat), etc.
            st.warning(f"Kunde inte rendera live-vyn: {e}")

        # Regression-only extras: latest-value cards, the live 1-min predictor,
        # and the market-closed-highlighted raw table.
        if mode_key == "Regression":
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
            with st.expander(
                "🔴 Live-prediktion (testar 1 min-intervall)", expanded=True
            ):
                live_predictor_panel(df, feature_cols, target)
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
                column_config={"market_closed": None},  # None = hide, Styler reads it
            )
            st.caption(
                "🟥 Röd rad = börsen stängd (helg/helgdag), föregående close används."
            )

        st.subheader("Evaluering (train/test)")
        if st.button("Kör evaluering"):
            try:
                st.session_state["eval_result"] = mode.evaluate(
                    df, feature_cols, target
                )
                st.session_state["eval_mode"] = mode_key
                st.session_state["eval_target"] = target
                st.session_state["eval_feature_cols"] = feature_cols
            except Exception as e:  # noqa: BLE001 - too little data, etc.
                st.session_state.pop("eval_result", None)
                st.warning(f"Kunde inte evaluera: {e}")

        # Only show a stored result for the mode that produced it — switching
        # modes hides a stale result until you re-run in the new mode.
        if (
            "eval_result" in st.session_state
            and st.session_state.get("eval_mode") == mode_key
        ):
            result = st.session_state["eval_result"]
            # compare features as sets: reordering the same picks doesn't
            # change the model, so it shouldn't flag the eval as stale.
            if st.session_state.get("eval_target") != target or set(
                st.session_state.get("eval_feature_cols", [])
            ) != set(feature_cols):
                st.info(
                    "Valen ovan har ändrats sedan senaste evalueringen — "
                    "resultaten nedan gäller föregående val. Klicka 'Kör "
                    "evaluering' för att uppdatera."
                )
            for panel in mode.evaluate_panels(result, name):
                render_panel(panel, days=None)
            context_extra = mode.context(result)
            if context_extra:
                render_chatbot(context_extra, name, feature_names)
        else:
            st.info(f"Klicka 'Kör evaluering' för resultat i läget **{mode.label}**.")

        # Asset grouping: target-independent, so it's its own always-on section
        # (not one of the predict-a-target modes above).
        with st.expander(
            "🔗 Tillgångsgruppering — vilka streams rör sig ihop (oberoende av mål)",
        ):
            try:
                for panel in analysis.asset_grouping_panels(df, feature_cols):
                    render_panel(panel, days=None)
            except ValueError as e:
                st.info(str(e))
    else:
        st.info("Välj minst en feature.")
