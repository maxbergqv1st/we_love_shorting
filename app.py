"""View: Streamlit trading-style workspace. Run with `streamlit run app.py`."""

import datetime as dt
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from we_love_shorting import analysis, chatbot, controller, features, signal_model
from we_love_shorting.sources import yahoo

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

st.set_page_config(
    page_title="we_love_shorting",
    page_icon=":material/query_stats:",
    layout="wide",
)

# Human names per stream stem (features.TICKERS is the source of the stem set, so
# a new ticker always gets label entries — unnamed here, it falls back to the
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
_SUFFIX_LABELS = {
    "close": "{}",
    "ret": "{} (daglig %)",
    "ma5": "{} (5d snitt)",
    "ma21": "{} (21d snitt)",
    "vol21": "{} (21d volatilitet)",
}
LABELS = {"tone": "News tone"}
for _stem in features.TICKERS:
    _name = STREAM_NAMES.get(_stem, _stem)
    for _sfx, _fmt in _SUFFIX_LABELS.items():
        LABELS[f"{_stem}_{_sfx}"] = _fmt.format(_name)

# Forecast horizon: 0 = estimate today's value from today's features (nowcast),
# 1 = train features(t) -> target(t+1), i.e. a real next-day forecast.
HORIZONS = {"Idag (nowcast)": 0, "Imorgon (+1 dag)": 1}

# Chart timespan filter: trading days to show, counting back from the latest row.
TIMESPANS = {"Vecka": 5, "Månad": 21, "År": 252, "Allt": None}


# Tunable hyperparameters. Toggle any subset on (multiselect); each active one
# gets a control and flows into the pipeline as a `params` dict. Off = the
# controller's own default. `key` matches the controller/model argument name so
# it can be forwarded straight through (analysis._kw). Numeric options render a
# slider, string options a selectbox. `default` = the value when toggled on.
@dataclass(frozen=True)
class HyperParam:
    key: str
    label: str
    default: float | str
    options: list  # numeric -> slider; string -> selectbox
    help: str  # shown as a (?)-tooltip on the control


HYPERPARAMS = [
    HyperParam(
        "alpha",
        "Ridge α — regularisering (Regression)",
        0.0,
        [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "Krymper koefficienterna mot 0. 0 = vanlig OLS. Högre värde = enklare "
        "modell som överanpassar mindre men kan missa svag signal. Prova högre "
        "om vikterna i evalueringen ser extrema ut eller test-RMSE >> träning.",
    ),
    HyperParam(
        "c",
        "Logistisk reg. C — lägre = hårdare (Riktning)",
        1.0,
        [0.01, 0.1, 1.0, 10.0, 100.0],
        "Omvänd regularisering för riktningsklassificeraren: lägre C = hårdare "
        "krympning (enklare modell, mindre överanpassning), högre C = friare. "
        "Sänk om träffsäkerheten på test ligger långt under träningens.",
    ),
    HyperParam(
        "flat_frac",
        "Dödzon — upp/oförändrad/ner (Riktning)",
        0.25,
        [0.0, 0.1, 0.25, 0.5, 1.0],
        "Bredden på 'oförändrad'-klassen: rörelser inom ±(flat_frac × "
        "träningens std) räknas som flata. 0 = ren upp/ner-klassning; större "
        "värde = fler dagar klassas som oförändrade. Skalas med varje streams "
        "egen volatilitet.",
    ),
    HyperParam(
        "test_frac",
        "Test-andel — train/test-split (Evaluering)",
        0.2,
        [0.1, 0.2, 0.3, 0.4],
        "Andel av historiken (den senaste, kronologiskt) som hålls undan som "
        "test. Större = stabilare mätning men mindre träningsdata. Splitten är "
        "alltid kronologisk — testet är en äkta framtid för modellen.",
    ),
    HyperParam(
        "n_groups",
        "Antal grupper (Gruppering)",
        3,
        [2, 3, 4, 5, 6],
        "Hur många grupper tillgångsklustringen delar in streamsen i. Fler "
        "grupper = finare uppdelning; jämför med heatmapens block för att se "
        "vad som är naturligt.",
    ),
    HyperParam(
        "linkage",
        "Länkning (Gruppering)",
        "average",
        ["average", "complete", "single"],
        "Hur avståndet mellan två kluster mäts när de slås ihop: average = "
        "snittavstånd (robust standard), complete = längsta avståndet (kompakta "
        "grupper), single = kortaste (kan kedja ihop olika tillgångar).",
    ),
    HyperParam(
        "n_estimators",
        "Antal träd (Random Forest)",
        100,
        [50, 100, 200, 400],
        "Fler träd = stabilare prediktioner men långsammare träning; nyttan "
        "avtar snabbt över ett par hundra. Sänk för snabbhet, höj om "
        "resultaten hoppar mellan körningar.",
    ),
    HyperParam(
        "max_depth",
        "Max träddjup (Random Forest)",
        6,
        [2, 4, 6, 8, 12],
        "Hur djupt varje träd får växa: djupare = mer komplexa mönster men "
        "större överanpassningsrisk på brusiga finansdata. Håll lågt (4–6) om "
        "test-metrics är mycket sämre än träningens.",
    ),
]
HP_BY_KEY = {h.key: h for h in HYPERPARAMS}

# reads the DB (no fetch); cleared after a top-up, 1h TTL bounds CLI-fill staleness
get_data = st.cache_data(ttl="1h")(controller.get_data)

# live-predictor: per-ticker intraday cache so toggling one stream pill only
# refetches the changed ticker; TTL matches the fragment cadence
get_intraday = st.cache_data(ttl=60)(yahoo.fetch_intraday_price)

# one fitted model per (frame, features, target, knobs): the minute-refresh and
# full-page reruns reuse it instead of refitting (in-memory only, never on disk)
train_model = st.cache_resource(show_spinner=False)(signal_model.train)

# session-only streams: free-text ticker search + a 5y history fetch at add-time
search_tickers = st.cache_data(ttl="1h")(yahoo.search_tickers)
fetch_5y = st.cache_data(ttl="1h", show_spinner="Hämtar kurshistorik…")(
    yahoo.fetch_prices
)


@st.cache_data(show_spinner=False)
def cached_eval(
    mode_key: str,
    df: pd.DataFrame,
    feature_cols: tuple[str, ...],
    spec_key: tuple,
    params_items: tuple,
):
    """Held-out evaluation, memoised so it auto-runs on selection without a
    button and only recomputes when mode/features/target/params change.
    `spec_key` is `astuple(spec)` and `params_items` is
    `tuple(sorted(params.items()))` so both are hashable cache keys."""
    return analysis.MODES[mode_key].evaluate(
        df, list(feature_cols), controller.TargetSpec(*spec_key), dict(params_items)
    )


@st.cache_data(show_spinner=False)
def cached_live(
    mode_key: str,
    df: pd.DataFrame,
    feature_cols: tuple[str, ...],
    spec_key: tuple,
    label: str,
    params_items: tuple,
):
    """The mode's always-visible panels, memoised like cached_eval — otherwise
    the full in-sample fit reruns on every widget interaction on the page."""
    return analysis.MODES[mode_key].live_panels(
        df,
        list(feature_cols),
        controller.TargetSpec(*spec_key),
        label,
        dict(params_items),
    )


def _fetch_live_closes(stems: list[str]) -> dict[str, tuple[float, pd.Timestamp]]:
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
    latest stored close (today's move so far), matching the daily `_ret` columns
    in features.build_features. A feature with no live source (e.g. `tone`) is
    simply absent — predict_live carries its stored value forward."""
    wanted = {features.stream_of(c) for c in feature_cols}
    fetched = _fetch_live_closes([s for s in features.TICKERS if s in wanted])
    # the _ret base must be a PRIOR day's close: after an intraday top-up the
    # last stored row can be today's own, which would collapse live _ret to ~0
    today = dt.datetime.now().astimezone().date()
    past = df[pd.to_datetime(df["date"]).dt.date < today]
    latest = (past if len(past) else df).iloc[-1]
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
    df_live: pd.DataFrame,
    df_train: pd.DataFrame,
    feature_cols: list[str],
    spec: controller.TargetSpec,
    label: str,
    params: dict[str, Any],
) -> None:
    """Live estimate, self-refreshing every minute (a fragment, so only this
    block reruns). Trains on `df_train` (the prepared frame: the target's move,
    horizon-shifted) but derives live feature values AND non-live fallbacks
    from `df_live` (the raw frame) — the shifted frame's last rows are dropped,
    so its tail is a trading day stale. A level target shows the reconstructed
    level with the predicted move as delta. The refresh + timestamp make it
    visibly live."""
    live_values, timestamps = _live_feature_values(df_live, feature_cols)
    now = dt.datetime.now().astimezone()
    if not live_values:
        st.caption(":red-badge[● LIVE] Ingen vald feature går att hämta live just nu.")
        return
    model_kw = {k: params[k] for k in analysis.MODEL_KEYS if k in params}
    model = train_model(df_train, feature_cols, spec.train_col, **model_kw)
    prediction = controller.predict_live(
        df_train,
        feature_cols,
        spec.train_col,
        live_values,
        model=model,
        live_df=df_live,
    )
    newest_local = max(timestamps.values()).to_pydatetime().astimezone()
    stale = newest_local.date() != now.date()
    when = "imorgon" if spec.horizon else "idag"
    heading = f":red-badge[● LIVE] Prognos {when} · {label}"
    if spec.kind == "identity":
        shown = prediction
        st.metric(heading, f"{prediction:.4f}")
    else:
        # base = the level the move builds on. horizon>=1: latest stored level.
        # nowcast: the PREVIOUS day's level — level_base backs it out per row,
        # but needs the move column, which only the prepared frame carries; at
        # horizon 0 that frame's tail equals the raw frame's, so it's current.
        frame = df_live if spec.horizon >= 1 else df_train
        base = float(controller.level_base(frame, spec).iloc[-1])
        shown = controller.reconstruct_level(base, prediction, spec)
        delta = f"{prediction:+.2%}" if spec.kind == "ret" else f"{prediction:+.4f}"
        st.metric(heading, f"{shown:.2f}", delta=delta)
    # the forecast in context: recent actuals vs the model's prediction PER
    # DAY (in-sample), continuing into the live forecast point. A separate
    # per-day line makes the model's misses visible as the gap between the
    # curves — a single anchored segment just melts into the actual line.
    days = 15  # short window: over a month the daily gaps compress to nothing
    hist = df_live.set_index("date")[spec.level_col or spec.train_col].tail(days)
    hist.index = pd.to_datetime(hist.index)
    tail = df_train.tail(days)
    day_preds = signal_model.predict(tail, model, feature_cols)
    if spec.kind != "identity":
        day_preds = controller.reconstruct_level(
            controller.level_base(tail, spec), day_preds, spec
        )
    # each row's prediction targets t+horizon; +horizon calendar days is a
    # ponytail: approximation (a weekend shifts a forecast label one day)
    dates = pd.to_datetime(tail["date"]) + pd.Timedelta(days=spec.horizon)
    pred_line = pd.Series(day_preds.to_numpy(), index=pd.DatetimeIndex(dates))
    pred_line[hist.index[-1] + pd.Timedelta(days=1)] = shown  # the live point
    chart = pd.DataFrame({label: hist, "Prognos": pred_line})
    chart.index.name = "date"
    line_chart(chart, colors=analysis.SERIES_COLORS)
    st.caption(
        f"{len(live_values)}/{len(feature_cols)} features live "
        f"({', '.join(LABELS.get(c, c) for c in live_values)}). "
        f"Uppdaterad {now:%H:%M:%S}, senaste kurs {newest_local:%Y-%m-%d %H:%M}"
        + (" · marknaden stängd" if stale else "")
        + ". Grafen: blå = faktiskt värde, orange = modellens prediktion per "
        "dag (in-sample; glappet mot blå = dagens miss); sista orange punkten "
        "= live-prognosen, ritad ett steg efter sista lagrade dagen."
    )


@contextmanager
def card(title: str):
    """A bordered section with a heading — the workspace's repeating unit."""
    with st.container(border=True):
        st.markdown(f"#### {title}")
        yield


def seg(label: str, options: list[str], default: str, **kw: Any) -> str:
    """segmented_control that can't be deselected: clicking off -> the default."""
    return st.segmented_control(label, options, default=default, **kw) or default


def _corr_cell(v: float) -> str:
    """Green for positive correlation, red for negative, alpha by magnitude —
    a −1..1 heatmap without pulling in matplotlib (Styler.background_gradient
    needs it; this element-wise map doesn't)."""
    if pd.isna(v):
        return ""
    rgb = "152, 195, 121" if v >= 0 else "224, 108, 117"  # theme green / red
    return f"background-color: rgba({rgb}, {abs(v):.2f})"


def line_chart(data: pd.DataFrame, colors: list[str] | None = None) -> None:
    """Date-indexed multi-series line via Altair WITHOUT pan/zoom, so scrolling
    the page over the chart doesn't hijack the mouse wheel (native st.line_chart
    zooms on scroll — the reported lag/'scroll away'). Tooltips still show every
    series' value at the hovered date."""
    x = data.index.name or "index"
    long = data.reset_index().melt(x, var_name="Serie", value_name="Värde")
    color = alt.Color("Serie:N", legend=alt.Legend(title=None))
    if colors:
        color = color.scale(range=colors)
    chart = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X(f"{x}:T", title=None),
            # zero=False: without it a price level ~6000 with ±1% weekly moves
            # renders as a flat line pinned to the top of a 0-anchored axis —
            # the axis must follow the visible window, not the origin
            y=alt.Y("Värde:Q", title=None, scale=alt.Scale(zero=False)),
            color=color,
            tooltip=[
                alt.Tooltip(f"{x}:T"),
                "Serie",
                alt.Tooltip("Värde:Q", format=".4f"),
            ],
        )
    )
    st.altair_chart(chart, width="stretch")


def render_panel(panel: analysis.Panel, days: int | None) -> None:
    """Draw one render-agnostic Panel. `days` trims a live time-series line to
    its trailing window; pass None (evaluation panels) to draw it whole."""
    if panel.caption:
        st.caption(panel.caption)
    data = panel.data
    if panel.kind == "line":
        # rows are chronological, so tail() keeps the most recent `days`
        if days is not None and data is not None:
            data = data.tail(days)
        line_chart(data, panel.colors)
    elif panel.kind == "bar":
        st.bar_chart(data, horizontal=panel.horizontal)
    elif panel.kind == "table":
        if panel.gradient and data is not None:
            data = data.style.map(_corr_cell).format(precision=2)
        st.dataframe(data, width="stretch")


def render_chat(context: str) -> None:
    """Always-available AI assistant, grounded in the current workspace state —
    not tied to having run an evaluation."""
    st.header("Assistent", icon=":material/smart_toy:")
    st.caption("Frågor om modellen, datan eller resultaten.")
    quick = [
        "Slår modellen sin baseline?",
        "Var är felen störst?",
        "Går riktningen att lita på?",
    ]
    clicked = None
    for q in quick:
        if st.button(q, key=f"q_{q}", width="stretch"):
            clicked = q
    st.session_state.setdefault("chat_history", [])
    # tall scrollable history so the chat fills the lower half of the sidebar
    with st.container(height=340, border=False):
        for role, text in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(text)
    question = clicked or st.chat_input("Fråga assistenten…")
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
            retry, more = controller.fetch_all(retry)
            errors |= more  # a retry can still surface a non-rate-limit error
    box.empty()
    get_data.clear()  # DB changed -> reload on next "Ladda data"
    if errors:
        st.error("Fel: " + "; ".join(f"{t}: {m}" for t, m in errors.items()))
    if retry:
        st.warning(f"Fortfarande rate-limitad: {', '.join(retry)}. Försök igen strax.")
    if not errors and not retry:
        st.success("Klart. Klicka 'Ladda data'.", icon=":material/check_circle:")


@st.dialog("Datakällor & påfyllning", width="large")
def setup_dialog() -> None:
    """Rare admin actions (backfill / top-up / load), out of the main flow."""
    query = st.text_input("GDELT-fråga (nyhetston)", "recession", key="gdelt_query")
    if st.button(
        "Första fyllning (~5 år)", icon=":material/download:", width="stretch"
    ):
        fill("Hämtar ~5 års historik…", lambda: controller.backfill(query))
    if st.button("Fyll på till idag", icon=":material/update:", width="stretch"):
        fill("Hämtar till idag…", lambda: controller.update(query))
    if st.button(
        "Ladda data", icon=":material/database:", type="primary", width="stretch"
    ):
        try:
            st.session_state["df"] = get_data()
            st.rerun()
        except Exception as e:  # noqa: BLE001 - empty/mismatched DB -> guide the user
            st.error(f"Kunde inte bygga feature-tabellen: {e}. Kör 'Första fyllning'.")


# ── Topbar ───────────────────────────────────────────────────────────────────
df = st.session_state.get("df")
top = st.columns([6, 4, 2], vertical_alignment="center")
top[0].markdown("### :material/query_stats: we_love_shorting")
# filled in after the extra-streams weave: the woven table can be shorter than
# the base one, and the header must describe the table the cards actually use
db_stats = top[1].empty()
if top[2].button("Setup", icon=":material/settings:", width="stretch"):
    setup_dialog()

if df is None:
    st.info(
        "Öppna **Setup** för att fylla och ladda data.", icon=":material/rocket_launch:"
    )
    st.stop()

# ── Sidebar: add arbitrary streams (session-only, never written to the DB) ───
st.session_state.setdefault("extra_streams", {})  # stem -> {"name", "prices"}
with st.sidebar, st.expander("Lägg till stream", icon=":material/search:"):
    q = st.text_input("Sök aktie (namn eller ticker)", key="stream_query")
    if q:
        try:
            hits = search_tickers(q)
        except Exception as e:  # noqa: BLE001 - search endpoint hiccup
            hits = []
            st.caption(f"Sökningen misslyckades: {e}")
        if hits:
            pick = st.selectbox(
                "Träffar",
                hits,
                format_func=lambda h: f"{h['name']} ({h['symbol']}, {h['exchange']})",
            )
            if st.button("Lägg till", icon=":material/add:", width="stretch"):
                stem = re.sub(r"[^a-z0-9]", "", pick["symbol"].lower())
                if (
                    stem in features.TICKERS
                    or stem in ("tone", "")
                    or stem in st.session_state["extra_streams"]
                ):
                    st.error("Namnet krockar med en befintlig stream.")
                else:
                    try:
                        prices = fetch_5y(pick["symbol"], "5y", f"{stem}_close")
                        st.session_state["extra_streams"][stem] = {
                            "name": pick["name"],
                            "prices": prices,
                        }
                        st.rerun()
                    except Exception as e:  # noqa: BLE001 - bad/empty ticker
                        st.error(f"Kunde inte hämta {pick['symbol']}: {e}")
        elif q:
            st.caption("Inga träffar.")
    if st.session_state["extra_streams"]:
        names = ", ".join(v["name"] for v in st.session_state["extra_streams"].values())
        st.caption(
            f"Tillagda: {names}. Sessionens tabell begränsas till den "
            "kortaste tillagda historiken."
        )
        if st.button("Rensa tillagda", icon=":material/delete:", width="stretch"):
            st.session_state["extra_streams"] = {}
            st.rerun()

if err := st.session_state.pop("stream_error", None):
    st.error(f"Kunde inte väva in tillagda streams: {err}")
extra = st.session_state["extra_streams"]
if extra:
    try:
        df = get_data({stem: v["prices"] for stem, v in extra.items()})
    except Exception as e:  # noqa: BLE001 - no overlap etc. -> drop the extras
        st.session_state["extra_streams"] = {}
        st.session_state["stream_error"] = str(e)
        st.rerun()  # sidebar/labels already rendered the extras — restart clean
    for stem, v in extra.items():
        for _sfx, _fmt in _SUFFIX_LABELS.items():
            LABELS[f"{stem}_{_sfx}"] = _fmt.format(v["name"])
db_stats.markdown(
    f":material/database: {df['date'].min()} → {df['date'].max()} · {len(df)} rader"
)

# ── Sidebar: settings (top) ──────────────────────────────────────────────────
candidates = [c for c in df.select_dtypes("number").columns if c != "market_closed"]

with st.sidebar:
    st.subheader("Inställningar", icon=":material/tune:")
    horizon_key = seg("Horisont", list(HORIZONS), "Idag (nowcast)")
    horizon = HORIZONS[horizon_key]
    mode_key = seg("Läge", list(analysis.MODES), "Regression")
    model_kind = "linear"
    if mode_key == "Regression" and (
        seg("Modell", ["Linjär", "Random Forest"], "Linjär", key="model_pick")
        == "Random Forest"
    ):
        model_kind = "forest"

    # Hyperparametrar: toggla på valfri delmängd (flera samtidigt), styr var och
    # en. Av = standardvärde. Bara aktiva hamnar i `params`.
    st.caption(":material/tune: Hyperparametrar")
    active = st.multiselect(
        "Toggla på för att tuna",
        [h.key for h in HYPERPARAMS],
        format_func=lambda k: HP_BY_KEY[k].label,
        label_visibility="collapsed",
        key="active_hp",
        help="Av = modellens standardvärde används. Toggla på en knapp för att "
        "styra den själv; hovra över (?) vid varje reglage för vad den gör.",
    )
    params: dict[str, Any] = {}
    for key in active:
        hp = HP_BY_KEY[key]
        if all(isinstance(o, int | float) for o in hp.options):
            params[key] = st.select_slider(
                hp.label, hp.options, value=hp.default, key=f"hp_{key}", help=hp.help
            )
        else:
            params[key] = st.selectbox(
                hp.label,
                hp.options,
                index=hp.options.index(hp.default),
                key=f"hp_{key}",
                help=hp.help,
            )
    if model_kind != "linear":
        params["model_kind"] = model_kind  # linear is the default downstream

# ── Selection (above the chart): one target, feature STREAMS as toggles ─────
# Toggling a stream includes all its columns (close/ret/ma/vol) as features —
# the picker works in streams, not in 5 near-identical chips per asset.
target_opts = [c for c in candidates if c == "tone" or c.endswith(("_close", "_ret"))]
default_target = features.TARGET if features.TARGET in target_opts else target_opts[0]


def stream_label(stem: str) -> str:
    if stem == "tone":
        return "News tone"
    if stem in st.session_state["extra_streams"]:
        return st.session_state["extra_streams"][stem]["name"]
    return STREAM_NAMES.get(stem, stem)


with st.container(border=True):
    sel = st.columns([4, 8], vertical_alignment="center")
    target = sel[0].selectbox(
        "Mål — vad ska förutsägas?",
        target_opts,
        index=target_opts.index(default_target),
        format_func=lambda c: LABELS.get(c, c),
        key="target",
    )
    # every stream in the table except the target's own (a stream must not
    # predict itself); per-target key keeps each target's toggles separate.
    target_stream = features.stream_of(target)
    all_streams = list(dict.fromkeys(features.stream_of(c) for c in candidates))
    stream_opts = [s for s in all_streams if s != target_stream]
    picked_streams = sel[1].pills(
        "Features — streams (alla kolumner för en vald stream räknas med)",
        stream_opts,
        selection_mode="multi",
        default=stream_opts,
        key=f"streams::{target}",
        format_func=stream_label,
    )

feature_cols = [c for c in candidates if features.stream_of(c) in picked_streams]
if not feature_cols:
    st.warning("Toggla på minst en stream ovan.", icon=":material/warning:")
    st.stop()

name = LABELS.get(target, target)
feature_names = ", ".join(stream_label(s) for s in picked_streams)
mode = analysis.MODES[mode_key]
# The prepared frame: a level target is re-targeted to its MOVE (spec.train_col
# — see controller.prepare_target), then row t's move becomes t+horizon's
# actual (no-op for nowcast). Everything model-related below uses df_h + spec;
# the compare/grouping cards keep the raw df — they describe the data, not the
# prediction task.
df_h, spec = controller.prepare_target(df, target, horizon)

# ── Live estimate (hero) ─────────────────────────────────────────────────────
with st.container(border=True):
    live_predictor_panel(df, df_h, feature_cols, spec, name, params)

# ── Main view (large focus card) ─────────────────────────────────────────────
with st.container(border=True):
    head = st.columns([6, 6], vertical_alignment="center")
    head[0].markdown(f"#### {mode.label} · {name} · {horizon_key}")
    with head[1].container(horizontal=True, horizontal_alignment="right"):
        timespan = seg("Visa", list(TIMESPANS), "Allt", label_visibility="collapsed")
    days = TIMESPANS[timespan]
    try:
        panels = cached_live(
            mode_key,
            df_h,
            tuple(feature_cols),
            astuple(spec),
            name,
            tuple(sorted(params.items())),
        )
        for panel in panels:
            render_panel(panel, days)
    except Exception as e:  # noqa: BLE001 - degenerate fit (e.g. all-flat), etc.
        st.warning(f"Kunde inte rendera vyn: {e}")

# date-indexed view (trimmed to the timespan) shared by the charts below
dated = df.set_index("date")
dated = dated if days is None else dated.tail(days)


def _indexed(cols: pd.DataFrame) -> pd.DataFrame:
    """Index each column to 100 at its first valid value, so different price
    levels compare on one axis and a late-starting stream doesn't vanish."""
    return cols / cols.bfill().iloc[0] * 100


# ── Compare (near the prediction): the streams, indexed onto one axis ─────────
with card("Jämför · streams"):
    close_cols = [c for c in df.columns if c.endswith("_close")]
    picked = st.pills(
        "Streams",
        close_cols,
        selection_mode="multi",
        format_func=lambda c: LABELS.get(c, c),
        default=close_cols[:3],
        label_visibility="collapsed",
    )
    if picked:
        indexed = _indexed(dated[picked]).rename(columns=lambda c: LABELS.get(c, c))
        line_chart(indexed)
        st.caption("Indexerat till 100 vid start — klicka i/ur streams.")
    else:
        st.caption("Välj en eller flera streams.")

# ── Evaluation (auto-run, cached) ────────────────────────────────────────────
with card("Evaluering · held-out test"):
    try:
        with st.spinner("Evaluerar…"):
            result = cached_eval(
                mode_key,
                df_h,
                tuple(feature_cols),
                astuple(spec),
                tuple(sorted(params.items())),
            )
        for panel in mode.evaluate_panels(result, name):
            render_panel(panel, None)
        chat_context = mode.context(result)
    except Exception as e:  # noqa: BLE001 - too little data after filtering, etc.
        st.warning(f"Kunde inte evaluera: {e}")
        chat_context = ""

# ── Asset grouping ───────────────────────────────────────────────────────────
with card("Tillgångsgruppering · vilka streams rör sig ihop"):
    try:
        for panel in analysis.asset_grouping_panels(df, feature_cols, params):
            render_panel(panel, None)
    except ValueError as e:
        st.caption(str(e))

# ── Sidebar: AI assistant (lower half, always available) ─────────────────────
with st.sidebar:
    st.divider()
    render_chat(
        f"Data {df['date'].min()}–{df['date'].max()} ({len(df)} rader). "
        f"Läge: {mode.label}. Horisont: {horizon_key}. Mål: {name}. "
        f"Features: {feature_names}. "
        f"Evaluering: {chat_context}"
    )
