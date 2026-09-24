"""Controller: wire the separate sources -> DB -> features -> ML -> results."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from . import db, evaluation, features, signal_model
from .sources import gdelt, yahoo

log = logging.getLogger(__name__)


def fetch_all(specs: dict) -> tuple[dict, dict]:
    """Run each {table: fetch-callable}, upserting successes independently.

    Every source commits on its own, so one source's failure doesn't discard the
    others. Returns two dicts:
      retry  = {table: fetch-callable} that hit a rate-limit (HTTP 429) — worth
               retrying after a cooldown.
      errors = {table: message} that failed for any other reason (bad ticker,
               empty backfill, …) — retrying won't help, surface immediately.
    An incremental fetch that's already current returns an empty frame, which
    db.upsert no-ops on, so "nothing new" is neither a retry nor an error.
    """
    retry, errors = {}, {}
    for table, fetch in specs.items():
        try:
            db.upsert(table, fetch())
        except Exception as e:  # noqa: BLE001 - keep going; classify below
            log.warning("fetch for %s failed: %s", table, e)
            if getattr(e, "code", None) == 429:  # urllib.error.HTTPError rate-limit
                retry[table] = fetch
            else:
                errors[table] = str(e)
    return retry, errors


def _sources(query: str, *, incremental: bool) -> dict:
    """Every stream keyed by its DB table -> a fetch-callable. Tone (GDELT) sits
    beside the price streams (Yahoo); only its data source differs. `incremental`
    fetches from each table's last stored date; otherwise a full ~5y / rolling
    backfill.
    """
    src: dict[str, Callable[..., pd.DataFrame]] = {}
    for stem, ticker in features.TICKERS.items():
        col = f"{stem}_close"
        if incremental:
            src[stem] = lambda t=ticker, c=col, s=stem: yahoo.fetch_prices(
                t, value_col=c, start=db.last_date(s)
            )
        else:
            src[stem] = lambda t=ticker, c=col: yahoo.fetch_prices(t, "5y", c)
    if incremental:
        src["tone"] = lambda: gdelt.fetch_tone(query, start=db.last_date("tone"))
    else:
        src["tone"] = lambda: gdelt.fetch_tone(query, timespan="60m")
    return src


def backfill(query: str = "recession") -> tuple[dict, dict]:
    """First fill: pull ~5 years of history for every stream into the DB. Tickers
    are fixed in features.TICKERS (the UI no longer asks). GDELT tone only covers
    a rolling window (~2017 on), so its table may start later than the price
    history. Run once, then keep it current with update().
    Returns fetch_all's (retry, errors) pair.
    """
    return fetch_all(_sources(query, incremental=False))


def update(query: str = "recession") -> tuple[dict, dict]:
    """Top up each table from its newest stored date to today (no full refetch).
    An empty table (never backfilled) falls back to the default fetch window.
    Returns fetch_all's (retry, errors) pair.
    """
    return fetch_all(_sources(query, incremental=True))


def get_data(extra_prices: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Read the stored data and join into the wide feature table (no fetch, no
    model). Fill the DB first via backfill()/update() — the DB is the source of
    truth. Kept separate from run() so the UI can cache this once and re-train on
    different feature/target picks without touching the data sources.

    `extra_prices` maps extra stem -> price frame (columns date, <stem>_close)
    for session-only streams the user searched up in the UI. They join the
    pipeline like any fixed stream but are never written to the DB — the
    table-name whitelist stays intact. NOTE: build_features drops rows any
    stream lacks, so an extra stream with a short history truncates the whole
    table to its own span.
    """
    prices = {stem: db.load(stem) for stem in features.TICKERS}
    return features.build_features(db.load("tone"), prices | (extra_prices or {}))


def shift_target(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Turn the nowcast frame into a forecast frame: row t keeps its features
    but its `target` value becomes the actual from t+`horizon` TRADING days, so
    a model trained on it predicts `horizon` days ahead. The last `horizon`
    rows (whose future value doesn't exist yet) are dropped. horizon=0 is the
    nowcast — returned as-is. Shift BEFORE any train/test split so nothing
    leaks.

    The shift runs on the market-open rows only: the raw frame has one row per
    tone date (daily, incl. weekends) where closed-day prices are carried
    forward, so a calendar-day shift would hand every Friday row Saturday's
    carried close — its own value — teaching the model a fake "tomorrow equals
    today" on ~1 row in 5. Forecasting is therefore defined over trading days.

    Every downstream piece then works unchanged: the persistence baseline
    becomes "predict tomorrow with today's actual" (the correct naive
    forecast), and direction classifies tomorrow's move.
    """
    if horizon == 0:
        return df
    out = df
    if "market_closed" in out.columns:
        out = out[~out["market_closed"]]
    out = out.copy().reset_index(drop=True)
    out[target] = out[target].shift(-horizon)
    return out.dropna(subset=[target]).reset_index(drop=True)


@dataclass(frozen=True)
class TargetSpec:
    """How the picked target is trained and displayed. The model always trains
    on `train_col` — the target's MOVE. A price level is a near-random-walk:
    trained on the level, the model mostly re-learns persistence ("tomorrow ≈
    today") and the actual signal (the move) drowns. `level_col` is the
    original level column, kept for reconstructing a level view; None when the
    pick already is a move. `kind` picks the arithmetic: "ret" multiplicative,
    "diff" additive, "identity" no reconstruction.
    """

    train_col: str
    level_col: str | None
    kind: str  # "ret" | "diff" | "identity"
    horizon: int

    def format_move(self, x: float) -> str:
        """Human formatting for a move of this kind — beside the kind's
        arithmetic so view code never enumerates kinds itself."""
        return f"{x:+.2%}" if self.kind == "ret" else f"{x:+.4f}"


def prepare_target(
    df: pd.DataFrame, target: str, horizon: int = 0
) -> tuple[pd.DataFrame, TargetSpec]:
    """Re-target a level pick to its move, then horizon-shift the MOVE column.

    Order matters: the re-target must happen BEFORE the shift — shifting the
    level column and re-targeting afterwards would leave the move column at
    nowcast alignment while the user asked for a forecast.

    `X_close` uses the stream's existing `X_ret` column; a level without a ret
    sibling (tone) gets a computed `<target>_diff` column (absolute day-over-day
    change). A `_ret` pick passes through untouched (identity).
    """
    if target.endswith("_ret"):
        spec = TargetSpec(target, None, "identity", horizon)
    elif target.endswith("_close"):
        spec = TargetSpec(f"{features.stream_of(target)}_ret", target, "ret", horizon)
    else:
        move_col = f"{target}_diff"
        if horizon and "market_closed" in df.columns:
            # the diff must span the SAME trading-day steps the shift uses:
            # diffing the full daily calendar first would hand a Friday row
            # Monday−Sunday as its shifted move (tone keeps changing on closed
            # days; prices don't — their ffill makes the ret kind immune)
            df = df[~df["market_closed"]].reset_index(drop=True)
        df = df.assign(**{move_col: df[target].diff()})
        spec = TargetSpec(move_col, target, "diff", horizon)
    out = shift_target(df, spec.train_col, horizon)
    # a computed diff has no move on its first row — NaN would poison the fit
    out = out.dropna(subset=[spec.train_col]).reset_index(drop=True)
    return out, spec


def level_base(frame: pd.DataFrame, spec: TargetSpec) -> pd.Series:
    """The known level each prediction builds on. For a forecast (horizon ≥ 1)
    the row's own level column IS the previous trading day's level — the shift
    moved only the move column. For a nowcast the previous level is backed out
    of the row itself (level ⊖ its actual move): exact per row, no calendar
    alignment needed."""
    level = frame[spec.level_col]
    if spec.horizon >= 1:
        return level
    move = frame[spec.train_col]
    # ponytail: on carried-forward (market-closed) rows the ffill'd move makes
    # this back-out drift from the true previous level — cosmetic in the
    # in-sample chart, and evaluation drops those rows anyway.
    return level / (1 + move) if spec.kind == "ret" else level - move


def reconstruct_level(base, moves, spec: TargetSpec):
    """Predicted level from a predicted move: multiplicative for returns,
    additive for diffs. Works elementwise on Series and on scalars (the live
    panel)."""
    return base * (1 + moves) if spec.kind == "ret" else base + moves


def add_level_view(result: "EvaluationResult", spec: TargetSpec) -> "EvaluationResult":
    """Attach the level-scale view to a move-trained evaluation: reconstructed
    level prediction, persistence baseline and level-scale metrics on test_df.

    Persistence on the level scale IS "predict zero move", so its prediction is
    the base itself — the model beats persistence on the level exactly when the
    move model beats always-predict-0 on the move.
    """
    test = result.test_df
    base = level_base(test, spec)
    # the actual level AT THE TARGET TIME (t+horizon), rebuilt from the actual
    # move — NOT the row's level column, which at horizon>=1 is feature-time
    # (t) and identical to the base, which would score persistence a fake 0
    actual = reconstruct_level(base, test[spec.train_col], spec)
    pred = reconstruct_level(base, test[f"predicted_{spec.train_col}"], spec)
    test["actual_level"] = actual
    test["predicted_level"] = pred
    test["baseline_level"] = base
    result.level_metrics = {
        "model": evaluation.regression_metrics(actual, pred),
        "baseline": evaluation.regression_metrics(actual, base),
    }
    return result


def run(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    alpha: float = 0.0,
    model_kind: str = "linear",
    n_estimators: int = 100,
    max_depth: int = 6,
) -> pd.DataFrame:
    """Train on the chosen features/target and add a `predicted_{target}` column.
    Model choice and knobs are forwarded to signal_model.train.
    """
    model = signal_model.train(
        df, feature_cols, target, alpha, model_kind, n_estimators, max_depth
    )
    df[f"predicted_{target}"] = signal_model.predict(df, model, feature_cols)
    return df


def predict_live(
    model: signal_model.Regressor,
    df: pd.DataFrame,
    feature_cols: list[str],
    live_values: dict[str, float],
) -> float:
    """Predict one point from a fitted model, using live-fetched feature
    values where available. A feature missing from `live_values` (no live
    source, e.g. `tone`, or a ticker whose live fetch failed) falls back to
    `df`'s most recent stored value — the same carry-forward idea as a
    market-closed day in features.build_features. Pass the RAW frame as
    `df`: with a forecast horizon the training frame's tail rows are
    dropped, so its last row is a trading day stale.
    """
    latest = df.iloc[-1]
    row = {c: live_values.get(c, latest[c]) for c in feature_cols}
    return float(signal_model.predict(pd.DataFrame([row]), model, feature_cols).iloc[0])


@dataclass
class EvaluationResult:
    """Chronological train/test evaluation output: the test set carries the
    model's predictions and the naive baseline's side by side (plus each
    one's residual), `metrics` holds both sides' RMSE/MAE, and `weights` is
    what the fitted model leans on per feature (see signal_model.weights).
    """

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target: str  # what was regressed; predictions live in f"predicted_{target}"
    baseline_kind: str
    metrics: dict[str, dict[str, float]]
    weights: pd.Series
    # level-scale view when the target was re-targeted to its move
    # (see add_level_view); None for a plain `_ret` target
    level_metrics: dict[str, dict[str, float]] | None = None


def evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    test_frac: float = 0.2,
    alpha: float = 0.0,
    model_kind: str = "linear",
    n_estimators: int = 100,
    max_depth: int = 6,
) -> EvaluationResult:
    """Chronologically split `df`, train on the train split only, and compare
    the model's held-out test predictions against a naive baseline.

    Market-closed (carry-forward) rows are dropped before the split — see
    evaluation.drop_market_closed — since ffill duplicates `_close`/`_ret`
    across closed days while `tone` keeps changing, which would otherwise
    teach the model an artificial repeated relationship. This is separate
    from run(), which trains on every row for the live chart.
    """
    clean = evaluation.drop_market_closed(df)
    train_df, test_df = evaluation.chronological_split(clean, test_frac)

    model = signal_model.train(
        train_df, feature_cols, target, alpha, model_kind, n_estimators, max_depth
    )
    predicted = signal_model.predict(test_df, model, feature_cols)
    baseline = evaluation.naive_baseline(train_df[target], test_df[target])

    test_df[f"predicted_{target}"] = predicted
    test_df[f"baseline_{target}"] = baseline
    test_df["residual"] = evaluation.residuals(test_df[target], predicted)
    test_df["baseline_residual"] = evaluation.residuals(test_df[target], baseline)

    metrics = {
        "model": evaluation.regression_metrics(test_df[target], predicted),
        "baseline": evaluation.regression_metrics(test_df[target], baseline),
    }
    return EvaluationResult(
        train_df,
        test_df,
        target,
        evaluation.baseline_kind(target),
        metrics,
        signal_model.weights(model, feature_cols),
    )


@dataclass
class DirectionResult:
    """Direction-classification evaluation: test_df carries the realised and
    predicted direction classes (1/0/-1), `metrics` holds the model's and the
    majority baseline's accuracy, and `threshold` is the dead-zone half-width
    that split flat from up/down.
    """

    test_df: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    threshold: float
    flat_frac: float  # the multiplier behind `threshold`, for honest captions


def evaluate_direction(
    df: pd.DataFrame,
    feature_cols: list[str],
    move_col: str,
    test_frac: float = 0.2,
    flat_frac: float = 0.25,
    c: float = 1.0,
) -> DirectionResult:
    """Like evaluate(), but classifies the *direction* of the target's move
    (up / flat / down) instead of regressing its value — the tractable question
    for a near-white-noise series (see signal_model.train_direction).
    `move_col` is the target's MOVE column from prepare_target (classifying a
    raw level is meaningless: news tone sits below zero every day, so every row
    would be one class). Model accuracy is compared against always guessing
    train's majority direction. `flat_frac` sizes the flat dead-zone; `c` is
    the classifier's regularisation.

    The flat-class dead-zone is fitted on the train moves only and reused to
    label the test actuals, so the threshold never sees held-out data.
    """
    # the move column arrives NaN-free from prepare_target (the one owner of
    # the first-row-diff drop), so no re-cleaning here
    clean = evaluation.drop_market_closed(df)
    train_df, test_df = evaluation.chronological_split(clean, test_frac)

    threshold = evaluation.direction_threshold(train_df[move_col], flat_frac)
    y_train = evaluation.direction_labels(train_df[move_col], threshold)
    y_test = evaluation.direction_labels(test_df[move_col], threshold)

    model = signal_model.train_direction(train_df[feature_cols], y_train, c)
    predicted = signal_model.predict_direction(test_df[feature_cols], model)
    baseline = evaluation.majority_baseline(y_train, y_test.index)

    test_df["actual_dir"] = y_test.to_numpy()
    test_df["predicted_dir"] = predicted.to_numpy()

    metrics = {
        "model": evaluation.direction_metrics(y_test, predicted),
        "baseline": evaluation.direction_metrics(y_test, baseline),
    }
    return DirectionResult(test_df, metrics, threshold, flat_frac)


def run_direction(
    df: pd.DataFrame,
    feature_cols: list[str],
    move_col: str,
    flat_frac: float = 0.25,
    c: float = 1.0,
) -> pd.DataFrame:
    """In-sample direction fit for the live chart: classify every row's
    up/flat/down and add `actual_dir`/`predicted_dir`, so a return target can
    show a followable direction view instead of a flat regression line.
    `move_col` is the target's MOVE column from prepare_target. Trains on all
    rows (no split) like run(), so it's a fit-quality view, not held-out.
    """
    # the move column arrives NaN-free from prepare_target — a NaN here would
    # silently label as 0/'Oförändrad'
    move = df[move_col]
    threshold = evaluation.direction_threshold(move, flat_frac)
    y = evaluation.direction_labels(move, threshold)
    model = signal_model.train_direction(df[feature_cols], y, c)
    df["actual_dir"] = y.to_numpy()
    df["predicted_dir"] = signal_model.predict_direction(
        df[feature_cols], model
    ).to_numpy()
    return df


@dataclass
class AssetGroupResult:
    """Asset-similarity grouping: which streams move together. `corr` is the
    return-correlation matrix reordered so grouped assets are adjacent (ready to
    render as a heatmap), and `groups` maps each `<stem>_ret` column to its
    group id, in the same order.
    """

    corr: pd.DataFrame
    groups: pd.Series


def group_assets(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_groups: int = 3,
    linkage: str = "average",
) -> AssetGroupResult:
    """Group the selected streams by how their daily returns co-move. Uses each
    selected stream's `_ret` column (returns are stationary — the honest basis
    for correlation, unlike trending price levels). Needs ≥2 streams.
    """
    stems = dict.fromkeys(features.stream_of(c) for c in feature_cols)  # order-stable
    ret_cols = [f"{s}_ret" for s in stems if f"{s}_ret" in df.columns]
    if len(ret_cols) < 2:
        raise ValueError("Välj minst två streams med avkastning (_ret) att gruppera.")

    returns = df[ret_cols]
    groups = signal_model.cluster_assets(returns, n_groups, linkage)
    order = groups.sort_values().index
    return AssetGroupResult(returns.corr().loc[order, order], groups.loc[order])


if __name__ == "__main__":  # one-time backfill: python -m we_love_shorting.controller
    import sys

    logging.basicConfig(level=logging.INFO)
    retry, errors = backfill()
    if (
        retry or errors
    ):  # partial backfill -> non-zero exit so it isn't mistaken for done
        for table, msg in {
            **{t: "rate-limited (429)" for t in retry},
            **errors,
        }.items():
            log.error("backfill incomplete for %s: %s", table, msg)
        sys.exit(1)
