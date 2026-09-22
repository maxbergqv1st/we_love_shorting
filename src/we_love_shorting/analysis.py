"""Analysis modes: pluggable ways to analyse the wide feature table.

Each mode owns its model (fit + held-out evaluation) and *describes* what to
show as a list of render-agnostic `Panel`s — no Streamlit in here, so a mode can
be unit-tested by asserting on the panels it returns. The Streamlit layer holds
one generic panel renderer and iterates `MODES`; adding a fourth mode is one
registry entry, not another `if` branch.

`live_panels` fits in-sample over every row (the live chart, a fit-quality
view); `evaluate` runs the held-out (or unsupervised-quality) evaluation and
returns a mode-specific result that the app caches, then `evaluate_panels`
turns that cached result into panels. `context` feeds the AI chatbot (only the
regression mode has anything to say; the rest return "").
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import controller, features

# Numeric direction classes -> human labels, shared by the modes that show them.
DIRECTION_LABELS = {1: "Upp", 0: "Oförändrad", -1: "Ner"}
_ACTUAL, _PRED = "Faktisk", "Prediktion"
_SERIES_COLORS = ["#4c78a8", "#f58518"]  # actual = blue, prediction = orange


@dataclass
class Panel:
    """One render-agnostic piece the app can display. `kind` picks the renderer;
    the remaining fields are the arguments that kind needs.
    """

    kind: str  # "text" | "line" | "bar" | "table"
    caption: str = ""
    data: pd.DataFrame | None = None
    colors: list[str] | None = None  # line: explicit per-series colours
    horizontal: bool = False  # bar: sideways bars (labelled categories)
    gradient: bool = False  # table: render as a −1..1 correlation heatmap


def _kw(params: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Pick the given hyperparameter keys that are active (present) in `params`;
    absent ones are left out so the controller's own default applies. Lets a
    mode forward only the knobs it understands from the shared params dict."""
    return {k: params[k] for k in keys if k in params}


class AnalysisMode(ABC):
    """Base strategy: one predict-a-target approach over the feature table. Each
    mode owns its always-visible view (`live_panels`), its held-out evaluation
    (`evaluate` + `evaluate_panels`), and optional AI grounding (`context`).

    Tunable hyperparameters flow in as a single `params` dict (only the keys the
    user toggled on); each mode forwards the ones it understands via `_kw`.
    """

    label: str
    needs_target: bool

    @abstractmethod
    def live_panels(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target: str,
        label: str,
        params: dict[str, Any],
    ) -> list[Panel]:
        """The always-visible view for this mode."""

    def evaluate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target: str,
        params: dict[str, Any],
    ) -> Any:
        """Held-out evaluation, cached by the app. Modes without one (evaluates
        = False) keep this default."""
        return None

    def evaluate_panels(self, result: Any, label: str) -> list[Panel]:
        """Turn a cached `evaluate` result into panels."""
        return []

    def context(self, result: Any) -> str:
        """Grounding text for the AI chatbot. Empty = no chatbot for this mode."""
        return ""


class RegressionMode(AnalysisMode):
    label = "Regression"
    needs_target = True

    _MODEL_KEYS = ("alpha", "model_kind", "n_estimators", "max_depth")

    def live_panels(self, df, feature_cols, target, label, params):
        out = controller.run(
            df.copy(), feature_cols, target, **_kw(params, *self._MODEL_KEYS)
        ).set_index("date")
        pred = f"predicted_{target}"
        chart = out[[target, pred]].rename(columns={target: _ACTUAL, pred: _PRED})
        # Actual and prediction nearly overlap for a level target, so the pair
        # alone "says little" — a prediction-error line centred on 0 makes the
        # miss legible, especially zoomed to a week/month.
        err = out[[target]].sub(out[pred].to_numpy(), axis=0)
        err = err.rename(columns={target: "Fel (faktisk − prediktion)"})
        return [
            Panel(
                "line", f"Faktisk vs prediktion: {label}", chart, colors=_SERIES_COLORS
            ),
            Panel(
                "line",
                "Prediktionsfel, centrerat kring 0 — avstånd från noll = hur "
                "mycket modellen missar den dagen.",
                err,
                colors=["#e45756"],
            ),
        ]

    def evaluate(self, df, feature_cols, target, params):
        return controller.evaluate(
            df, feature_cols, target, **_kw(params, "test_frac", *self._MODEL_KEYS)
        )

    def evaluate_panels(self, result, label):
        # Kept deliberately tight: the metrics table (the honest model-vs-
        # baseline verdict), the weight view (what the model leans on) and one
        # residual view. The scatter and over-time residual lines were two
        # more views of the same information.
        test_df = result.test_df
        metrics = pd.DataFrame(result.metrics).rename(
            columns={"model": "Modell", "baseline": "Baseline"},
            index={"rmse": "RMSE", "mae": "MAE"},
        )
        w = result.weights.reindex(result.weights.abs().sort_values().index)
        counts, edges = np.histogram(test_df["residual"], bins=20)
        mids = (edges[:-1] + edges[1:]) / 2
        hist = pd.DataFrame({"Antal": counts}, index=pd.Index(mids, name="Residual"))
        return [
            Panel("table", "Mätvärden (modell vs baseline)", metrics),
            Panel(
                "bar",
                "Vad modellen lutar sig på: koefficienter (linjär/Ridge — Ridges "
                "är på skalade features och därmed jämförbara; rå OLS bär "
                "featurens enhet) eller feature importances (Random Forest).",
                w.rename("Vikt").to_frame(),
                horizontal=True,
            ),
            Panel("bar", "Histogram över residualer (0-centrerat = oskevt fel)", hist),
        ]

    def context(self, result):
        test_df = result.test_df
        return (
            f"Baseline-typ: {result.baseline_kind}. "
            f"Mätvärden (modell vs baseline): {result.metrics}. "
            f"Testperiod: {test_df['date'].min()} till {test_df['date'].max()} "
            f"({len(test_df)} rader)."
        )


class DirectionMode(AnalysisMode):
    label = "Riktning"
    needs_target = True

    _WINDOW = 21  # ~one trading month

    def live_panels(self, df, feature_cols, target, label, params):
        # Per-day hit/miss is ~coin-flip noise smeared near y=0 and unreadable.
        # A rolling hit-rate line vs the "always guess the commonest direction"
        # baseline shows when (and whether) the model actually has an edge.
        out = controller.run_direction(
            df.copy(), feature_cols, target, **_kw(params, "flat_frac", "c")
        )
        correct = (out["actual_dir"] == out["predicted_dir"]).astype(float)
        rolling = correct.rolling(self._WINDOW, min_periods=self._WINDOW // 2).mean()
        baseline = float(out["actual_dir"].value_counts(normalize=True).max())
        chart = pd.DataFrame(
            {
                f"Träffsäkerhet ({self._WINDOW}d glidande)": rolling.to_numpy(),
                "Baseline (gissa vanligaste)": baseline,
            },
            index=pd.Index(out["date"], name="date"),
        )
        return [
            Panel(
                "line",
                f"Rullande träffsäkerhet på riktningen för {label}. Över "
                "baseline-linjen = modellen slår 'gissa alltid vanligaste "
                "riktningen'; under = ingen edge.",
                chart,
                colors=_SERIES_COLORS,
            ),
        ]

    def evaluate(self, df, feature_cols, target, params):
        return controller.evaluate_direction(
            df, feature_cols, target, **_kw(params, "test_frac", "flat_frac", "c")
        )

    def evaluate_panels(self, result, label):
        acc = pd.DataFrame(result.metrics).rename(
            columns={"model": "Modell", "baseline": "Majoritets-baseline"},
            index={"accuracy": "Träffsäkerhet"},
        )
        confusion = pd.crosstab(
            result.test_df["actual_dir"].map(DIRECTION_LABELS),
            result.test_df["predicted_dir"].map(DIRECTION_LABELS),
            rownames=["Faktisk"],
            colnames=["Predikterad"],
        )
        return [
            Panel(
                "text",
                f"Dödzon för 'oförändrad': ±{result.threshold:.4f} (0.25 × "
                "tränings-std). Baseline = gissa alltid vanligaste klassen.",
            ),
            Panel("table", "Träffsäkerhet (modell vs baseline)", acc),
            Panel(
                "table",
                "Förväxlingsmatris (rad = faktiskt, kolumn = predikterat)",
                confusion,
            ),
        ]


# Registry: the predict-a-target modes the user switches between. Add a mode =
# add one entry. Asset grouping is NOT here — it's target-independent (about the
# features, not a prediction), so it lives in its own always-on section.
MODES: dict[str, AnalysisMode] = {
    m.label: m for m in (RegressionMode(), DirectionMode())
}


def asset_grouping_panels(
    df: pd.DataFrame, feature_cols: list[str], params: dict[str, Any] | None = None
) -> list[Panel]:
    """Group the selected streams by return co-movement: a correlation heatmap
    (reordered so co-moving streams form blocks on the diagonal) plus the
    resulting groups. Target-independent, so it's a standalone view, not a mode.
    """
    result = controller.group_assets(
        df, feature_cols, **_kw(params or {}, "n_groups", "linkage")
    )
    strip = {c: features.stream_of(c) for c in result.corr.columns}  # gold_ret -> gold

    # one row per group: "Grupp 1 | gold, silver, copper"
    per_asset = result.groups.rename_axis("Tillgång").reset_index()
    per_asset["Tillgång"] = per_asset["Tillgång"].map(features.stream_of)
    members = per_asset.groupby("group")["Tillgång"].apply(", ".join).reset_index()
    members["group"] = "Grupp " + (members["group"] + 1).astype(str)
    members.columns = ["Grupp", "Tillgångar"]
    return [
        Panel(
            "table",
            "Samvariation mellan tillgångarnas avkastning (1 = rör sig lika, −1 = "
            "motsatt). Omordnad så grupper hamnar i block längs diagonalen.",
            result.corr.rename(index=strip, columns=strip),
            gradient=True,
        ),
        Panel(
            "table",
            "Tillgångsgrupper (agglomerativ klustring på 1 − korrelation — "
            "tillgångar som rör sig ihop hamnar i samma grupp).",
            members,
        ),
    ]
