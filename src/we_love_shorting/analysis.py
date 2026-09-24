"""Analysis modes: pluggable ways to analyse the wide feature table.

Each mode owns its model (fit + held-out evaluation) and *describes* what to
show as a list of render-agnostic `Panel`s — no Streamlit in here, so a mode can
be unit-tested by asserting on the panels it returns. The Streamlit layer holds
one generic panel renderer and iterates `MODES`; adding a fourth mode is one
registry entry, not another `if` branch.

`live_panels` fits in-sample over every row (the live chart, a fit-quality
view); `evaluate` runs the held-out (or unsupervised-quality) evaluation and
returns a mode-specific result that the app caches, then `evaluate_panels`
turns that cached result into panels. `context` feeds the AI chatbot the
mode-specific grounding for the latest evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import controller, evaluation, features
from .signal_model import MODEL_KEYS

# Numeric direction classes -> human labels, shared by the modes that show them.
DIRECTION_LABELS = {1: "Upp", 0: "Oförändrad", -1: "Ner"}
_ACTUAL, _PRED = "Faktisk", "Prediktion"
SERIES_COLORS = ["#4c78a8", "#f58518"]  # actual = blue, prediction = orange


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

    The target arrives as a `controller.TargetSpec` (from prepare_target): the
    model always trains on `spec.train_col` — the target's move — and a level
    pick additionally carries the level column for reconstruction/display.

    Tunable hyperparameters flow in as a single `params` dict (only the keys the
    user toggled on); each mode forwards the ones it understands via `_kw`.
    """

    label: str

    @abstractmethod
    def live_panels(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        spec: controller.TargetSpec,
        label: str,
        params: dict[str, Any],
    ) -> list[Panel]:
        """The always-visible view for this mode."""

    def evaluate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        spec: controller.TargetSpec,
        params: dict[str, Any],
    ) -> Any:
        """Held-out evaluation, cached by the app. A mode without one keeps
        this default."""
        return None

    def evaluate_panels(self, result: Any, label: str) -> list[Panel]:
        """Turn a cached `evaluate` result into panels."""
        return []

    def context(self, result: Any) -> str:
        """Grounding text for the AI chatbot. Empty = no chatbot for this mode."""
        return ""


class RegressionMode(AnalysisMode):
    label = "Regression"

    def live_panels(self, df, feature_cols, spec, label, params):
        out = controller.run(
            df.copy(), feature_cols, spec.train_col, **_kw(params, *MODEL_KEYS)
        ).set_index("date")
        pred = f"predicted_{spec.train_col}"
        # In-sample fit view only — no error line here: in-sample error reads
        # as skill (a forest fits ~perfectly); the honest error view is the
        # held-out residual histogram in the evaluation card.
        #
        # ALWAYS the move scale, never a reconstructed level: a level curve
        # rebuilt on the previous ACTUAL level hugs the actual by construction
        # (that closeness is persistence, not skill) and hides the one thing
        # the model actually says — up or down. On the move scale the
        # prediction line's sign IS the model's daily call.
        chart = out[[spec.train_col, pred]].rename(
            columns={spec.train_col: _ACTUAL, pred: _PRED}
        )
        caption = f"Faktisk vs prediktion: {label}"
        if spec.kind != "identity":
            caption = (
                f"Daglig förändring, faktisk vs prediktion: {label}. Nivån är "
                "~en random walk, så modellen tränas på förändringen — "
                "prediktionslinjens tecken är modellens upp/ner-bedömning per "
                "dag. Nivåskalan finns i live-kortet och i evalueringens "
                "mätvärden."
            )
        return [Panel("line", caption, chart, colors=SERIES_COLORS)]

    def evaluate(self, df, feature_cols, spec, params):
        result = controller.evaluate(
            df, feature_cols, spec.train_col, **_kw(params, "test_frac", *MODEL_KEYS)
        )
        if spec.kind != "identity":
            result = controller.add_level_view(result, spec)
        return result

    def evaluate_panels(self, result, label):
        # Kept deliberately tight: the metrics table (the honest model-vs-
        # baseline verdict), a held-out actual-vs-prediction line, the weight
        # view (what the model leans on) and one residual view. The scatter
        # and over-time residual lines were two more views of the same
        # information.
        test_df = result.test_df.set_index("date")
        metrics = pd.DataFrame(result.metrics).rename(
            index={"rmse": "RMSE", "mae": "MAE"}
        )
        if result.level_metrics:
            # move scale = model vs mean-baseline (the honest skill verdict);
            # level scale = reconstructed prediction vs persistence (intuitive)
            metrics = metrics.rename(lambda i: f"{i} (förändring)")
            level = pd.DataFrame(result.level_metrics).rename(
                index={"rmse": "RMSE (nivå)", "mae": "MAE (nivå)"}
            )
            metrics = pd.concat([metrics, level])
        metrics = metrics.rename(columns={"model": "Modell", "baseline": "Baseline"})
        # held-out chart on the MOVE scale for every kind — a reconstructed
        # level line hugs the actual by construction (see live_panels)
        target, pred = result.target, f"predicted_{result.target}"
        held = test_df[[target, pred]].rename(columns={target: _ACTUAL, pred: _PRED})
        held_caption = (
            f"Held-out: faktisk vs prediktion för {label} över testperioden — "
            "data modellen aldrig sett."
        )
        if result.level_metrics:
            held_caption += (
                " Skalan är den dagliga förändringen; prediktionens tecken = "
                "modellens upp/ner-bedömning. Nivå-RMSE:n står i tabellen ovan."
            )
        w = result.weights.reindex(result.weights.abs().sort_values().index)
        counts, edges = np.histogram(result.test_df["residual"], bins=20)
        mids = (edges[:-1] + edges[1:]) / 2
        hist = pd.DataFrame({"Antal": counts}, index=pd.Index(mids, name="Residual"))
        return [
            Panel("table", "Mätvärden (modell vs baseline)", metrics),
            Panel("line", held_caption, held, colors=SERIES_COLORS),
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
        text = (
            f"Baseline-typ: {result.baseline_kind}. "
            f"Mätvärden på daglig förändring (modell vs baseline): "
            f"{result.metrics}. "
            f"Testperiod: {test_df['date'].min()} till {test_df['date'].max()} "
            f"({len(test_df)} rader)."
        )
        if result.level_metrics:
            text += (
                " Nivåskala (rekonstruerad prediktion vs persistence): "
                f"{result.level_metrics}."
            )
        return text


class DirectionMode(AnalysisMode):
    label = "Riktning"

    _WINDOW = 21  # ~one trading month

    def live_panels(self, df, feature_cols, spec, label, params):
        # Per-day hit/miss is ~coin-flip noise smeared near y=0 and unreadable.
        # A rolling hit-rate line vs the "always guess the commonest direction"
        # baseline shows when (and whether) the model actually has an edge.
        out = controller.run_direction(
            df.copy(), feature_cols, spec.train_col, **_kw(params, "flat_frac", "c")
        )
        correct = (out["actual_dir"] == out["predicted_dir"]).astype(float)
        rolling = correct.rolling(self._WINDOW, min_periods=self._WINDOW // 2).mean()
        # the same majority-baseline definition as the held-out evaluation —
        # one concept, one source
        majority = evaluation.majority_baseline(out["actual_dir"], out.index)
        baseline = evaluation.direction_metrics(out["actual_dir"], majority)["accuracy"]
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
                colors=SERIES_COLORS,
            ),
        ]

    def evaluate(self, df, feature_cols, spec, params):
        return controller.evaluate_direction(
            df,
            feature_cols,
            spec.train_col,
            **_kw(params, "test_frac", "flat_frac", "c"),
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
                f"Dödzon för 'oförändrad': ±{result.threshold:.4f} "
                f"({result.flat_frac:g} × tränings-std). "
                "Baseline = gissa alltid vanligaste klassen.",
            ),
            Panel("table", "Träffsäkerhet (modell vs baseline)", acc),
            Panel(
                "table",
                "Förväxlingsmatris (rad = faktiskt, kolumn = predikterat)",
                confusion,
            ),
        ]

    def context(self, result):
        return (
            f"Riktningsklassificering (upp/oförändrad/ner). "
            f"Träffsäkerhet (modell vs majoritets-baseline): {result.metrics}. "
            f"Dödzon för 'oförändrad': ±{result.threshold:.4f}."
        )


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
