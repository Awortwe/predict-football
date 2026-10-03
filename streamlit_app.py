"""Streamlit front end for predict_football.

Run it with::

    streamlit run streamlit_app.py

The app's job is to show probabilities without overselling them. Two rules are
visible on screen, not just in the code:

* Every forecast is out of sample -- the model behind a fixture is trained only
  on matches that kicked off strictly before it -- and carries the number of
  matches it was trained on.
* Model quality is reported next to a base-rate and a closing-market benchmark,
  with the sample size each metric was measured on. The interpretable baseline
  and the gradient-boosted challenger are always scored on the same matches, so
  neither can win by being evaluated on the easier fixtures.

All data loading and model fitting is cached, so the expensive walk-forward
evaluation runs once per (data, hyperparameter) combination rather than on every
keystroke.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.is_dir():
    sys.path.insert(0, str(_SRC_DIR))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from predict_football.benchmark import BenchmarkError, ModelBenchmark, benchmark_models
from predict_football.config.leagues import get_league
from predict_football.config.licences import LicenceViolation, assert_can_serve_publicly, attribution_for
from predict_football.config.settings import Settings
from predict_football.data.bootstrap import (
    PUBLIC_LEAGUE,
    PUBLIC_PROVIDER,
    BootstrapError,
    PublicDataStatus,
    ensure_public_data,
)
from predict_football.data.repository import Database, MatchRepository
from predict_football.evaluation import ForecastReport
from predict_football.inference import (
    InferenceError,
    ModelOptions,
    predict_fixtures,
)
from predict_football.models.dixon_coles import score_matrix
from predict_football.models.feature_model import FeatureModelOptions

logger = logging.getLogger(__name__)

#: Recency half-life choices, in matches. ``None`` weights every match equally.
HALF_LIFE_OPTIONS: dict[str, float | None] = {
    "Even": None,
    "60": 60.0,
    "120": 120.0,
    "240": 240.0,
}

#: Forecast models a reader may choose between. The baseline is transparent and
#: interpretable; the challenger spends the engineered features and has to earn
#: its place on the same matches.
MODEL_OPTIONS: tuple[str, ...] = ("Dixon-Coles", "LightGBM")

#: Outcome labels, in the canonical home/draw/away order.
OUTCOME_LABELS: dict[str, str] = {"H": "Home win", "D": "Draw", "A": "Away win"}

#: Flag marking a deployment as publicly reachable. Set it in the host's secrets
#: to turn on the licence guard and the first-run bootstrap; a local checkout
#: leaves it unset and keeps using whatever is already in ``data/``.
PUBLIC_DEPLOY_FLAG = "PREDICT_FOOTBALL_PUBLIC_DEPLOY"

#: Name of the secret holding the football-data.org token. The free tier still
#: requires a registered token.
API_KEY_SECRET = "FOOTBALL_DATA_ORG_API_KEY"


@st.cache_resource(show_spinner=False)
def _repository(database_path: str) -> MatchRepository:
    """Open the shared SQLite repository once per process.

    Args:
        database_path: Absolute path to the SQLite file.

    Returns:
        A repository bound to that database.
    """
    return MatchRepository(Database(database_path))


@st.cache_data(show_spinner=False)
def _load_matches(database_path: str, signature: float) -> pd.DataFrame:
    """Load every stored match, cached against the file's modification time.

    Args:
        database_path: Absolute path to the SQLite file.
        signature: Modification time, used only to invalidate the cache when the
            database changes (for example after a poll).

    Returns:
        Canonical match frame ordered chronologically.
    """
    return _repository(database_path).load_matches(include_stats=False)


@st.cache_data(show_spinner=False)
def _load_odds(database_path: str, signature: float) -> pd.DataFrame:
    """Load the stored closing odds, cached against the file's modification time.

    Args:
        database_path: Absolute path to the SQLite file.
        signature: Modification time, used only to invalidate the cache.

    Returns:
        Odds frame with match date and teams attached.
    """
    return _repository(database_path).load_odds()


@st.cache_data(show_spinner=False)
def _forecast(
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    min_train: int,
    model: str,
    ridge: float,
    half_life: float | None,
) -> pd.DataFrame:
    """Forecast fixtures, cached on the history and hyperparameters.

    Args:
        history: Completed matches used for training.
        fixtures: Unplayed fixtures to forecast.
        min_train: Warm-up matches required before a forecast is made.
        model: Model label, one of :data:`MODEL_OPTIONS`.
        ridge: Ridge penalty on team ratings. Ignored by the challenger.
        half_life: Recency half-life in matches, or ``None``. Ignored by the
            challenger.

    Returns:
        Forecast frame as produced by :func:`predict_fixtures`.
    """
    return predict_fixtures(
        history,
        fixtures,
        min_train_matches=min_train,
        model_factory=_model_factory(model, ridge, half_life),
    )


def _model_factory(model: str, ridge: float, half_life: float | None):
    """Return a fresh model factory for the requested label.

    Args:
        model: Model label, either ``"Dixon-Coles"`` or ``"LightGBM"``.
        ridge: Ridge penalty, used only by Dixon-Coles.
        half_life: Recency half-life, used only by Dixon-Coles.

    Returns:
        A zero-argument callable returning an unfitted model.
    """
    if model == "LightGBM":
        return FeatureModelOptions().build
    return ModelOptions(ridge=ridge, half_life_matches=half_life).build


@st.cache_data(show_spinner=False)
def _benchmark(
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    min_train: int,
    refit: int,
    ridge: float,
    half_life: float | None,
) -> ModelBenchmark:
    """Run the head-to-head walk-forward once per data/hyperparameter combination.

    Both the baseline and the challenger are walked forward and then scored on
    the intersection of matches they covered, so the comparison cannot flatter
    whichever model happened to predict the easier fixtures.

    Args:
        matches: Completed canonical match frame for one competition.
        odds: Closing-odds frame, possibly empty.
        min_train: Warm-up matches before the first forecast.
        refit: Distinct matchdays predicted per fitted model.
        ridge: Ridge penalty on team ratings, used by Dixon-Coles.
        half_life: Recency half-life in matches, used by Dixon-Coles.

    Returns:
        A :class:`ModelBenchmark` carrying the aligned predictions and reports.
    """
    models = {
        "Dixon-Coles": ModelOptions(ridge=ridge, half_life_matches=half_life).build,
        "LightGBM": FeatureModelOptions().build,
    }
    return benchmark_models(
        matches,
        odds,
        league_key="",
        models=models,
        min_train_matches=min_train,
        refit_every_dates=refit,
    )


def _league_options(matches: pd.DataFrame) -> dict[str, str]:
    """Build the competition selector, showing only leagues with stored data.

    Args:
        matches: Full match frame.

    Returns:
        Mapping of display label to internal league key, sorted by key.
    """
    keys = sorted(key for key in matches["league_key"].dropna().unique())
    options: dict[str, str] = {}
    for key in keys:
        try:
            name = get_league(str(key)).name
        except KeyError:
            name = str(key)
        options[f"{name} ({key})"] = str(key)
    return options


def _fixture_frame(forecast: pd.DataFrame) -> pd.DataFrame:
    """Reshape a forecast frame into the columns shown to a reader.

    Args:
        forecast: Output of :func:`_forecast`.

    Returns:
        Display frame with a readable date and percentages.
    """
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(forecast["match_date"]).dt.strftime("%Y-%m-%d"),
            "Home": forecast["home_team"],
            "Away": forecast["away_team"],
            "Home win": forecast["prob_home"],
            "Draw": forecast["prob_draw"],
            "Away win": forecast["prob_away"],
            "Trained on": forecast["n_train_matches"],
            "Note": forecast["note"],
        }
    )


def _calibration_bins(report: ForecastReport, outcome: str) -> pd.DataFrame:
    """Build the per-bin reliability table for one outcome.

    Args:
        report: Scored report carrying per-outcome calibration bins.
        outcome: One of ``"H"``, ``"D"``, ``"A"``.

    Returns:
        Frame with one row per populated probability bin.
    """
    summary = report.calibration[outcome]
    return pd.DataFrame(
        {
            "Bin": np.arange(len(summary.counts)),
            "Predicted": summary.mean_predicted,
            "Observed": summary.observed_frequency,
            "Matches": summary.counts,
        }
    ).dropna(subset=["Predicted", "Observed"])


def _reliability_frame(bins: pd.DataFrame) -> pd.DataFrame:
    """Reshape the reliability table into long form for a line chart.

    Args:
        bins: Output of :func:`_calibration_bins`.

    Returns:
        Frame with one row per bin per series, suitable for ``st.line_chart``.
    """
    return bins.melt(
        id_vars=["Bin", "Matches"],
        value_vars=["Predicted", "Observed"],
        var_name="Series",
        value_name="Probability",
    )


def _next_saturday(today: pd.Timestamp | None = None) -> date:
    """Return the next Saturday, the common default matchday.

    Args:
        today: Reference date. Defaults to the current date.

    Returns:
        The next Saturday strictly after ``today``.
    """
    moment = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    ahead = (5 - moment.dayofweek) % 7
    ahead = 7 if ahead == 0 else ahead
    return (moment + pd.Timedelta(days=ahead)).date()


def _default_match_date(league: pd.DataFrame) -> date:
    """Pick the matchday the predictor opens on.

    Args:
        league: Selected competition's matches, with a datetime ``match_date``.

    Returns:
        The earliest stored match date from today onward, else the most recent
        stored date, else next Saturday. Opening on a real matchday beats a blank
        calendar day.
    """
    dated = pd.to_datetime(league["match_date"], errors="coerce").dropna()
    if dated.empty:
        return _next_saturday()
    today = pd.Timestamp.today().normalize()
    upcoming = dated[dated >= today]
    if not upcoming.empty:
        return upcoming.min().date()
    return dated.max().date()


def _nearby_matchdays(league: pd.DataFrame, when: date, window_days: int = 21) -> str:
    """List stored matchdays close to a date that has none.

    Args:
        league: Selected competition's matches.
        when: The date the reader picked, which has no matches.
        window_days: How far either side of ``when`` to look.

    Returns:
        Comma-separated dates, or an empty string when nothing is near.
    """
    dated = pd.to_datetime(league["match_date"], errors="coerce").dropna()
    target = pd.Timestamp(when)
    window = dated[(dated - target).abs() <= pd.Timedelta(days=window_days)]
    nearest = sorted({stamp.date().isoformat() for stamp in window})
    return ", ".join(nearest[:8])


def _render_scoreline_heatmap(lam_home: float, lam_away: float, rho: float, home: str, away: str) -> None:
    """Draw the joint scoreline distribution behind a forecast.

    This is the matrix the three outcome probabilities are summed from, so the
    reader can see *why* the home, draw or away number is what it is rather than
    taking it on trust.

    Args:
        lam_home: Expected home goals.
        lam_away: Expected away goals.
        rho: Fitted low-score dependence parameter.
        home: Home club name, used for the axis.
        away: Away club name, used for the axis.
    """
    matrix = score_matrix(lam_home, lam_away, rho).iloc[:7, :7]
    long = pd.DataFrame(
        {
            "home_goals": np.repeat(matrix.index.to_numpy(), matrix.shape[1]),
            "away_goals": np.tile(matrix.columns.to_numpy(), matrix.shape[0]),
            "Probability": matrix.to_numpy().ravel(),
        }
    )
    base = alt.Chart(long)
    heat = base.mark_rect().encode(
        x=alt.X("home_goals:O", title=f"{home} goals"),
        y=alt.Y("away_goals:O", title=f"{away} goals", sort="descending"),
        color=alt.Color("Probability:Q", scale=alt.Scale(scheme="blues"), title="Probability"),
        tooltip=[
            alt.Tooltip("home_goals:O", title=home),
            alt.Tooltip("away_goals:O", title=away),
            alt.Tooltip("Probability:Q", format=".1%"),
        ],
    )
    labels = base.mark_text(size=11).encode(
        x=alt.X("home_goals:O"),
        y=alt.Y("away_goals:O", sort="descending"),
        text=alt.Text("Probability:Q", format=".0%"),
        color=alt.condition(
            alt.datum.Probability > float(long["Probability"].max()) * 0.5,
            alt.value("white"),
            alt.value("#1f2937"),
        ),
    )
    st.altair_chart(heat + labels, alt=f"Probability of each scoreline, {home} versus {away}")
    st.caption(
        "Joint probability of each scoreline from the Dixon-Coles fit. Cells go up to six goals "
        "each; the model also prices higher scores, so the displayed cells sum to just under 100%."
    )


def _render_day_predictor(matches: pd.DataFrame, controls: dict[str, object]) -> None:
    """Let the reader pick a day and forecast the matches stored on it.

    The reader chooses a competition (in the sidebar) and a date; every stored
    match on that date is forecast by a model trained only on matches that kicked
    off strictly before it, so a played date is scored honestly and a future one
    is a genuine forecast. Selecting a match shows the outcome probabilities and,
    for Dixon-Coles, the scoreline distribution behind them.

    Args:
        matches: Full match frame.
        controls: Current sidebar control values.
    """
    league_key = str(controls["league"])
    league = matches[matches["league_key"] == league_key].copy()
    if league.empty or "match_date" not in league.columns:
        return
    league["match_date"] = pd.to_datetime(league["match_date"], errors="coerce")
    league = league.dropna(subset=["match_date", "home_team", "away_team"])

    complete = league["result"].notna() & league["home_goals"].notna() & league["away_goals"].notna()
    history = league[complete].copy()

    st.subheader("Predict a match")
    st.caption(
        "Pick a date. The matches stored for that day appear below, each forecast by a model "
        "trained only on matches that kicked off strictly before it, so the result of a match "
        "already played is never used against it."
    )
    if history.empty:
        st.warning("This competition needs at least one completed match before anything can be predicted.")
        return

    when = st.date_input("Match date", value=_default_match_date(league), key=f"day_date_{league_key}")
    day = league[league["match_date"].dt.date == when].reset_index(drop=True)
    if day.empty:
        nearby = _nearby_matchdays(league, when)
        st.info(
            f"No stored matches on {when:%Y-%m-%d}." + (f" Nearby matchdays: {nearby}." if nearby else "")
        )
        return

    model = str(controls["model"])
    try:
        forecast = _forecast(
            history,
            day[["match_id", "match_date", "home_team", "away_team"]],
            int(controls["min_train"]),
            model,
            float(controls["ridge"]),
            controls["half_life"],  # type: ignore[arg-type]
        )
    except InferenceError as error:
        st.error(f"Could not forecast this matchday: {error}")
        return

    played = (day["result"].notna() & day["home_goals"].notna() & day["away_goals"].notna()).to_numpy()
    actual = [
        f"{int(home)}-{int(away)}" if is_played else ""
        for home, away, is_played in zip(day["home_goals"], day["away_goals"], played, strict=True)
    ]
    st.dataframe(
        pd.DataFrame(
            {
                "Home": forecast["home_team"],
                "Away": forecast["away_team"],
                "Home win": forecast["prob_home"],
                "Draw": forecast["prob_draw"],
                "Away win": forecast["prob_away"],
                "Actual": actual,
                "Trained on": forecast["n_train_matches"],
            }
        ),
        hide_index=True,
        alt="Matches on the selected day with home, draw and away probabilities",
        column_config={
            "Home win": st.column_config.NumberColumn(format="percent"),
            "Draw": st.column_config.NumberColumn(format="percent"),
            "Away win": st.column_config.NumberColumn(format="percent"),
            "Trained on": st.column_config.NumberColumn(format="%d"),
        },
    )

    labels = [f"{row.home_team} vs {row.away_team}" for row in day.itertuples()]
    selected_label = st.selectbox("Match detail", options=labels, key=f"day_match_{league_key}")
    choice = labels.index(selected_label)
    row = forecast.iloc[choice]
    home = str(row["home_team"])
    away = str(row["away_team"])

    st.markdown(f"**{home} vs {away}** — {when:%Y-%m-%d}")
    if not bool(row["is_forecast"]):
        st.warning(str(row["note"]) or "Not enough prior matches to make a forecast.")
        return

    with st.container(horizontal=True):
        st.metric("Home win", f"{row['prob_home']:.1%}", border=True)
        st.metric("Draw", f"{row['prob_draw']:.1%}", border=True)
        st.metric("Away win", f"{row['prob_away']:.1%}", border=True)

    chart_col, heat_col = st.columns(2)
    with chart_col:
        st.markdown("**Outcome probabilities**")
        st.bar_chart(
            pd.DataFrame(
                {
                    "Outcome": [f"{home} win", "Draw", f"{away} win"],
                    "Probability": [
                        row["prob_home"] * 100,
                        row["prob_draw"] * 100,
                        row["prob_away"] * 100,
                    ],
                }
            ),
            x="Outcome",
            y="Probability",
            sort=False,
            y_label="Probability (%)",
            alt=f"Home, draw and away win probabilities for {home} versus {away}",
        )
    with heat_col:
        st.markdown("**Most likely scorelines**")
        if pd.notna(row["lambda_home"]) and pd.notna(row["model_rho"]):
            _render_scoreline_heatmap(
                float(row["lambda_home"]),
                float(row["lambda_away"]),
                float(row["model_rho"]),
                home,
                away,
            )
        else:
            st.caption("This model does not report a scoreline distribution.")

    if pd.notna(row["lambda_home"]):
        st.caption(f"Expected goals: {home} {row['lambda_home']:.2f} - {row['lambda_away']:.2f} {away}.")
    st.caption(
        f"Trained on {int(row['n_train_matches'])} earlier matches with the {model} model. "
        "Percentages are estimates, not guarantees."
    )
    if int(row["unknown_teams"]) > 0:
        st.warning(
            "At least one club was absent from the training window, so it was given a "
            "league-average rating; this forecast rests on no evidence about that side."
        )
    if bool(played[choice]):
        actual_home = int(day.iloc[choice]["home_goals"])
        actual_away = int(day.iloc[choice]["away_goals"])
        actual_outcome = "H" if actual_home > actual_away else "A" if actual_home < actual_away else "D"
        best = max(
            zip(("H", "D", "A"), (row["prob_home"], row["prob_draw"], row["prob_away"]), strict=True),
            key=lambda pair: pair[1],
        )[0]
        verdict = "correct" if best == actual_outcome else "wrong"
        st.caption(
            f"This match has been played: {home} {actual_home}-{actual_away} {away}. "
            f"The model's most likely outcome was {OUTCOME_LABELS[best]} ({verdict})."
        )


def _render_header() -> None:
    """Render the title and the honesty banner."""
    st.title("predict_football")
    st.caption("Calibrated pre-match probabilities, scored out of sample.")
    st.info(
        "Every probability below comes from a model trained only on matches that "
        "kicked off **strictly before** the fixture. No result shown was available "
        "when the prediction was made, and every metric states the number of "
        "matches it was measured on.",
    )


def _render_sidebar(league_options: dict[str, str]) -> dict[str, object]:
    """Render the controls and capture a run request.

    Args:
        league_options: Mapping of display label to league key.

    Returns:
        The current control values, so a caller can forecast with them without
        waiting for **Run forecast** to be pressed. Pressing the button also
        stores the same values under ``session_state["run"]`` to trigger the
        fixtures and diagnostics sections.
    """
    labels = list(league_options)
    with st.sidebar:
        st.header("Controls")
        label = st.selectbox("Competition", labels, key="league_label")
        model = st.segmented_control(
            "Model",
            options=list(MODEL_OPTIONS),
            default="Dixon-Coles",
            help="Dixon-Coles is the interpretable baseline; LightGBM spends the engineered features.",
        )
        min_train = st.slider("Warm-up matches", min_value=100, max_value=760, value=380, step=20)
        ridge = st.select_slider("Ridge penalty", options=[0.02, 0.05, 0.1, 0.2], value=0.1)
        half_life_label = st.segmented_control(
            "Recency half-life (matches)",
            options=list(HALF_LIFE_OPTIONS),
            default="240",
            help="How quickly older matches lose influence. 'Even' weights them equally.",
        )
        refit = st.slider(
            "Refit every N matchdays",
            min_value=1,
            max_value=20,
            value=20,
            help="A larger cadence is far faster but scores slightly staler models. One refits on every matchday.",
        )
        st.divider()
        run_pressed = st.button(
            "Run forecast", key="run_button", type="primary", icon=":material/play_arrow:"
        )
        st.caption(
            "The walk-forward evaluation refits every model on an expanding window, "
            "so the first run can take a little while; later runs are cached. Ridge "
            "and half-life apply to Dixon-Coles only."
        )
    controls: dict[str, object] = {
        "league": league_options[label],
        "model": model or "Dixon-Coles",
        "min_train": int(min_train),
        "ridge": float(ridge),
        "half_life": HALF_LIFE_OPTIONS.get(half_life_label or "240"),
        "refit": int(refit),
    }
    if run_pressed:
        st.session_state["run"] = dict(controls)
    return controls


def _render_fixtures(matches: pd.DataFrame, run: dict[str, object], odds: pd.DataFrame) -> ForecastReport | None:
    """Render the upcoming-fixtures table and return the latest model report.

    Args:
        matches: Full match frame.
        run: Run parameters captured from the sidebar.
        odds: Closing-odds frame, possibly empty.

    Returns:
        The model's :class:`ForecastReport`, or ``None`` when the competition has
        no unplayed fixtures to show.
    """
    league_key = str(run["league"])
    league = matches[matches["league_key"] == league_key]
    complete = league["result"].notna() & league["home_goals"].notna() & league["away_goals"].notna()
    history, fixtures = league[complete].copy(), league[~complete].copy()

    st.subheader("Upcoming fixtures")
    if history.empty:
        st.warning("No completed matches are stored for this competition, so nothing can be trained.")
        return None
    if fixtures.empty:
        st.warning("Every stored match for this competition already has a result, so there is nothing to forecast.")
    else:
        try:
            forecast = _forecast(
                history,
                fixtures,
                int(run["min_train"]),
                str(run["model"]),
                float(run["ridge"]),
                run["half_life"],  # type: ignore[arg-type]
            )
        except InferenceError as error:
            st.error(f"Could not forecast this competition: {error}")
        else:
            made = int(forecast["is_forecast"].sum())
            st.caption(
                f"Forecast {made} of {len(forecast)} stored fixtures. A fixture needs "
                f"{run['min_train']} prior matches before a number is shown; the rest are "
                "left blank rather than guessed."
            )
            st.dataframe(
                _fixture_frame(forecast),
                hide_index=True,
                alt="Upcoming fixtures with home, draw and away probabilities",
                column_config={
                    "Home win": st.column_config.NumberColumn(format="percent"),
                    "Draw": st.column_config.NumberColumn(format="percent"),
                    "Away win": st.column_config.NumberColumn(format="percent"),
                    "Trained on": st.column_config.NumberColumn(format="%d"),
                },
            )
            unknown = forecast[forecast["unknown_teams"] > 0]
            if len(unknown):
                st.warning(
                    f"{len(unknown)} fixture(s) involve a club absent from the training window. "
                    "Those sides receive the league-average rating, so the forecast rests on no "
                    "evidence about them.",
                )

    st.subheader("Model diagnostics")
    st.caption("Out-of-sample walk-forward evaluation. Every model is scored on the same matches.")
    try:
        benchmark = _benchmark(
            league,
            odds if odds is not None else pd.DataFrame(),
            int(run["min_train"]),
            int(run["refit"]),
            float(run["ridge"]),
            run["half_life"],  # type: ignore[arg-type]
        )
    except BenchmarkError as error:
        st.warning(f"Not enough history to evaluate this competition: {error}")
        return None

    if not benchmark.reports:
        return None
    selected = next(
        (report for report in benchmark.reports if report.label == str(run["model"])),
        benchmark.reports[0],
    )
    covered = benchmark.n_matches
    with st.container(horizontal=True):
        st.metric(
            f"{selected.label} Brier",
            f"{selected.brier:.4f}",
            border=True,
            help="Lower is better; base rate ~0.66",
        )
        st.metric(
            f"{selected.label} log loss",
            f"{selected.log_loss:.4f}",
            border=True,
            help="Lower is better; uniform 1.0986",
        )
        st.metric(
            f"{selected.label} accuracy",
            f"{selected.accuracy:.4f}",
            border=True,
            help="Majority class ~0.45",
        )
    st.caption(f"Measured on {covered} out-of-sample matches.")
    if benchmark.has_market:
        st.caption(
            "The models, base rate and closing market are all scored on the same "
            f"{benchmark.market_matches} matches, so the comparison is fair."
        )

    comparison = benchmark.comparison()
    st.dataframe(
        comparison,
        hide_index=True,
        alt="Comparison of each model, the base rate and the closing market",
        column_config={
            column: st.column_config.NumberColumn(format="%.4f")
            for column in comparison.columns
            if column not in {"label", "n_matches"}
        },
    )

    outcome = st.segmented_control(
        "Calibration detail",
        options=list(OUTCOME_LABELS),
        default="H",
        format_func=lambda key: OUTCOME_LABELS[key],
    )
    if outcome:
        bins = _calibration_bins(selected, outcome)
        if bins.empty:
            st.caption("No populated probability bins for this outcome.")
        else:
            st.line_chart(
                _reliability_frame(bins),
                x="Bin",
                y="Probability",
                color="Series",
                alt=f"Reliability diagram for the {OUTCOME_LABELS[outcome].lower()}, predicted versus observed",
            )
            st.caption(
                "A well-calibrated model tracks the diagonal: over a bin where it predicted "
                f"X on average, the {OUTCOME_LABELS[outcome].lower()} occurred X of the time. "
                "Bins with few matches are noisy; counts are shown below."
            )
            st.dataframe(
                bins,
                hide_index=True,
                alt="Reliability bin probabilities and match counts",
                column_config={
                    "Predicted": st.column_config.NumberColumn(format="percent"),
                    "Observed": st.column_config.NumberColumn(format="percent"),
                    "Matches": st.column_config.NumberColumn(format="%d"),
                },
            )
    return selected


def _secret(name: str) -> str | None:
    """Read a deployment secret, falling back to the process environment.

    Args:
        name: Secret name.

    Returns:
        The secret value, or ``None`` when it is not configured. Streamlit
        Community Cloud exposes secrets through ``st.secrets``; a local ``.env``
        or shell variable is honoured too, so the same code path is testable
        offline.
    """
    try:
        if name in st.secrets:
            value = st.secrets[name]
            return str(value) if value else None
    except StreamlitSecretNotFoundError:
        pass
    return os.environ.get(name) or None


def _public_deploy() -> bool:
    """Report whether the app is running as a publicly reachable deployment.

    Returns:
        True when the public-deploy flag is set to anything other than an
        explicit falsey value.
    """
    value = (_secret(PUBLIC_DEPLOY_FLAG) or "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _read_database(database_path: Path) -> pd.DataFrame:
    """Load stored matches, tolerating a database that does not exist yet.

    Args:
        database_path: Path to the SQLite file.

    Returns:
        Canonical match frame, empty when nothing is stored.
    """
    if not database_path.exists():
        return pd.DataFrame()
    return _load_matches(str(database_path), database_path.stat().st_mtime)


@st.cache_resource(show_spinner=False)
def _bootstrap_public_data(database_path: str, api_key: str, attempt: int) -> PublicDataStatus:
    """Populate an empty deployment with licence-compliant data, once.

    Args:
        database_path: Path the store must be written to. Part of the cache key,
            so pointing the app at a different directory re-runs the load.
        api_key: football-data.org token, or an empty string.
        attempt: Retry counter from session state; bumping it busts the cache so
            a transient failure can be retried without restarting the app.

    Returns:
        The bootstrap status. A failure is returned rather than raised so the
        app can explain it without a traceback.

    Note:
        The database is written to the ephemeral container disk, so a restart
        re-runs this. It is deliberately cheap: a handful of API calls, well
        inside the free tier's ten-per-minute budget.
    """
    try:
        os.environ.setdefault("PREDICT_FOOTBALL_ALLOW_NETWORK", "1")
        return ensure_public_data(Settings.from_env(), api_key=api_key or None)
    except (BootstrapError, LicenceViolation) as error:
        return PublicDataStatus(
            ready=False,
            provider=PUBLIC_PROVIDER,
            league_key=PUBLIC_LEAGUE,
            message=str(error),
        )


def _render_bootstrap_help(status: PublicDataStatus) -> None:
    """Explain why the first-run data load did not finish, and offer a retry.

    Args:
        status: The failed bootstrap status.
    """
    st.warning(f"This deployment could not load its match data.\n\n{status.message}")
    if not _secret(API_KEY_SECRET):
        st.info(
            "Add a football-data.org token as the **FOOTBALL_DATA_ORG_API_KEY** secret "
            "(free at https://www.football-data.org/client/register), then retry."
        )
    st.caption((status.attribution or (attribution_for(PUBLIC_PROVIDER),))[0])
    if st.button("Retry loading data", key="bootstrap_retry", icon=":material/refresh:"):
        st.session_state["bootstrap_attempt"] = int(st.session_state.get("bootstrap_attempt", 0)) + 1
        st.rerun()


def _assert_publicly_servable(matches: pd.DataFrame) -> None:
    """Raise if any stored source may not be served from a public app.

    Args:
        matches: Full match frame.

    Raises:
        LicenceViolation: If a stored source's licence forbids public serving.
        KeyError: If a stored source has never had its licence reviewed.
    """
    if "source" not in matches.columns:
        return
    for source in sorted(set(matches["source"].dropna().astype(str))):
        assert_can_serve_publicly(source)


def _render_footer() -> None:
    """Render the data-provenance note."""
    st.divider()
    if _public_deploy():
        st.caption(
            f"{attribution_for(PUBLIC_PROVIDER)}. Delayed scores only; bookmaker odds and lineups "
            "are not in the free tier. Probabilities are estimates, not guarantees."
        )
    else:
        st.caption(
            "Data: football-data.co.uk history and odds (personal/internal use), plus optional live "
            "result sources. Raw provider payloads are never redistributed. Probabilities are estimates, "
            "not guarantees."
        )


def main() -> None:
    """Render the app."""
    st.set_page_config(page_title="predict_football", layout="wide")
    _render_header()

    settings = Settings.from_env()
    database_path = settings.database_path
    matches = _read_database(database_path)

    if matches.empty and _public_deploy():
        with st.spinner("Loading this deployment's match data from football-data.org..."):
            status = _bootstrap_public_data(
                str(database_path),
                _secret(API_KEY_SECRET) or "",
                int(st.session_state.get("bootstrap_attempt", 0)),
            )
        if status.ready:
            matches = _read_database(database_path)
        else:
            _render_bootstrap_help(status)
            _render_footer()
            return

    if matches is None or matches.empty:
        st.warning(
            "No matches are stored yet. Fetch some history first, for example:\n\n"
            "```\npython scripts/download_data.py --league ENG_PL --from 2018 --to 2025\n```"
        )
        _render_footer()
        return

    if _public_deploy():
        try:
            _assert_publicly_servable(matches)
        except (LicenceViolation, KeyError) as error:
            st.error(
                "This deployment holds data that may not be served from a public app, so it has "
                f"stopped rather than breach the source licence. {error}"
            )
            _render_footer()
            return

    league_options = _league_options(matches)
    controls = _render_sidebar(league_options)
    _render_day_predictor(matches, controls)

    run = st.session_state.get("run")
    if not run:
        st.info(
            "Press **Run forecast** in the sidebar for upcoming fixtures and the "
            "out-of-sample diagnostics."
        )
        _render_footer()
        return

    odds = _load_odds(str(database_path), database_path.stat().st_mtime)
    _render_fixtures(matches, run, odds)
    _render_footer()


main()
