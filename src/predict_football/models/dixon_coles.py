"""Dixon-Coles model for association football match scores.

The model treats each team's goals as an independent Poisson variate whose mean
depends on the club's attack, the opponent's defence, and whether the club is at
home. Independent Poisson has a known defect for football: it badly
under-predicts 0-0 and 1-1 while over-predicting 1-2 and 2-1, because the real
score distribution is more concentrated in the low scores than independence
implies. Dixon and Coles fix this with a four-cell correction to the joint
probability of the low scores, controlled by a single parameter ``rho``.

Fitting maximises the Poisson log-likelihood with that correction, plus a ridge
penalty on the team parameters. The penalty is not cosmetic: attack and defence
are only identified relative to one another, and a promoted club with three
matches would otherwise receive an extreme rating from a maximum-likelihood fit.

All estimation here is in-sample on the matches passed to :meth:`fit`. Nothing in
this module knows about dates; keeping the time ordering in the caller is what
makes leakage auditable rather than hidden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

module_logger = logging.getLogger(__name__)

#: Outcome labels in the canonical order used by every probability column.
OUTCOMES: tuple[str, str, str] = ("H", "D", "A")

#: Probability column names, in the same order as :data:`OUTCOMES`.
PROBABILITY_COLUMNS: tuple[str, str, str] = ("prob_home", "prob_draw", "prob_away")

_REQUIRED_COLUMNS = frozenset({"home_team", "away_team", "home_goals", "away_goals"})

#: Largest number of goals any side can be given in the score matrix. Poisson
#: tails beyond this are negligible, and the matrix is renormalised so the three
#: outcome probabilities sum to one exactly.
MAX_GOALS = 10

#: rho must keep the low-score correction positive. Beyond roughly +/-0.2 the
#: correction distorts more than it fixes, so the bound is a modelling choice
#: rather than a numerical convenience.
_RHO_BOUNDS = (-0.2, 0.2)

#: Margin kept between every low-score correction factor and zero when rho is
#: shrunk for an extrapolating fixture. It is a numerical guard, not a modelling
#: parameter.
_RHO_EPSILON = 1e-9


def _centre_ratings(attack: np.ndarray, defence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix the attack/defence gauge without changing any prediction.

    Attack and defence enter the model only through ``attack[home] -
    defence[away]``, so adding the same constant to both leaves every prediction
    identical. Subtracting each array's own mean would apply two different
    shifts and quietly move the model off the fitted optimum, so the *same*
    constant - the mean attack - is removed from both. This still gives
    ``mean(attack) == 0`` for interpretability.

    Args:
        attack: Fitted attack strengths, one per team.
        defence: Fitted defence strengths, one per team.

    Returns:
        Tuple of centred attack and defence arrays whose pairwise differences
        are unchanged.
    """
    shift = float(attack.mean())
    return attack - shift, defence - shift


def _valid_rho(lam: np.ndarray | float, mu: np.ndarray | float, rho: np.ndarray | float) -> np.ndarray:
    """Shrink rho to the region where the low-score correction is positive.

    A model fitted on ordinary fixtures can be asked to price a lopsided one
    with much larger expected goals. The fitted rho can then make a correction
    factor negative, which is outside the region where the Dixon-Coles correction
    is a probability. Rather than fail the whole batch, rho is pulled back to the
    largest value that keeps every factor above :data:`_RHO_EPSILON`.

    Args:
        lam: Expected home goals, per fixture.
        mu: Expected away goals, per fixture.
        rho: Fitted dependence parameter.

    Returns:
        rho clipped per fixture so that ``1 - lam*mu*rho``, ``1 + lam*rho``,
        ``1 + mu*rho`` and ``1 - rho`` are all strictly positive.
    """
    lam = np.asarray(lam, dtype="float64")
    mu = np.asarray(mu, dtype="float64")
    rho = np.asarray(rho, dtype="float64")

    positive_cap = (1.0 - _RHO_EPSILON) / np.maximum(lam * mu, _RHO_EPSILON)
    negative_floor = -(1.0 - _RHO_EPSILON) / np.maximum(np.maximum(lam, mu), _RHO_EPSILON)
    return np.where(rho > 0.0, np.minimum(rho, positive_cap), np.maximum(rho, negative_floor))


def _tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float | np.ndarray,
) -> np.ndarray:
    """Dixon-Coles low-score correction factor.

    Args:
        home_goals: Home goals as integer array.
        away_goals: Away goals as integer array.
        lam: Expected home goals.
        mu: Expected away goals.
        rho: Dependence parameter.

    Returns:
        The multiplicative correction for each observation. One everywhere
        except the four low-score cells.
    """
    factor = np.ones(home_goals.shape, dtype="float64")
    factor = np.where((home_goals == 0) & (away_goals == 0), 1.0 - lam * mu * rho, factor)
    factor = np.where((home_goals == 0) & (away_goals == 1), 1.0 + lam * rho, factor)
    factor = np.where((home_goals == 1) & (away_goals == 0), 1.0 + mu * rho, factor)
    return np.where((home_goals == 1) & (away_goals == 1), 1.0 - rho, factor)


def _score_matrix(lam: float, mu: float, rho: float) -> np.ndarray:
    """Joint probability of every scoreline up to :data:`MAX_GOALS`.

    Args:
        lam: Expected home goals.
        mu: Expected away goals.
        rho: Dependence parameter.

    Returns:
        Array of shape ``(MAX_GOALS + 1, MAX_GOALS + 1)`` holding the joint
        probability of each scoreline, corrected and renormalised to sum to one.
        rho is first shrunk with :func:`_valid_rho`, so an extrapolating fixture
        yields a valid distribution instead of failing.
    """
    grid = np.arange(MAX_GOALS + 1)
    home, away = np.meshgrid(grid, grid, indexing="ij")
    independent = poisson.pmf(home, lam) * poisson.pmf(away, mu)
    corrected = independent * _tau(home, away, lam, mu, float(_valid_rho(lam, mu, rho)))

    # _valid_rho keeps every factor strictly positive; the clip only removes
    # rounding noise so the normalisation below is always well defined.
    corrected = np.clip(corrected, 0.0, None)
    return corrected / corrected.sum()


def score_matrix(lam: float, mu: float, rho: float) -> pd.DataFrame:
    """Joint scoreline distribution for one fixture, as a labelled table.

    This is the same matrix :meth:`DixonColesModel.outcome_probabilities` sums
    over, exposed so a caller can show *why* a probability is what it is: the
    likely scorelines behind a home, draw or away outcome. It is a free function
    rather than a method so it can be applied to the expected goals reported by a
    forecast without refitting the model.

    Args:
        lam: Expected home goals.
        mu: Expected away goals.
        rho: Fitted low-score dependence parameter.

    Returns:
        Frame indexed by home goals and columned by away goals, holding the
        corrected joint probability of each scoreline up to :data:`MAX_GOALS`.
        The cells sum to one.
    """
    matrix = _score_matrix(float(lam), float(mu), float(rho))
    return pd.DataFrame(
        matrix,
        index=pd.Index(range(matrix.shape[0]), name="home_goals"),
        columns=pd.Index(range(matrix.shape[1]), name="away_goals"),
    )


@dataclass
class DixonColesModel:
    """Poisson attack/defence model with a Dixon-Coles low-score correction.

    Attributes:
        teams: Teams the model knows, sorted for stable output.
        intercept: Log base rate of goals per team per match.
        home_advantage: Log multiplier applied to the home side's attack.
        rho: Low-score dependence parameter.
        ridge: Penalty strength on attack and defence.
        half_life_matches: If set, weight each match by ``0.5 ** (age / half_life)``
            so recent matches count for more. ``None`` weights every match equally.
        max_goals: Largest goals per side in the score matrix.
        converged: Whether the optimiser reported success.
        n_matches: Matches used for fitting.
        objective: Penalised negative log-likelihood at the optimum.
    """

    teams: list[str] = field(default_factory=list)
    intercept: float = 0.0
    home_advantage: float = 0.0
    rho: float = 0.0
    ridge: float = 0.05
    half_life_matches: float | None = None
    max_goals: int = MAX_GOALS
    converged: bool = False
    n_matches: int = 0
    objective: float = float("nan")

    _attack: dict[str, float] = field(default_factory=dict, repr=False)
    _defence: dict[str, float] = field(default_factory=dict, repr=False)

    # -- internals ---------------------------------------------------------

    def _teams(self) -> list[str]:
        """Return the team list in a stable order.

        Returns:
            Teams sorted alphabetically, so parameter vectors do not depend on
            the order rows happened to arrive in.
        """
        return sorted(self._attack)

    def _unpack(self, theta: np.ndarray, teams: list[str]) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        """Split a flat parameter vector into its parts.

        Args:
            theta: Packed parameters.
            teams: Team order the arrays were packed against.

        Returns:
            Tuple of intercept, home advantage, rho, attack array and defence
            array.
        """
        n = len(teams)
        intercept = float(theta[0])
        home_advantage = float(theta[1])
        rho = float(theta[2])
        attack = theta[3 : 3 + n]
        defence = theta[3 + n : 3 + 2 * n]
        return intercept, home_advantage, rho, attack, defence

    def _lam_mu(
        self,
        home_index: np.ndarray,
        away_index: np.ndarray,
        intercept: float,
        home_advantage: float,
        attack: np.ndarray,
        defence: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expected goals for a set of fixtures.

        Args:
            home_index: Row indices into ``attack`` for the home sides.
            away_index: Row indices into ``defence`` for the away sides.
            intercept: Base log rate.
            home_advantage: Home log multiplier.
            attack: Attack parameters.
            defence: Defence parameters.

        Returns:
            Tuple of expected home goals and expected away goals.
        """
        lam = np.exp(intercept + home_advantage + attack[home_index] - defence[away_index])
        mu = np.exp(intercept + attack[away_index] - defence[home_index])
        return lam, mu

    # -- fitting -----------------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> DixonColesModel:
        """Fit attack, defence, home advantage and rho by maximum likelihood.

        Args:
            matches: Frame with home_team, away_team, home_goals and away_goals.
                Rows must be matches whose results are already known.

        Returns:
            The fitted model, for chaining.

        Raises:
            ValueError: If required columns are missing, if fewer than two
                teams are present, or if any team is listed on both sides of the
                same fixture.
        """
        missing = _REQUIRED_COLUMNS - set(matches.columns)
        if missing:
            raise ValueError(f"DixonColesModel.fit: frame is missing {sorted(missing)}")

        played = matches.dropna(subset=["home_goals", "away_goals"])
        if played.empty:
            raise ValueError("DixonColesModel.fit: no row has a result")

        home_goals = played["home_goals"].to_numpy(dtype="float64")
        away_goals = played["away_goals"].to_numpy(dtype="float64")

        teams = sorted(set(played["home_team"]) | set(played["away_team"]))
        if len(teams) < 2:
            raise ValueError("DixonColesModel.fit: need at least two teams to identify attack and defence")

        lookup = {team: i for i, team in enumerate(teams)}
        home_index = played["home_team"].map(lookup).to_numpy(dtype="int64")
        away_index = played["away_team"].map(lookup).to_numpy(dtype="int64")

        weights = self._match_weights(len(played))
        n = len(teams)

        def objective(theta: np.ndarray) -> float:
            """Return the penalised negative log-likelihood."""
            intercept, home_advantage, rho, attack, defence = self._unpack(theta, teams)
            try:
                lam, mu = self._lam_mu(home_index, away_index, intercept, home_advantage, attack, defence)
                if not np.isfinite(lam).all() or not np.isfinite(mu).all():
                    return 1e12
                factor = _tau(home_goals.astype("int64"), away_goals.astype("int64"), lam, mu, rho)
                if (factor <= 0).any():
                    # rho has left the region where the correction is a
                    # probability. Penalise rather than take a log of a negative.
                    return 1e12
                log_likelihood = np.sum(
                    weights
                    * (
                        np.log(factor)
                        + home_goals * np.log(lam)
                        - lam
                        + away_goals * np.log(mu)
                        - mu
                    )
                )
            except FloatingPointError:
                return 1e12

            penalty = self.ridge * (np.sum(attack**2) + np.sum(defence**2))
            return float(-log_likelihood + penalty)

        start = np.zeros(2 * n + 3, dtype="float64")
        start[0] = float(np.log(max(played[["home_goals", "away_goals"]].to_numpy().mean(), 0.05)))
        start[1] = 0.25
        start[2] = -0.05

        bounds = [(-2.0, 2.0), (-1.0, 1.0), _RHO_BOUNDS] + [(-3.0, 3.0)] * (2 * n)
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)

        intercept, home_advantage, rho, attack, defence = self._unpack(result.x, teams)
        # Attack and defence are only identified up to a shared constant. Centring
        # them keeps the numbers interpretable without changing any prediction.
        attack, defence = _centre_ratings(attack, defence)

        self._attack = dict(zip(teams, attack, strict=True))
        self._defence = dict(zip(teams, defence, strict=True))
        self.teams = teams
        self.intercept = intercept
        self.home_advantage = home_advantage
        self.rho = rho
        self.converged = bool(result.success)
        self.n_matches = len(played)
        self.objective = float(result.fun)

        module_logger.info(
            "Fitted Dixon-Coles on %d matches across %d teams (converged=%s, intercept=%.3f, home=%.3f, rho=%.4f)",
            self.n_matches,
            len(self.teams),
            self.converged,
            self.intercept,
            self.home_advantage,
            self.rho,
        )
        return self

    def _match_weights(self, n: int) -> np.ndarray:
        """Weight each match by recency.

        Args:
            n: Number of matches, ordered from oldest to newest.

        Returns:
            One weight per match. All ones when time weighting is disabled.
        """
        if not self.half_life_matches:
            return np.ones(n, dtype="float64")
        age = np.arange(n - 1, -1, -1, dtype="float64")
        return np.power(0.5, age / float(self.half_life_matches))

    # -- prediction --------------------------------------------------------

    def expected_goals(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return expected goals for each fixture.

        Args:
            matches: Frame with home_team and away_team.

        Returns:
            Frame with the fixture identifiers plus ``lambda_home`` and
            ``lambda_away``. A team absent from the fitted set is given the
            league-average rating of zero on both attack and defence, which is
            the honest assumption: no evidence either way.
        """
        missing = {"home_team", "away_team"} - set(matches.columns)
        if missing:
            raise ValueError(f"expected_goals: frame is missing {sorted(missing)}")
        if not self.teams:
            raise ValueError("expected_goals: model has not been fitted")

        unknown = (set(matches["home_team"]) | set(matches["away_team"])) - set(self._attack)
        if unknown:
            module_logger.warning(
                "%d team(s) absent from training, given league-average ratings: %s",
                len(unknown),
                sorted(unknown),
            )

        attack = np.array([self._attack.get(team, 0.0) for team in matches["home_team"]])
        home_defence = np.array([self._defence.get(team, 0.0) for team in matches["home_team"]])
        away_attack = np.array([self._attack.get(team, 0.0) for team in matches["away_team"]])
        away_defence = np.array([self._defence.get(team, 0.0) for team in matches["away_team"]])

        lam = np.exp(self.intercept + self.home_advantage + attack - away_defence)
        mu = np.exp(self.intercept + away_attack - home_defence)

        identifiers = [
            c
            for c in ("match_id", "match_date", "season", "season_code", "home_team", "away_team")
            if c in matches.columns
        ]
        return pd.DataFrame(
            {**{c: matches[c].to_numpy() for c in identifiers}, "lambda_home": lam, "lambda_away": mu}
        )

    def outcome_probabilities(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return home, draw and away probabilities for each fixture.

        Args:
            matches: Frame with home_team and away_team.

        Returns:
            Frame with fixture identifiers, ``lambda_home``, ``lambda_away`` and
            the three probabilities, which sum to one exactly.
        """
        expected = self.expected_goals(matches)
        home = expected["lambda_home"].to_numpy()
        away = expected["lambda_away"].to_numpy()

        result = expected.copy()
        for position, column in enumerate(PROBABILITY_COLUMNS):
            values = np.empty(len(result), dtype="float64")
            for i in range(len(result)):
                matrix = _score_matrix(float(home[i]), float(away[i]), self.rho)
                values[i] = _outcome_mass(matrix, position)
            result[column] = values
        return result

    def score_matrix(self, home_team: str, away_team: str) -> pd.DataFrame:
        """Return the scoreline probability matrix for one fixture.

        Args:
            home_team: Home club.
            away_team: Away club.

        Returns:
            DataFrame indexed by away goals and columned by home goals.
        """
        expected = self.expected_goals(pd.DataFrame({"home_team": [home_team], "away_team": [away_team]}))
        matrix = _score_matrix(
            float(expected["lambda_home"].iloc[0]),
            float(expected["lambda_away"].iloc[0]),
            self.rho,
        )
        return pd.DataFrame(matrix, index=range(self.max_goals + 1), columns=range(self.max_goals + 1))

    # -- diagnostics -------------------------------------------------------

    def log_likelihood(self, matches: pd.DataFrame) -> float:
        """Return the in-sample weighted log-likelihood.

        Args:
            matches: The matches the model was fitted on.

        Returns:
            Log-likelihood. Higher is better; comparable only between models
            fitted to the same rows.
        """
        played = matches.dropna(subset=["home_goals", "away_goals"])
        expected = self.expected_goals(played)
        rho = _valid_rho(
            expected["lambda_home"].to_numpy(),
            expected["lambda_away"].to_numpy(),
            self.rho,
        )
        factor = _tau(
            played["home_goals"].to_numpy(dtype="int64"),
            played["away_goals"].to_numpy(dtype="int64"),
            expected["lambda_home"].to_numpy(),
            expected["lambda_away"].to_numpy(),
            rho,
        )
        return float(
            np.sum(
                np.log(factor)
                + played["home_goals"].to_numpy() * np.log(expected["lambda_home"].to_numpy())
                - expected["lambda_home"].to_numpy()
                + played["away_goals"].to_numpy() * np.log(expected["lambda_away"].to_numpy())
                - expected["lambda_away"].to_numpy()
            )
        )

    def team_ratings(self) -> pd.DataFrame:
        """Return the fitted attack and defence strength per team.

        Returns:
            DataFrame of team, attack and defence, ordered by attack.
        """
        return pd.DataFrame(
            {
                "team": self._teams(),
                "attack": [self._attack[t] for t in self._teams()],
                "defence": [self._defence[t] for t in self._teams()],
            }
        ).sort_values("attack", ascending=False, ignore_index=True)


def _outcome_mass(matrix: np.ndarray, position: int) -> float:
    """Sum the score matrix over one outcome.

    Args:
        matrix: Scoreline probability matrix.
        position: 0 for home, 1 for draw, 2 for away.

    Returns:
        Probability of that outcome.
    """
    home, away = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), indexing="ij")
    if position == 0:
        selected = home > away
    elif position == 1:
        selected = home == away
    else:
        selected = home < away
    return float(matrix[selected].sum())


__all__ = ["MAX_GOALS", "OUTCOMES", "PROBABILITY_COLUMNS", "DixonColesModel", "score_matrix"]
