# predict_football

A football probability and analysis system. It ingests match results, bookmaker
odds and open event data, stores them in a normalised database, and produces
calibrated probabilities for match outcomes.

The design goal is **traceability**: every probability shown to a user is
computed from data that was genuinely available at the moment the prediction was
made, and every reported metric states how many matches it was measured on.

> **Status: baseline + challenger + live ingest + app.** Ingest is complete.
> Phase 2 adds leak-free engineered features, a Dixon-Coles/Poisson model, a
> hyperparameter sweep, a post-hoc calibration step, and a chronological
> walk-forward backtest scored with Brier, log loss, accuracy and calibration;
> the model is compared honestly against the closing market. Phase 3 adds two
> live adapters (football-data.org free tier and API-Football) and a
> network-gated daily poller. Phase 4 adds a leakage-safe inference path and a
> Streamlit app that shows forecasts beside their sample sizes and benchmarks.
> Phase 5 spends the engineered features in a gradient-boosted challenger and
> benchmarks it against the baseline on the identical matches. A head-to-head
> `benchmark` command now reports both models, the base rate and the closing
> market on one sample, with a per-season breakdown; on the stored Premier League
> history the challenger does not yet beat the baseline or even the base rate,
> and that negative result is reported rather than hidden. The app can also predict
> any named matchup -- two clubs and a date -- without ingesting a fixture first.

## Requirements

- Python 3.10
- No paid API keys required. The default data sources are free.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Verify the environment

```powershell
python tools/smoke_check.py
python -m pytest -q
python -m ruff check .
```

`smoke_check.py` confirms every pinned dependency is importable and numerically
functional. A green import is not evidence a stack works, so it fits a model and
computes a metric rather than just importing.

## Fetch data

Network access is off unless you opt in, so a scheduled run cannot silently hit
the network or block on a timeout.

```powershell
$env:PREDICT_FOOTBALL_ALLOW_NETWORK = "1"
python scripts/download_data.py --league ENG_PL --season 2324
```

Omit `--season` and pass `--from`/`--to` for a range of seasons:

```powershell
python scripts/download_data.py --league ENG_PL --from 2018 --to 2025
```

Inspect what was stored:

```powershell
python scripts/download_data.py --status
```

Once a season is cached, later runs are offline and reproducible.

## Poll a live source

The daily job tops up the current season from a live provider. Because rows are
stored under the cross-provider `match_id`, polling is idempotent: a fixture that
has already been played is rewritten in place, and a newly played one gains its
result without creating a duplicate.

```powershell
$env:PREDICT_FOOTBALL_ALLOW_NETWORK = "1"
$env:FOOTBALL_DATA_ORG_API_KEY = "<token>"
python -m predict_football.cli poll --league ENG_PL
python -m predict_football.cli poll --provider api_football --league ESP_LA_LIGA --max-age-days 0
```

The poller refuses to run when network access is off, and it forces the cache
TTL down so a poll re-downloads instead of replaying the research cache. The
licence for the chosen provider is printed on every run.

`football-data.org` and `API-Football` are both live *result* sources: the free
tiers carry no odds and no xG, so the closing-market comparison still comes from
`football-data.co.uk`. `API-Football`'s terms could not be read, so unlike
`football-data.org` it is marked as not safe to serve from a public app.

## Features and modelling

`features.py` builds each fixture's features from matches that kicked off
strictly before it. A match never sees its own result: rolling form sums are
taken from a shifted cumulative total, and the league-average fallback for a
club's first appearance is computed from earlier matches only, so it is left
`NULL` for the very first match rather than leaked. Every engineered column is
registered as `PRE_MATCH` in the schema, and `assert_no_leakage()` refuses any
feature that is not.

`models/dixon_coles.py` is a Dixon-Coles Poisson model with a low-score
correction, fitted by maximum likelihood with optional ridge shrinkage on team
ratings. Its `outcome_probabilities` returns the home/draw/away distribution
used everywhere else. When a lopsided fixture is extrapolated far beyond the
fitted data, `rho` is shrunk per fixture so the low-score correction stays a
probability instead of aborting the run.

`models/feature_model.py` is a small gradient-boosted classifier over the same
engineered features. It is the challenger, not the default: it has no
expected-goals interpretation and is deliberately kept shallow so a few thousand
matches cannot be memorised. Its forecasts carry no `lambda_*` columns, and the
inference path leaves those null rather than inventing them. Training features
are built from strictly earlier matches, and a fixture's features are re-derived
from the fitted history, so a match can never be described using its own result.

`benchmark.py` scores several models on one timeline and then re-scores all of
them -- plus the base rate and, when prices exist, the de-vigged closing market --
on the **intersection** of matches every model covered. That restriction is the
point: comparing a model on its own predictions to a market priced on a different
set is how a favourite gets manufactured.

`tuning.py` sweeps recency half-life and ridge penalty with the same chronological
walk-forward used for final scoring, and returns the whole table. The best row is
out of sample, and the runner-up gap is printed because a win by 0.0001 Brier is
noise.

`calibration.py` fits a multinomial logistic recalibration on an expanding window
of earlier matchdays. It is deliberately fitted only on strictly earlier data, so
the calibrated probabilities are as honest as the raw ones. Whether it helps is
an empirical question and is reported either way.

## Run a backtest

```powershell
python -m predict_football.cli backtest --league ENG_PL
python -m predict_football.cli backtest --league ENG_PL --min-train 380 --refit-every 5
python -m predict_football.cli backtest --league ENG_PL --half-life 240 --ridge 0.1 --calibrate
python -m predict_football.cli backtest --league ENG_PL --no-market
```

The walk is strictly chronological: the model is refit on an expanding window of
earlier matches and predicts only later ones. Every reported score states the
number of matches it was measured on, and the closing market is scored on the
same matches so the comparison is fair. With `--calibrate` the raw model, the
calibrated model and the market are all scored on exactly the subset where a
calibration map was available.

## Benchmark models against each other

```powershell
python -m predict_football.cli benchmark --league ENG_PL
python -m predict_football.cli benchmark --league ENG_PL --models dixon_coles,lightgbm --refit-every 20
python -m predict_football.cli benchmark --league ENG_PL --no-market
```

`benchmark.py` runs every model through the same walk-forward and then re-scores
all of them, the base rate and the de-vigged closing market on the intersection
of matches they all covered. The command prints each report, one comparison table
and a per-season Brier breakdown, so a good average cannot hide a bad season. The
stored Premier League history has around 835 prediction dates and only refits
every 20 of them by default, which keeps a full two-model comparison to a few
minutes; each extra model is one more walk-forward. `--no-market` skips the
closing-odds benchmark when no prices are stored.

### What the challenger actually scores (English Premier League)

The full **2660** out-of-sample matches (2019/20 to 2025/26; 2018/19 is the
380-match warm-up), refitting every 20 prediction dates. That is the whole
prediction set, not the 2360-match calibration subset used in the baseline table
above, so the two tables are not directly comparable. All four rows here are
scored on the identical matches, and the stored closing prices cover all of them:

| Forecast | Brier | Log loss | Accuracy | Calibration error (H/D/A) |
| --- | --- | --- | --- | --- |
| Dixon-Coles (default) | 0.5966 | 1.0009 | 0.5079 | 0.021 / 0.008 / 0.025 |
| LightGBM | 0.6507 | 1.1037 | 0.4805 | 0.071 / 0.092 / 0.081 |
| Base rate | 0.6471 | 1.0690 | 0.4342 | 0.000 / 0.000 / 0.000 |
| Closing market (de-vigged) | 0.5717 | 0.9639 | 0.5496 | 0.019 / 0.004 / 0.014 |

The honest reading: **the gradient-boosted challenger has no demonstrated skill
here.** It is worse than the base rate and worse than Dixon-Coles on every metric,
and it is the least calibrated row in the table. Dixon-Coles was better in all
seven scored seasons (2019/20 to 2025/26). That is a real result, not a reason to
delete the challenger: the tree model is deliberately shallow and untuned, and the
point of the benchmark is that a challenger has to earn its place rather than be
assumed better because it is more modern. The Dixon-Coles row here uses the
default ridge with no recency half-life, so it is weaker than the tuned row in the
table above.

## Tune hyperparameters

```powershell
python -m predict_football.cli tune --league ENG_PL --half-lives none,60,120,240 --ridges 0.02,0.1
```

Every candidate is fitted only on earlier matches, so the winning row is an
out-of-sample result. Longer recency half-lives won on the Premier League; an
aggressive 60-match half-life was clearly worse, which is itself a finding worth
recording.

### What the baseline actually scores (English Premier League)

Measured out of sample over **2660 matches** (2019/20 to 2025/26; 2018/19 was
the 380-match warm-up). The rows below are scored on the **same 2360 matches**
where a walk-forward calibration map existed, so raw, calibrated and market are
directly comparable:

| Forecast | Brier | Log loss | Accuracy |
| --- | --- | --- | --- |
| Base rate | 0.6469 | 1.0686 | 0.4331 |
| Dixon-Coles (tuned) | 0.5873 | 0.9871 | 0.5250 |
| Dixon-Coles (tuned + calibrated) | 0.5880 | 0.9884 | 0.5288 |
| Closing market (de-vigged) | 0.5708 | 0.9618 | 0.5525 |

The sweep (half-life 240, ridge 0.1) improved the honest baseline from about
0.600 to about 0.587 Brier. Calibration did **not** improve Brier here (0.5873 to
0.5880) and is therefore reported, not relied upon; it slightly improved accuracy
and rebalanced the per-outcome calibration errors. The model still loses to the
closing market, by about 0.017 Brier. On the full 2660 predictions the winning
sweep row scored 0.5867 Brier, and the runner-up was level to four decimals, so
the tuning win is modest rather than decisive. The 2020/21 season was played
largely without crowds, which is worth keeping in mind when reading any
per-season figure.

## Run the app

```powershell
streamlit run streamlit_app.py
```

Pick a competition and press **Run forecast** to load the upcoming-fixture and
diagnostics sections (the matchday predictor is always visible). The **Model**
control switches the upcoming-fixture forecasts between the Dixon-Coles baseline
and the gradient-boosted challenger; the diagnostics always compare both. The
app shows three things:

- **Predict a matchday**, a date picker that lists every match stored for the
  chosen day and forecasts each one with a model trained only on matches that
  kicked off strictly before it. Picking a date that has already been played
  scores the model honestly against the result; picking a future date is a
  genuine forecast, and the day's fixtures need not be the next ones ingested.
  The selected match shows home/draw/away probabilities, a bar chart of the same
  numbers and, for Dixon-Coles, a scoreline heat map of the joint goal
  distribution the probabilities are summed from, plus expected goals and the
  number of matches behind the fit.
- **Upcoming fixtures**, each forecast by a model trained only on matches that
  kicked off strictly before it. A fixture without enough prior history is left
  blank, with the shortfall stated, rather than given a confident number from a
  thin fit. A club the training window has never seen is flagged, because the
  model falls back to a league-average rating for it.
- **Model diagnostics**, the same out-of-sample walk-forward the backtest runs,
  with every model, the base rate and the closing market scored on the identical
  matches. Brier, log loss and accuracy all carry their sample size, and a
  reliability chart shows predicted against observed frequency.

`inference.py` is the boundary that makes the forecasts honest: it is the only
place that decides which matches are allowed to inform a prediction, and it
applies the strict `<` comparison on dates that the backtest uses. Everything it
computes is cached, so the walk-forward evaluation runs once per data and
hyperparameter combination.

## Deploy (Streamlit Community Cloud)

The app is ready for Streamlit Community Cloud. Add these as the app's
**Secrets** (copy `.streamlit/secrets.toml.example` for a local run):

| Secret | Value |
| --- | --- |
| `FOOTBALL_DATA_ORG_API_KEY` | a free token from [football-data.org](https://www.football-data.org/client/register) |
| `PREDICT_FOOTBALL_PUBLIC_DEPLOY` | `1` |
| `PREDICT_FOOTBALL_ALLOW_NETWORK` | `1` |

What happens on a cold start: a cloud checkout has no SQLite store (it is
gitignored), so with the public flag set the app fetches the three most recent
completed seasons plus the season in progress from **football-data.org** into
the container's ephemeral disk, then serves them. That is the only source it
uses in this mode, because football-data.co.uk — the source behind a local
checkout — grants no right to serve its data from a public app.
`data/bootstrap.py` checks the licence registry before fetching, and the app
re-checks it on whatever it is about to show, so a deployment cannot silently
serve restricted data.

Consequences worth knowing before you deploy:

- The free tier is delayed scores only. There are no bookmaker odds, so the
  closing-market benchmark row is absent, and there are no lineups.
- The container disk is ephemeral: a reboot re-fetches (a handful of API calls,
  inside the ten-per-minute budget) and the walk-forward diagnostics recompute.
- `packages.txt` installs `libgomp1`, which LightGBM's wheel needs on the slim
  base image.
- Streamlit Cloud reads `requirements.txt` (pinned) and `runtime.txt`
  (`python-3.10`), so the deployed stack matches the local one exactly.

To push and connect: commit the tree, push to your GitHub remote, then create the
app at [share.streamlit.io](https://share.streamlit.io) pointing at
`streamlit_app.py`.

## Layout

| Path | Purpose |
| --- | --- |
| `src/predict_football/config/` | Settings, competition registry, licence terms |
| `src/predict_football/data/` | Schema, cleaning, identifiers, team resolver, cache |
| `src/predict_football/data/providers/` | One module per source, registered by name |
| `src/predict_football/data/bootstrap.py` | Licence-guarded first-run load for public deploys |
| `src/predict_football/data/repository.py` | SQLite storage behind a single interface |
| `src/predict_football/features.py` | Leak-free pre-match feature engineering |
| `src/predict_football/models/` | Dixon-Coles baseline and gradient-boosted challenger |
| `src/predict_football/benchmark.py` | Same-matches comparison of several models |
| `src/predict_football/tuning.py` | Out-of-sample hyperparameter sweep |
| `src/predict_football/calibration.py` | Walk-forward probability recalibration |
| `src/predict_football/evaluation.py` | Brier, log loss, accuracy, calibration |
| `src/predict_football/backtest.py` | Chronological walk-forward evaluation |
| `src/predict_football/inference.py` | Leakage-safe forecasting of unplayed fixtures |
| `streamlit_app.py` | Streamlit front end: forecasts, coverage and diagnostics |
| `scripts/` | Command-line entry points |
| `tools/` | Smoke checks |
| `docs/DATA_SOURCES.md` | Source-by-source licence and access review |

## What happens during ingest

1. **Download.** The provider fetches a season document through `RawCache`, which
   stores the response verbatim with a fetch timestamp. Results and odds come
   from one file, so they share one cache key and the document is downloaded
   once.
2. **Map.** The payload is mapped onto the canonical schema in `data/schema.py`
   and coerced to canonical dtypes. Provider-specific extras such as
   `statsbomb_match_id` are preserved for child lookups.
3. **Resolve teams.** Names are resolved to canonical form through the registry
   in `data/teams.py`. An unrecognised name is stored as-is and flagged, so the
   gap stays visible instead of being merged away.
4. **Clean.** Validation runs, with two different severities: a few impossible
   scorelines are dropped with an error log, while more than a fifth signals a
   systematic parsing problem and raises. Self-matches, which indicate a broken
   join, are handled separately from bad scores.
5. **Store.** Rows are written to SQLite through `MatchRepository`. A match is
   identified by a hash of `league|date|home|away`, so the same fixture ingested
   from two providers collapses to one row rather than two.

## Data sources and licensing

Read `docs/DATA_SOURCES.md` before adding or redistributing any data. The short
version:

- **football-data.co.uk** publishes no licence at all. Richest free source of
  history and odds, but the absence of a licence means nothing grants us
  redistribution rights. Treat as personal and internal use only.
- **football-data.org** free tier allows single-application use with attribution
  ("Football data provided by the Football-Data.org API") but forbids commercial
  use; its live scores and lineups are paid. Delayed results only.
- **API-Football (API-Sports)** free tier is 100 requests/day and its terms could
  not be read, so the licence is treated as unknown and public serving is off.
- **StatsBomb open data** forbids redistribution and any publicly served or
  commercial derived analysis, and requires their logo on derived visualisations.
- **openfootball** is CC0.
- **Wyscout soccer-logs** is CC BY 4.0.

Only football-data.org may be served from the public deployment.
`assert_can_serve_publicly()` rejects every other source, which is why the
deployed bootstrap cannot fall back to the richer football-data.co.uk history.

Raw provider payloads are gitignored and are never committed. The licence terms
are also encoded in `config/licences.py` and printed on ingest, so the warning
travels with the data.

## Data integrity

The canonical schema classifies every column as pre-match, halftime, post-match
or target. `assert_no_leakage()` enforces this: a pre-match model built on
`home_xg` fails immediately rather than silently producing an impressive and
meaningless accuracy.

Unknown values are stored as `NULL`, never as `0`, `False` or "not started". A
provider that publishes no captain flag yields an unknown captain, not a
non-captain.

## Licence

Code is MIT licensed. The data is not: see `docs/DATA_SOURCES.md`.
