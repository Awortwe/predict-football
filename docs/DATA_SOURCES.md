# Free & Open Football Data Sources — Verified Report

**Verified:** 2 October 2026 (all pages/APIs fetched on this date unless stated)
**Scope:** historical 1X2 + bookmaker odds, timestamped events with xG for replay, lineups/subs/player impact, and future live data — for a Premier League-first project expanding to La Liga, Champions League, AFCON and World Cup.

---

## 0. Headline table

| Source | Granularity | Shot xG | Lineups / Subs | Live | Limit | Licence verdict |
|---|---|---|---|---|---|---|
| football-data.co.uk | Match row | Match total only (26/27+) | No | No | None stated | **No licence published** — treat as all-rights-reserved |
| StatsBomb Open Data | Full event stream | **Yes, per shot** (+ freeze frames) | **Yes** (lineups + Starting XI tactics) | No | None | Custom agreement: research use, no redistribution, no commercial, logo credit required |
| Wyscout Open Data (soccer-logs) | Full event stream | No | **Yes** (lineup, bench, subs, coach, formation) | No | None | **CC BY 4.0** — commercial OK with attribution |
| football-data.org | Match + standings + scorers | No | €29/mo tier only | Free tier: **no** (delayed) | 10 calls/min | Free ToS, attribution required, no redistribution after cancellation |
| API-Football (API-Sports) | Match + events + lineups | No documented | Yes | Yes (free tier) | **100 req/day** | Terms **unverified** (site blocked) |
| openfootball | Fixture/result | No | No | No | None | **CC0 1.0** — fully open |
| SkillCorner open data | Broadcast tracking + events | No (EPV only) | Yes | No | None | **MIT** — fully open, but 10 matches |
| Metrica sample-data | Tracking + events | No | Yes | No | None | **No licence file** — all rights reserved |
| ClubElo | Club rating series | No | No | Ratings update post-match | None stated | **No licence published** |
| Understat | Team/player xG aggregates | Yes (aggregate) | No | No | robots.txt `Disallow: /` | **No licence; crawling disallowed** — do not scrape |

---

## 1. football-data.co.uk — best free historical odds, unclear rights

**Bulk URL pattern** (verified working):
```
https://www.football-data.co.uk/mmz4281/{season}/{league}.csv
```
- Season code = two-digit start year: `2526` = 2025/26, `2627` = 2026/27.
- Examples: `https://www.football-data.co.uk/mmz4281/2526/E0.csv`, `https://www.football-data.co.uk/mmz4281/2627/SP1.csv`
- League index: https://www.football-data.co.uk/data.php · Column docs: https://www.football-data.co.uk/notes.txt

**Freshness (verified)**
- Site header "Updated: 02/10/26".
- England page: files last updated **30/09/2026**.
- Spain page: files last updated **29/09/2026**.

**Coverage verified for 2026/27** (file exists, row count at fetch time, xG present?)

| Code | League | Rows | `HxG`/`AxG` |
|---|---|---|---|
| E0 | Premier League | 50 | Yes |
| E1 | Championship | 95 | Yes |
| E2 | League One | 87 | Yes |
| EC | National League | 132 | No |
| SC0 | Scottish Premiership | 42 | Yes |
| SC1 | Scottish Championship | 39 | No |
| D1 | Bundesliga | 36 | Yes |
| D2 | 2. Bundesliga | 54 | Yes |
| I1 | Serie A | 50 | Yes |
| SP1 | La Liga | 69 | Yes |
| F1 | Ligue 1 | 45 | Yes |
| N1 | Eredivisie | 63 | Yes |
| P1 | Primeira Liga | 62 | Yes |
| T1 | Turkish Süper Lig | 54 | Yes |

Historic seasons `2223`, `2324`, `2425`, `2526` for E0 do **not** contain xG — it is new from 2026/27 only.

**Columns** (`2627/E0.csv`, 114 columns; SP1 has 113 — no `Referee`):
`Div, Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, [Referee], HxG, AxG, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR` followed by bookmaker blocks — `B365*`, `PS*`/`WH*`, `Max*`, `Avg*` for 1X2 (opening and closing), `AHh` plus Asian handicap lines, and over/under 2.5 and both-teams-to-score.
- xG alignment sanity-checked on `2627/SP1.csv`: Alavés 3–0 Getafe, 18 shots, 8 on target, `HxG=1.93`, `AxG=0.24`. Values are plausible team xG totals, not odds.

**⚠️ Licensing — not resolved**
- The site only states the data is free and funded by advertising (principally bet365): *"I very much feel the content should remain free"* — https://www.football-data.co.uk/help_footballdata.php
- There is **no** published licence, terms page, or attribution requirement for the CSV files.
- The only disclaimer on the domain is for the **separate** livescore site (`livescore.football-data.co.uk`, Enetpulse data, "protected by copyright… any unauthorised use is strictly prohibited") — that does not cover the CSV, but it shows the operator's posture.
- Context: Football DataCo states it holds copyright in official league match data — https://www.football-dataco.com/
- **Practical ruling: personal research and internal modelling are fine; do not redistribute the CSVs or ship them in a public product without written permission from the operator (Joseph Buchwald).**

**Gaps:** `HxG`/`AxG` are **not documented in `notes.txt`** and no provenance is stated — treat as unverified third-party aggregate xG. No events, no per-shot xG, no lineups, no live.

---

## 2. StatsBomb Open Data — the only open per-shot xG with timestamps

**Repo:** `https://github.com/statsbomb/open-data` → now `https://github.com/hudl/open-data` (repo activity through 2026-09-07).

**Verified URL patterns** (base `https://raw.githubusercontent.com/hudl/open-data/master/data/`):
```
competitions.json
matches/{competition_id}/{season_id}.json
events/{match_id}.json
lineups/{match_id}.json
three-sixty/{match_id}.json
```

**Volume (measured from the Git tree)**
- 4,235 `events/*.json` files and 4,235 `lineups/*.json` files; **426** `three-sixty/*.json` files.
- 3,961 event files map cleanly to published `matches/{comp}/{season}.json` lists (the small gap is orphaned files — filter by the match manifest, do not glob the directory).

**Coverage**
| Competition | Season | Matches with events |
|---|---|---|
| Premier League | 2015/16 | **380 (complete)** |
| Premier League | 2003/04 | 38 (partial) |
| FIFA World Cup | 2018, 2022 | **64 / 64 (complete)** |
| AFCON | 2023 | **52 (complete)** |
| UEFA Euro | 2020, 2024 | **51 / 51 (complete)** |
| Copa América | 2024 | **32 (complete)** |
| UEFA Champions League | various | ~1 match per selected season — effectively unusable |
| La Liga / Bundesliga / Ligue 1 | recent seasons | one match per matchday (samples) |

**Event richness (verified on PL match `3754217`, 3,732 events)**
- Replay keys: `minute`, `second`, `timestamp` (`"00:01:47.549"` format), `period`, `possession`, `team`, `players`.
- Shots: `shot.statsbomb_xg` (e.g. `0.0388317`), `shot.location`, `shot.end_location`, `shot.type`, `shot.outcome`, `shot.technique`, `shot.body_part`, and often `shot.freeze_frame`.
- Lineups/tactics: `Starting XI` events carry `tactics.formation` and `tactics.lineup`; separate `lineups/{match_id}.json` per team.
- Also: passes, carries, pressures, duels, fouls, cards, goals, `Substitution`, `Player On`, `Player Off`.
- 426 selected matches have 360° freeze-frame data (e.g. AFCON 2023 360 metadata marked updated 2026-05-02).

**⚠️ Licensing — custom, restrictive, and fully readable**
Full text extracted from `LICENSE.pdf` (165,130 bytes, 5 pages): *StatsBomb Public Data User Agreement, Standard Terms — last updated 8 September 2023*, StatsBomb Services Ltd (no. 10377735), law of England & Wales.

- Preamble: "made this data freely available… aimed to be a research tool… **Any analysis or conclusions** that are created as a result of using this data **may be shared publicly** but are not necessarily the opinions or analytical insights of StatsBomb."
- **1.2.1** — "The User may not: edit, distort, distribute, reproduce, sell or in any way provide the data to any external or third party."
- **1.2.2** — "…commercially exploit the data or any analysis derived from the use of the Service."
- **1.4** — "The User is **required to credit any publication of analysis** formed from StatsBomb Data **with the StatsBomb brand logo**."
- **2.1** — Delivered via GitHub; StatsBomb "have full rights to withhold the Service at any time without prior warning."
- **§7** — All data is the property of StatsBomb.
- Registration (name + email) requested at `https://www.statsbomb.com/resource-centre`.

**Practical ruling:** excellent for research, notebooks and published analysis **with the StatsBomb logo and attribution**. You may **not** commit the JSON files to a public repo, serve them from an app, or monetise anything derived from them.

---

## 3. Wyscout Open Data (soccer-logs, Figshare) — genuinely open, commercial-friendly

**Collection:** https://figshare.com/collections/Soccer_match_event_dataset/4415000
**Licence:** **CC BY 4.0** on Matches, Events, Players, Teams, Competitions, Coaches, Referees (verified in Figshare API `license` field).
- Matches: `https://api.figshare.com/v2/articles/7770422` → `matches.zip` (645,097 bytes)
- Events: `https://api.figshare.com/v2/articles/7770599` → `events.zip` (77,323,413 bytes)

**Coverage — counted by downloading `matches.zip` (7 files, 1,941 matches total):**

| File | Competition | Matches | Date range |
|---|---|---|---|
| `matches_England.json` | Premier League | 380 | 2017-08-11 → 2018-05-13 |
| `matches_France.json` | Ligue 1 | 380 | 2017-08-04 → 2018-05-19 |
| `matches_Germany.json` | Bundesliga | 306 | 2017-08-18 → 2018-05-12 |
| `matches_Italy.json` | Serie A | 380 | 2017-08-19 → 2018-05-20 |
| `matches_Spain.json` | La Liga | 380 | 2017-08-18 → 2018-05-20 |
| `matches_European_Championship.json` | UEFA Euro | 51 | 2016-06-10 → 2016-07-10 |
| `matches_World_Cup.json` | FIFA World Cup | 64 | 2018-06-14 → 2018-07-15 |

**Event schema:** `eventId`/`eventName` (7 types: pass, foul, shot, duel, free kick, offside, touch), `subEventId`/`subEventName`, `tags`, **`eventSec`** (seconds since start of the current half), `id`, `matchId`, `matchPeriod` (`1H`/`2H`/`E1`/`E2`/`P`), `playerId`, `teamId`, `positions` (x,y in [0,100] from the attacking team's perspective).

**Lineups:** in `matches.teamsData` — `lineup`, `bench`, `substitutions` (with minute), `coachId`, `hasFormation`, `score`/`scoreHT`/`scoreET`/`scoreP`, `side`, `winner`, `venue`, `duration`.

**Gaps:** no shot xG (you must model it), no freeze frames, no live, and it stops at 2018.

**Attribution required:** Pappalardo, L.; Massucco, E. (2019), *figshare*, plus the paper Pappalardo et al., *Scientific Data* — https://www.nature.com/articles/s41597-019-0247-7

---

## 4. football-data.org — good free API, but live and lineups are paid

**Pricing verified** (https://www.football-data.org/pricing, © 2014–2026):

| Plan | Price | Contents |
|---|---|---|
| **Free** | €0 | 12 competitions, **scores delayed**, fixtures, schedules delayed, league tables, **10 calls/minute** |
| Free w/ Livescores | €12/mo | + live scores, 20 calls/min |
| Free + Deep Data | €29/mo | + live, **line-ups & subs**, goal scorers, bookings/cards, squads, 30 calls/min |
| Odds Add-On | €15/mo | pre-match H/D/A odds, 40 competitions |
| Statistics Add-On | €15/mo | corners, free kicks, goal kicks, offsides, fouls, possession, saves, throw-ins, shots on/off, cards |

**So the free tier gives you no live, no lineups, no odds, no xG.**

**Terms** (https://www.football-data.org/about, General T&C last updated 1 June 2018; provider Freitag Web Tec UG; Dutch law):
- Registration and API key required.
- **Attribution required:** "Football data provided by the Football-Data.org API".
- Use limited to a single application/domain, subject to a fair-use policy.
- **§9.1:** after cancelling, "the Customer is not permitted to reference the football data (incl. match fixtures, results, league tables, player/squad data, top scorers) obtained through the Football-Data API on their own site or service."

**Endpoints (v4):** `/v4/competitions/{code}/matches`, `/v4/competitions/{code}/standings`, `/v4/competitions/{code}/scorers`, `/v4/fixtures`, `/v4/teams`, `/v4/matches/{id}/head2head`.

**Implemented** as `data/providers/football_data_org.py`. It reads the free-tier
`/v4/competitions/{code}/matches` document only and declares the free tier's
absence of odds by letting the base class refuse `fetch_odds`. Delayed scores are
topped up by `scripts/poll_live.py`; because rows are keyed by `match_id`, a
poll is idempotent and never duplicates a fixture.

---

## 5. API-Sports / API-Football — only free live option, quota is tiny

- **Free tier: 100 requests/day**, resets 00:00 UTC, unused requests are lost (official api-sports.io page).
- Endpoints advertised include fixtures, results, standings, teams, livescore, events, lineups, top scorers, players, coaches, injuries, sidelined, odds, predictions, statistics.
- **No xG endpoint or field is documented.**
- ⚠️ `www.api-football.com` and `/documentation-v3` returned **HTTP 403** from this environment, so the **football-specific plan matrix and the terms/licence are unverified**. Treat the licence as unknown until you read the dashboard's terms after registering.

**Practical:** 100 requests/day cannot backfill a league (380 matches × 3 endpoints). It is workable as a small daily poller for fixtures + live scores only.

**Implemented** as `data/providers/api_football.py`. The numeric competition id is
discovered from the `/leagues` manifest by matching the published name **and**
country, so no opaque id is hard-coded and an ambiguous match raises rather than
guessing. Because the licence is still unverified, the licence policy keeps
`may_serve_from_public_app` False and the adapter is for internal reconciliation
only.

---

## 6. openfootball — CC0, safe to build on, low resolution

- Licence: **CC0 1.0 Universal** — https://raw.githubusercontent.com/openfootball/england/master/LICENSE.md
- Schema/data/scripts dedicated to the public domain — https://raw.githubusercontent.com/openfootball/world/master/README.md
- Coverage (Football.TXT): England, Scotland, Germany, Italy, Spain, France, Netherlands, Belgium, Portugal, Turkey, Greece; the `world` repo adds Americas (MLS, Liga MX), Asia, Africa, Pacific, Middle East.
- Files are per-league-per-season plain text (e.g. `https://github.com/openfootball/england/tree/master/2023-24`).
- **No events, no xG, no odds, no lineups.** Manually maintained and uneven — some leagues stop years ago.
- **This is the only source in the list you can freely redistribute and build a commercial product on.**

---

## 7. ClubElo — useful rating prior, no published terms

- HTML rankings live and current at https://clubelo.com/Ranking (page generated 2026-10-02).
- The documented CSV endpoint pattern is `http://api.clubelo.com/{Club}` (e.g. `Arsenal`), but from this environment it returned **502**, **403**, or connection failure, so the **response schema could not be verified**.
- **No licence, attribution requirement, or commercial/redistribution terms are published.** `clubelo.com/About` is a JS app with no terms text.
- Treat ClubElo ratings as a feature, not as licensed data you can ship.

---

## 8. Understat — xG is great, but there is no permission to take it

- Homepage confirms EPL, La Liga, Bundesliga, Serie A, Ligue 1 and RFPL coverage, with team and player xG, historically, and CSV/JSON/XLSX export of the aggregate tables.
- `https://understat.com/terms` → **404**. No licence, terms, or attribution policy found anywhere on the domain.
- `https://understat.com/robots.txt` is unambiguous:
  ```
  User-agent: *
  Disallow: /
  ```
- **Ruling:** no open licence + a site-wide crawl prohibition means scraping or mirroring is not authorised. Unofficial GitHub/Kaggle mirrors carry no verifiable licence and must not be treated as redistributable.
- **Legal alternatives:** StatsBomb (research/published analysis with logo) for shot xG; Wyscout CC BY 4.0 for events you may commercialise; SkillCorner MIT for spatial data. For Understat-equivalent top-5-league xG in a product, buy a licence (Sportmonks, Stats Perform, Opta).

---

## 9. Metrica Sports sample data — schema testing only

- Repo: https://github.com/metrica-sports/sample-data — exactly **three** directories: `Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3` (tracking + event JSON, plus `documentation/`).
- **There is no LICENSE file** in the repository root → default copyright, all rights reserved.
- README asks for responsible use and acknowledgement if published.
- No xG. Useful only for building and unit-testing a tracking/event ingestion pipeline.

---

## 10. SkillCorner open data — MIT, but tiny

- Repo: https://github.com/SkillCorner/opendata · **MIT licence** (verified: `LICENSE`, 1,097 bytes, on `master`).
- 10 matches of **broadcast tracking** data plus event data (repo description: A-League 2024/25), with `src/`, `viz_tools/`, `notebooks/`, and bodypose assets mirrored on Hugging Face.
- Contains frame-level x/y locations, dynamic events, and **EPV** (expected possession value).
- **EPV is not shot xG** — there is no per-shot xG field.
- MIT means you can use, modify and redistribute commercially (keep the notice). Limitation is volume: 10 matches.

---

## 11. Other checked

- **TheSportsDB** free API (`https://www.thesportsdb.com/free-api`) returned **502** from this environment — unusable as a primary source here.
- **football-data.co.uk paid alternative:** the site now promotes its own *FootballStatsAPI* — an option if you need a licence from that operator.

---

## 12. Rankings by your use case

1. **Historical 1X2 + bookmaker odds → football-data.co.uk.** Nothing else free comes close in depth or coverage. Cost: no published licence → keep it internal, don't redistribute.
2. **Event replay with per-shot xG → StatsBomb Open Data.** The only open source with `minute`/`second`/`timestamp`, `statsbomb_xg`, freeze frames, and Starting XI tactics. Research-only; logo credit required.
3. **Event data you may commercialise → Wyscout soccer-logs (CC BY 4.0).** 1,941 matches across PL, La Liga, Serie A, Bundesliga, Ligue 1, Euro and World Cup, with full lineups/benches/subs/coaches. No xG — model it yourself.
4. **Lineups and player impact → StatsBomb `lineups/` + `Starting XI` tactics** (offline, research) or **football-data.org "Free + Deep Data" €29/mo** / **API-Football events+lineups** (live-ish).
5. **Future live data → there is no free option that clears the bar.** API-Football's 100 req/day free tier is prototype-only; football-data.org charges €12/mo just for live. For anything commercial or reliability-critical, buy a licensed feed (Sportmonks, Stats Perform/Opta, or ask football-data.co.uk's operator about FootballStatsAPI).
6. **Team-strength prior → openfootball CC0** (safe, redistributable) + ClubElo (better ratings, **no published terms** — internal use only).
7. **Spatial/tracking R&D → SkillCorner (MIT)** for a licence-clean 10 matches; Metrica for 3 more sample games but **no licence**.

### Suggested stack
- **Baseline features:** openfootball CC0 fixtures/results → team strength priors you can ship.
- **Odds features:** football-data.co.uk CSVs, cached locally, never redistributed.
- **Model training / explainability:** StatsBomb open data for shot xG and event features; Wyscout for extra event volume under CC BY 4.0.
- **Production live:** licensed provider. Treat both free APIs as development aids.

---

## 13. Explicitly unverified / open questions

| Item | Status |
|---|---|
| API-Football plan matrix, terms and licence | Blocked by HTTP 403 — check the dashboard after registering |
| ClubElo API CSV schema, update cadence, terms | API unreachable (502/403); no terms published |
| `HxG`/`AxG` provenance on football-data.co.uk | Columns exist from 2026/27 but are undocumented in `notes.txt` |
| StatsBomb orphan event files (4,235 vs 3,961 mapped) | Likely duplicate/unlisted matches; filter via the `matches/` manifests |
| Open Understat mirrors with a verifiable licence | None found; do not use |