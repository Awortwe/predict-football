# AGENTS.md

Guidance for AI agents working in this repository. Read this before changing code.

## What this project is

A football probability and analysis system. It produces calibrated probabilities
for match outcomes, and the central technical requirement is **honesty**: every
number shown to a user must be traceable to data that was genuinely available at
the moment the prediction was made.

## Non-negotiable rules

1. **No target leakage.** A pre-match model may only use columns marked
   `Availability.PRE_MATCH` in `data/schema.py`. Halftime models may add
   halftime columns. Never add a feature without classifying it, and never
   bypass `assert_no_leakage()`.
2. **No time-based backtest leaks.** Split by date, never randomly. A model
   trained on later data must never be evaluated on earlier data.
3. **State coverage and sample size.** Every reported accuracy, Brier score or
   log loss must be accompanied by the number of matches evaluated. `380 of 380`
   and `380 of 120` are different claims and the difference must be visible.
4. **Calibration is part of the result.** A probability model is judged on
   calibration as well as discrimination. Report reliability data, not just
   accuracy. An uncalibrated model presented without that caveat is a defect.
5. **Never commit or redistribute restricted raw data.** See
   `docs/DATA_SOURCES.md` and `config/licences.py`. Raw provider payloads are
   gitignored. Test fixtures must be synthetic and must not resemble real rows.
6. **Null is not False.** When a value is unknown, store SQL `NULL` / `pd.NA` and
   say "unknown". Do not default to zero, False or "not started". This applies
   especially to flags like `is_captain`.
7. **Do not merge teams you cannot confidently merge.** The team resolver returns
   `None` for unknown names so a gap stays visible. A wrong merge silently
   corrupts match identity and every downstream feature.

## Environment

- Python 3.10 exactly (`requires-python = ">=3.10"`, `runtime.txt`).
- All dependencies are pinned in both `pyproject.toml` and `requirements.txt`.
  Do not add an unpinned dependency.
- Install: `pip install -e ".[dev]"`.
- Verify: `python -m pytest -q` and `python -m ruff check .`. Both must pass.
- Also run `python tools/smoke_check.py` after changing dependencies.

## Conventions

- Google-style docstrings on every public function, class and module, with
  `Args`, `Returns` and `Raises` sections. The docstring should explain *why*,
  not restate the signature.
- Type hints everywhere; the package ships `py.typed`.
- Standard library `sqlite3` behind `MatchRepository`, not an ORM. This is a
  deliberate choice to keep the dependency surface small and swappable.
- Logging via `module_logger = logging.getLogger(__name__)`. No `print` outside
  `scripts/` and `tools/`.
- Line length 120, enforced by ruff.

## Writing files on Windows

This project is developed on Windows with PowerShell 5.1. Two traps:

- `Set-Content -Encoding utf8` writes a **BOM**. This has already corrupted
  `pyproject.toml` and `cache.py`. Write UTF-8 without BOM.
- In PowerShell strings, escape backticks rather than embedding them naively,
  and prefer `[System.IO.File]::WriteAllText` with an explicit
  `UTF8Encoding($false)` when a file must be rewritten wholesale.

## Adding a data source

1. Check `docs/DATA_SOURCES.md` for the licence first, and add an entry to
   `config/licences.py`.
2. Implement `DataProvider` in `data/providers/`.
3. Register it in `data/providers/registry.py`. Factories must accept
   `**kwargs` so callers can inject a cache, resolver or fake session.
4. Map the payload to the **canonical** schema and return canonical dtypes via
   `apply_dtypes(..., keep_extra=True)`.
5. Use one cache resource key per downloaded document. Two accessors reading the
   same file must not each fetch it.
6. Add tests with synthetic fixtures.

## Testing expectations

- Every new function needs a test that would fail if the logic were wrong.
- Test the unhappy paths: malformed payloads, HTML error pages, unknown team
  names, missing columns, empty frames.
- Verify a numeric expectation by computing it independently. Do not copy a
  figure from the implementation into the test; that only asserts consistency.
- Use `FakeSession` from `tests/conftest.py` so tests never touch the network.
  Mark genuinely network-dependent tests with `@pytest.mark.network`.
- Tests must not mutate frozen `Settings`; construct a second one instead.