"""Phase 1 environment smoke check.

Verifies that the pinned dependency stack not only imports but actually
computes correctly on this interpreter. Imports alone will not catch binary
incompatibility between wheels (e.g. a NumPy 2.x ABI mismatch against a
SciPy or LightGBM build).

Run with:
    python tools/smoke_check.py
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    """Run every smoke check and report pass/fail per library.

    Returns:
        Process exit code: 0 if all checks pass, 1 otherwise.
    """
    print(f"python      : {platform.python_version()} ({sys.executable})")
    print("-" * 68)

    failures: list[str] = []

    def _numpy() -> str:
        import numpy as np

        a = np.arange(10, dtype=np.float64)
        return f"numpy {np.__version__}: mean={a.mean():.3f}"

    def _pandas() -> str:
        import pandas as pd

        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        grouped = df.groupby("y", observed=True)["x"].sum().sum()
        assert int(grouped) == 6, "pandas groupby produced wrong result"
        return f"pandas {pd.__version__}: groupby_sum={int(grouped)}"

    def _pandas_numpy_interop() -> str:
        """Guard against the NumPy 2.x / pandas ABI split."""
        import numpy as np
        import pandas as pd

        s = pd.Series(np.linspace(0.0, 1.0, 5))
        assert abs(float(s.mean()) - 0.5) < 1e-12, "numpy->pandas conversion broken"
        return "pandas<->numpy interop ok"

    def _scipy() -> str:
        from scipy import optimize, stats

        root = optimize.brentq(lambda x: x**2 - 2.0, 0.0, 2.0)
        assert abs(root - 2.0**0.5) < 1e-9, "brentq returned wrong root"
        ppf = stats.norm.ppf(0.975)
        assert abs(ppf - 1.959963985) < 1e-6, "norm.ppf returned wrong value"
        return f"scipy: brentq(sqrt2)={root:.9f}, norm.ppf(0.975)={ppf:.6f}"

    def _sklearn() -> str:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss

        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 3))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        model = LogisticRegression().fit(x, y)
        proba = model.predict_proba(x)
        assert np.allclose(proba.sum(axis=1), 1.0), "predict_proba rows do not sum to 1"
        loss = log_loss(y, proba, labels=[0, 1])
        assert loss < 0.7, f"log_loss implausibly high: {loss}"
        return f"sklearn: fit ok, log_loss={loss:.4f}"

    def _statsmodels() -> str:
        import numpy as np
        import statsmodels.api as sm

        rng = np.random.default_rng(0)
        x = sm.add_constant(rng.normal(size=(120, 2)))
        beta = np.array([0.5, 1.2, -0.8])
        y = x @ beta + rng.normal(scale=0.1, size=120)
        res = sm.OLS(y, x).fit()
        assert np.allclose(res.params, beta, atol=0.15), "OLS params off"
        poisson = sm.GLM(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), sm.add_constant(np.arange(6.0)), family=sm.families.Poisson()
        ).fit()
        assert np.isfinite(poisson.params).all(), "Poisson GLM produced non-finite params"
        return f"statsmodels: OLS+Poisson GLM ok, r2={res.rsquared:.4f}"

    def _lightgbm() -> str:
        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 4))
        y = (x[:, 0] > 0).astype(int)
        train = lgb.Dataset(x, label=y)
        booster = lgb.train(
            {"objective": "binary", "verbose": -1, "num_leaves": 8, "min_data_in_leaf": 5},
            train,
            num_boost_round=15,
        )
        raw = booster.predict(x, raw_score=True)
        assert np.isfinite(raw).all(), "LightGBM produced non-finite predictions"
        return f"lightgbm {lgb.__version__}: trained booster ok, top_feat_0={booster.feature_importance('split')[0]}"

    def _plotly() -> str:
        import plotly.graph_objects as go

        fig = go.Figure(go.Scatter(x=[0, 1, 2], y=[0.1, 0.6, 0.3]))
        payload = fig.to_json()
        assert len(payload) > 10, "plotly figure serialised to nothing"
        return f"plotly: figure serialises ok ({len(payload)} chars)"

    def _streamlit() -> str:
        import streamlit as st

        assert hasattr(st, "cache_data"), "st.cache_data missing"
        assert hasattr(st, "cache_resource"), "st.cache_resource missing"
        assert hasattr(st, "components"), "st.components missing"
        return f"streamlit {st.__version__}: cache_data/cache_resource present"

    def _sqlite() -> str:
        import sqlite3
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            conn = sqlite3.connect(str(Path(tmp) / "t.db"))
            conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
            conn.execute("INSERT INTO t VALUES (?, ?)", (1, "x"))
            conn.commit()
            got = conn.execute("SELECT a, b FROM t").fetchone()
            conn.close()
        assert got == (1, "x"), "sqlite roundtrip failed"
        return f"sqlite {sqlite3.sqlite_version}: roundtrip ok"

    def _pytest_ruff() -> str:
        import pytest
        import ruff  # noqa: F401  (import presence check only)

        return f"pytest {pytest.__version__} + ruff present"

    def _python310_sanity() -> str:
        """Confirm we are genuinely on 3.10 and not accidentally on a newer runtime."""
        assert sys.version_info[:2] == (3, 10), f"expected Python 3.10, got {sys.version_info[:3]}"
        return f"version gate ok: {sys.version_info.major}.{sys.version_info.minor}"

    for fn in (
        _python310_sanity,
        _numpy,
        _pandas,
        _pandas_numpy_interop,
        _scipy,
        _sklearn,
        _statsmodels,
        _lightgbm,
        _plotly,
        _streamlit,
        _sqlite,
        _pytest_ruff,
    ):
        try:
            msg = fn()  # type: ignore[operator]
            print(f"[PASS] {msg}")
        except Exception as exc:
            failures.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
            print(f"[FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")

    print("-" * 68)
    total = 12
    if failures:
        print(f"{len(failures)} of {total} checks FAILED")
        return 1
    print(f"All {total} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
