import numpy as np
import pandas as pd
from systematic_lab.report import run_research

def test_walk_forward_report_outputs_tables_and_dashboard(tmp_path):
    rng = np.random.default_rng(22)
    months = pd.date_range("2013-01-31", periods=105, freq="ME")
    ret = rng.normal(0.005, 0.03, size=(105, 4))
    prices = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=months, columns=["SPY", "A", "B", "C"])
    summary = run_research(prices, "SPY", 36, 5.0, tmp_path, "test fixture")
    assert summary["strategy"]["months"] > 0
    assert (tmp_path / "walk_forward_returns.csv").exists()
    assert (tmp_path / "selected_parameters.csv").exists()
    assert (tmp_path / "performance_dashboard.png").exists()
    checks = pd.read_json(tmp_path / "research_checks.json", typ="series")
    assert bool(checks["costs_nonnegative"])
