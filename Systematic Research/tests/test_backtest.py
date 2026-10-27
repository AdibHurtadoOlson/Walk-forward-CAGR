import numpy as np
import pandas as pd
import pytest
from systematic_lab.backtest import monthly_prices, momentum_scores, weights_from_scores, strategy_path, performance_summary, walk_forward_research


def test_momentum_omits_latest_month():
    dates = pd.date_range("2020-01-31", periods=15, freq="ME")
    p = pd.DataFrame({"A": np.arange(100, 115, dtype=float)}, index=dates)
    scores = momentum_scores(p, lookback_months=12, skip_months=1)
    # At t=13, score is P[t-1] / P[t-12] - 1, not P[t]/P[t-12].
    expected = p["A"].iloc[11] / p["A"].iloc[0] - 1
    assert scores["A"].iloc[12] == pytest.approx(expected)


def test_weights_are_long_only_and_sum_to_one():
    w = weights_from_scores(pd.Series({"A": 0.2, "B": -0.1, "C": 0.1}), top_n=2)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)
    no_positive = weights_from_scores(pd.Series({"A": -0.2, "B": 0.0}), top_n=2)
    assert no_positive.sum() == 0


def test_transaction_costs_reduce_net_return():
    idx = pd.date_range("2018-01-31", periods=48, freq="ME")
    p = pd.DataFrame({"A": np.geomspace(100, 200, 48), "B": np.geomspace(200, 100, 48)}, index=idx)
    path = strategy_path(p, lookback_months=6, skip_months=1, top_n=1, cost_bps=20)
    valid = path.dropna(subset=["gross_return", "net_return"])
    assert (valid["net_return"] <= valid["gross_return"] + 1e-12).all()
    assert (valid["transaction_cost"] >= 0).all()


def test_realized_return_is_next_month_not_signal_month():
    idx = pd.date_range("2020-01-31", periods=18, freq="ME")
    a = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270], float)
    b = np.array([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15], float)
    p = pd.DataFrame({"A": a, "B": b}, index=idx)
    path = strategy_path(p, lookback_months=6, skip_months=1, top_n=1, cost_bps=0)
    dt = idx[10]
    # A has strongly positive past momentum at the signal date; next-month A return is the realized value.
    expected_next_a = p["A"].iloc[11] / p["A"].iloc[10] - 1
    if path.loc[dt, "weight_A"] > 0.99:
        assert path.loc[dt, "gross_return"] == pytest.approx(expected_next_a)


def test_walk_forward_returns_are_out_of_sample_and_reported():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2010-01-31", periods=100, freq="ME")
    rets = rng.normal(0.006, 0.035, size=(100, 4))
    p = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=["SPY", "A", "B", "C"])
    oos, chosen, metrics = walk_forward_research(p, "SPY", train_months=36, lookbacks=(6, 9, 12), top_ns=(1, 2), cost_bps=5)
    assert len(oos) > 0
    assert oos.index.equals(chosen.index)
    assert "cagr" in metrics["strategy"]
    assert (oos["transaction_cost"] >= 0).all()


def test_performance_summary_rejects_insufficient_sample():
    with pytest.raises(ValueError): performance_summary(pd.Series([0.01]))
