"""Monthly cross-sectional momentum signals and walk-forward research."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd


def monthly_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Sample the last available adjusted close in each calendar month."""
    if daily_prices.empty or daily_prices.index.has_duplicates:
        raise ValueError("Daily prices must be non-empty and have unique dates.")
    return daily_prices.sort_index().resample("ME").last().dropna(how="any")


def momentum_scores(monthly: pd.DataFrame, lookback_months: int = 12, skip_months: int = 1) -> pd.DataFrame:
    """Compute 12-1-style momentum with only data known at each signal date.

    With lookback=12 and skip=1, the score uses month-end price t-1 divided
    by month-end price t-12, so the most recent month is omitted.
    """
    if lookback_months <= skip_months or skip_months < 0:
        raise ValueError("lookback_months must be greater than nonnegative skip_months.")
    if (monthly <= 0).any().any():
        raise ValueError("Prices must be positive.")
    return monthly.shift(skip_months).div(monthly.shift(lookback_months), axis=0) - 1.0


def weights_from_scores(scores: pd.Series, top_n: int = 3, require_positive: bool = True) -> pd.Series:
    """Allocate equal long-only weights to the strongest eligible assets."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    s = pd.Series(scores, dtype=float).dropna()
    w = pd.Series(0.0, index=s.index)
    eligible = s[s > 0] if require_positive else s
    if eligible.empty:
        return w
    selected = eligible.nlargest(min(top_n, len(eligible))).index
    w.loc[selected] = 1.0 / len(selected)
    return w


def strategy_path(monthly: pd.DataFrame, lookback_months: int = 12, skip_months: int = 1,
                  top_n: int = 3, cost_bps: float = 10.0) -> pd.DataFrame:
    """Build signal-date rows with the next month's returns and trading costs.

    Each row is indexed by the date when the signal is formed. The weights
    formed at that close earn the *following* calendar month's return. Turnover
    cost is charged on the rebalance at the signal date, avoiding same-month
    look-ahead. Cost bps apply to one-way turnover (half of gross weight change).
    """
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative.")
    scores = momentum_scores(monthly, lookback_months, skip_months)
    fwd_returns = monthly.pct_change(fill_method=None).shift(-1)
    all_weights = []
    prev = pd.Series(0.0, index=monthly.columns)
    prev_cash = 1.0  # Start the simulated portfolio entirely in cash.
    signal_ready = scores.notna().any(axis=1)
    for dt in monthly.index:
        row = scores.loc[dt]
        w = weights_from_scores(row, top_n=top_n).reindex(monthly.columns, fill_value=0.0)
        cash_weight = max(0.0, 1.0 - float(w.sum()))
        # Include the implicit cash sleeve in the two-sided turnover formula.
        # This makes the initial move from 100% cash to fully invested equal 100% turnover.
        turnover = float(((w - prev).abs().sum() + abs(cash_weight - prev_cash)) / 2.0)
        all_weights.append((dt, w, turnover))
        prev, prev_cash = w, cash_weight
    weights_df = pd.DataFrame({dt: w for dt, w, _ in all_weights}).T
    weights_df.index = pd.DatetimeIndex(weights_df.index)
    turnover = pd.Series({dt: trn for dt, _, trn in all_weights}, name="turnover").reindex(monthly.index)
    # A signal row t maps to subsequent-month asset returns stored at t+1.
    realized = fwd_returns.reindex(monthly.index)
    gross = (weights_df * realized).sum(axis=1, min_count=1)
    # Do not count momentum warm-up rows as zero-return cash performance: they
    # are not eligible strategy observations because the signal is unavailable.
    gross = gross.where(signal_ready)
    cost = turnover * cost_bps / 10000.0
    net = gross - cost
    result = pd.DataFrame({"gross_return": gross, "turnover": turnover,
                           "transaction_cost": cost, "net_return": net})
    result = result.join(weights_df.add_prefix("weight_"))
    result = result.join(realized.add_prefix("forward_return_"))
    return result


def performance_summary(returns: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    """Compute annualized, drawdown, and distribution metrics for monthly returns."""
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2:
        raise ValueError("At least two returns are required.")
    gross_growth = float((1 + r).prod())
    cagr = gross_growth ** (periods_per_year / len(r)) - 1 if gross_growth > 0 else -1.0
    vol_period = float(r.std(ddof=1))
    annual_vol = vol_period * np.sqrt(periods_per_year)
    sharpe = float(r.mean() / vol_period * np.sqrt(periods_per_year)) if vol_period else float("nan")
    downside = r[r < 0]
    downside_vol = float(np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(periods_per_year))
    sortino = float((r.mean() * periods_per_year) / downside_vol) if downside_vol else float("nan")
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    max_dd = float(dd.min())
    return {"months": int(len(r)), "cagr": float(cagr), "annualized_volatility": float(annual_vol),
            "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd,
            "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else float("nan"),
            "hit_rate": float((r > 0).mean()), "best_month": float(r.max()), "worst_month": float(r.min())}


def _selection_score(r: pd.Series) -> float:
    """Annualized Sharpe-like selection score; NaN for short/constant samples."""
    x = pd.Series(r).dropna()
    if len(x) < 24 or x.std(ddof=1) <= 1e-12:
        return -np.inf
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12))


def walk_forward_research(monthly: pd.DataFrame, benchmark: str = "SPY", train_months: int = 60,
                           lookbacks: tuple[int, ...] = (6, 9, 12), top_ns: tuple[int, ...] = (2, 3, 5),
                           cost_bps: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Choose strategy parameters on historical returns, then apply out of sample.

    At every signal month, the candidate's historical realized returns are
    restricted to dates strictly earlier than that signal date. The next-month
    return for the current signal is not available to parameter selection.
    This is computationally simple and intentionally transparent rather than
    a black-box optimizer.
    """
    if benchmark not in monthly.columns:
        raise ValueError(f"Benchmark {benchmark!r} is not in the price universe.")
    if train_months < 24:
        raise ValueError("train_months must be at least 24 to reduce unstable selection.")
    candidates = {}
    for lb, top in itertools.product(lookbacks, top_ns):
        candidates[(lb, top)] = strategy_path(monthly, lb, 1, top, cost_bps)
    min_lb = max(lookbacks)
    first = max(train_months + min_lb, min_lb + 2)
    if len(monthly) <= first + 1:
        raise ValueError(f"Need more than {first + 1} month-end observations; received {len(monthly)}.")
    records, selection = [], []
    for idx in range(first, len(monthly) - 1):
        signal_date = monthly.index[idx]
        scores = []
        for (lb, top), path in candidates.items():
            # Historical training outcomes only: signal dates strictly before t.
            hist = path.loc[path.index < signal_date, "net_return"].dropna()
            scores.append((_selection_score(hist), lb, top, len(hist)))
        scores.sort(key=lambda z: (z[0], -z[1], -z[2]), reverse=True)
        best_score, best_lb, best_top, hist_n = scores[0]
        chosen = candidates[(best_lb, best_top)].loc[signal_date]
        gross = chosen["gross_return"]
        net = chosen["net_return"]
        benchmark_return = monthly[benchmark].pct_change(fill_method=None).shift(-1).loc[signal_date]
        records.append({"signal_date": signal_date, "gross_return": gross, "net_return": net,
                        "transaction_cost": chosen["transaction_cost"], "turnover": chosen["turnover"],
                        "benchmark_return": benchmark_return, "selected_lookback": best_lb,
                        "selected_top_n": best_top})
        selection.append({"signal_date": signal_date, "lookback_months": best_lb, "top_n": best_top,
                          "training_observations": hist_n, "training_score": best_score})
    oos = pd.DataFrame(records).set_index("signal_date")
    oos = oos.dropna(subset=["net_return", "benchmark_return"])
    sel = pd.DataFrame(selection).set_index("signal_date").loc[oos.index]
    metrics = {"strategy": performance_summary(oos["net_return"]),
               "benchmark": performance_summary(oos["benchmark_return"]),
               "average_monthly_turnover": float(oos["turnover"].mean()),
               "average_monthly_transaction_cost": float(oos["transaction_cost"].mean()),
               "mean_excess_monthly_return": float((oos["net_return"] - oos["benchmark_return"]).mean())}
    return oos, sel, metrics
