"""Generate reusable tables, summary JSON, and a visual research pack."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .backtest import walk_forward_research


def run_research(daily_prices: pd.DataFrame, benchmark: str, train_months: int,
                 cost_bps: float, outdir: Path, source: str) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    monthly = daily_prices.resample("ME").last().dropna(how="any")
    oos, selected, metrics = walk_forward_research(monthly, benchmark=benchmark,
                                                    train_months=train_months, cost_bps=cost_bps)
    oos.to_csv(outdir / "walk_forward_returns.csv", index_label="signal_date")
    selected.to_csv(outdir / "selected_parameters.csv", index_label="signal_date")
    weight_cols = [c for c in oos.columns if c.startswith("weight_")]
    # The chosen per-month strategy weights are reconstructed from the chosen
    # parameter path and recorded for an audit trail.
    from .backtest import strategy_path
    rows = []
    for dt, row in selected.iterrows():
        wrow = strategy_path(monthly.loc[:dt], int(row["lookback_months"]), 1,
                             int(row["top_n"]), cost_bps).iloc[-1]
        rows.append({"signal_date": dt, **{k: float(v) for k, v in wrow.items() if k.startswith("weight_")}})
    pd.DataFrame(rows).set_index("signal_date").to_csv(outdir / "monthly_weights.csv")
    summary = {"source": source, "daily_start": str(daily_prices.index.min().date()),
               "daily_end": str(daily_prices.index.max().date()), "benchmark": benchmark,
               "lookahead_control": "Walk-forward parameter selection uses only returns from signal dates before each current signal date.",
               **metrics}
    (outdir / "performance_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    wealth_strategy = (1 + oos["net_return"]).cumprod()
    wealth_bench = (1 + oos["benchmark_return"]).cumprod()
    wealth_strategy.name = "Walk-forward strategy"
    wealth_bench.name = f"{benchmark} benchmark"
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    pd.concat([wealth_strategy, wealth_bench], axis=1).plot(ax=axes[0], title="Out-of-sample growth of $1")
    axes[0].set_ylabel("Growth")
    ((1 + oos["net_return"]).cumprod() / (1 + oos["net_return"]).cumprod().cummax() - 1).plot(ax=axes[1], title="Strategy drawdown")
    axes[1].set_ylabel("Drawdown")
    oos["turnover"].plot(ax=axes[2], title="Monthly one-way turnover")
    axes[2].set_ylabel("Turnover")
    fig.savefig(outdir / "performance_dashboard.png", dpi=160); plt.close(fig)
    checks = {"strategy_and_benchmark_same_months": bool(oos["benchmark_return"].notna().all()),
              "strictly_positive_month_count": int((oos["net_return"] > 0).sum()),
              "out_of_sample_months": int(len(oos)),
              "costs_nonnegative": bool((oos["transaction_cost"] >= 0).all()),
              "average_gross_minus_net_matches_cost": bool(((oos["gross_return"] - oos["net_return"] - oos["transaction_cost"]).abs().max() < 1e-9))}
    (outdir / "research_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return summary
