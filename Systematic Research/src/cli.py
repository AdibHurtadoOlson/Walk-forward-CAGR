from __future__ import annotations
import argparse
from pathlib import Path
from .data import get_prices
from .report import run_research

DEFAULT = "SPY,QQQ,VOO,NVDA,AMD,TSM,MSFT,AMZN,GOOG,COST,TGT,WMT"

def main() -> None:
    p = argparse.ArgumentParser(description="Run ETF momentum research with walk-forward parameter selection.")
    p.add_argument("--tickers", default=DEFAULT)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--train-months", type=int, default=60)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--cache", default="data/cache/prices.csv")
    p.add_argument("--outdir", default="outputs")
    a = p.parse_args()
    tickers = [s.strip().upper() for s in a.tickers.split(",") if s.strip()]
    if a.benchmark.upper() not in tickers:
        p.error("The benchmark must be included in --tickers so its prices can be fetched.")
    if len(set(tickers)) != len(tickers):
        p.error("Duplicate tickers are not allowed.")
    prices, source = get_prices(tickers, a.start, Path(a.cache), a.use_cache, a.end)
    summary = run_research(prices, a.benchmark.upper(), a.train_months, a.cost_bps, Path(a.outdir), source)
    print(f"Research pack written to {Path(a.outdir).resolve()}")
    print(f"Source: {source}; out-of-sample months: {summary['strategy']['months']}")
    print(f"Walk-forward CAGR: {summary['strategy']['cagr']:.2%}; benchmark CAGR: {summary['benchmark']['cagr']:.2%}")

if __name__ == "__main__": main()
