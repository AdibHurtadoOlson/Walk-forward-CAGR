from __future__ import annotations
from pathlib import Path
import pandas as pd


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install dependencies with `pip install -r requirements.txt` to fetch Yahoo Finance data.") from exc
    raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True,
                       actions=False, progress=False, threads=True)
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no rows; verify the symbols and dates.")
    if isinstance(raw.columns, pd.MultiIndex):
        # Accommodate both (field, ticker) and (ticker, field) column layouts.
        close_level = None
        for level in range(raw.columns.nlevels):
            level_values = list(raw.columns.get_level_values(level).unique())
            if "Close" in level_values:
                close_level = level
                break
        if close_level is None:
            raise ValueError(f"No Close field found in yfinance response: {raw.columns}")
        prices = raw.xs("Close", axis=1, level=close_level).copy()
    else:
        if "Close" not in raw:
            raise ValueError("No Close field found in yfinance response.")
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices.columns = [str(c).upper() for c in prices.columns]
    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        raise ValueError(f"Symbols not returned by Yahoo Finance: {missing}")
    prices = prices[tickers].apply(pd.to_numeric, errors="coerce")
    prices.index = pd.to_datetime(prices.index)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = "Date"
    return validate_prices(prices)


def validate_prices(prices: pd.DataFrame, min_rows: int = 260) -> pd.DataFrame:
    p = prices.copy()
    p.index = pd.to_datetime(p.index)
    if p.empty or p.index.has_duplicates:
        raise ValueError("Price panel is empty or has duplicate dates.")
    if (p.dropna(how="all") <= 0).any().any():
        raise ValueError("Prices must be positive.")
    missing_pct = p.isna().mean()
    too_sparse = missing_pct[missing_pct > 0.10]
    if len(too_sparse):
        raise ValueError(f"More than 10% missing observations for: {too_sparse.to_dict()}")
    # For a cross-sectional strategy, keeping only common dates prevents the
    # backtester from silently assigning different universes on missing dates.
    p = p.sort_index().dropna(how="any")
    if len(p) < min_rows:
        raise ValueError(f"Only {len(p)} complete daily observations; need at least {min_rows}.")
    return p.astype(float)


def get_prices(tickers: list[str], start: str, cache_path: Path, use_cache: bool = False,
               end: str | None = None) -> tuple[pd.DataFrame, str]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if use_cache:
        if not cache_path.exists():
            raise FileNotFoundError(f"No cached price file at {cache_path}; first run the project online.")
        p = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        p = validate_prices(p, min_rows=2)
        missing = sorted(set(tickers) - set(p.columns))
        if missing:
            raise ValueError(f"Cache is missing symbols: {missing}")
        return p[tickers].loc[lambda x: x.index >= pd.Timestamp(start)], "cache"
    p = download_prices(tickers, start, end)
    p.to_csv(cache_path, index_label="Date")
    return p, "Yahoo Finance via yfinance"
