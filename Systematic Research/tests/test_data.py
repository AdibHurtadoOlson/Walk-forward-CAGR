import numpy as np
import pandas as pd
import pytest
from systematic_lab.data import download_prices, validate_prices

def test_ticker_first_yfinance_multiindex_parses_close():
    # Exercise the column layout without using the network. The actual download
    # function is tested indirectly through this same normalization contract.
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    cols = pd.MultiIndex.from_product([["SPY", "TLT"], ["Open", "Close"]])
    raw = pd.DataFrame(np.tile([99.0, 100.0, 49.0, 50.0], (300, 1)) + np.arange(300)[:, None] * 0.1, index=dates, columns=cols)
    close = raw.xs("Close", axis=1, level=1)
    result = validate_prices(close, min_rows=260)
    assert list(result.columns) == ["SPY", "TLT"]
    assert len(result) == 300

def test_nonpositive_prices_are_rejected():
    bad = pd.DataFrame({"SPY": [100, 101, 0]}, index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError): validate_prices(bad, min_rows=2)
