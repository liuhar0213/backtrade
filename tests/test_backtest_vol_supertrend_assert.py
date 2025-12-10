import pandas as pd
import os
from backtest_vol_supertrend_api import run_backtests


def make_synthetic_df(n=120):
    times = pd.date_range('2020-01-01', periods=n, freq='8H')
    base = 10.0
    close = [base + 0.05 * i for i in range(n)]
    df = pd.DataFrame({
        'datetime': times.astype(str),
        'open': close,
        'high': [c + 0.2 for c in close],
        'low': [c - 0.2 for c in close],
        'close': close,
        'volume': [100 + i for i in range(n)],
    })
    return df


def test_vol_supertrend_wrapper_runs(tmp_path):
    df = make_synthetic_df(80)
    trades, extra = run_backtests(df=df, cleanup=True)
    assert isinstance(trades, list)
    assert isinstance(extra, dict)
