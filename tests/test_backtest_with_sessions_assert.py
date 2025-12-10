import pandas as pd
import os
from backtest_with_sessions_api import run_backtests


def make_synthetic_sessions_df(n=120):
    times = pd.date_range('2020-01-01', periods=n, freq='8h')
    base = 50.0
    close = [base + 0.1 * i for i in range(n)]
    df = pd.DataFrame({
        'datetime': times.astype(str),
        'open': close,
        'high': [c + 0.5 for c in close],
        'low': [c - 0.5 for c in close],
        'close': close,
        'volume': [100 + i for i in range(n)],
    })
    return df


def test_backtest_with_sessions_wrapper_runs(tmp_path):
    df = make_synthetic_sessions_df(80)
    # write a CSV into repo root so legacy script can see it if it looks for specific filename
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'LINKUSDT_8h_2020-01-01_to_2025-11-09.csv')
    csv_path = os.path.abspath(csv_path)
    df.to_csv(csv_path, index=False)

    try:
        trades, summary = run_backtests(csv_path=csv_path)
        assert isinstance(trades, list)
        assert isinstance(summary, dict)
    finally:
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
            except Exception:
                pass
