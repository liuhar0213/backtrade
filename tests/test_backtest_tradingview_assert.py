import pandas as pd
import os
from backtest_tradingview_api import run_backtests


def make_synthetic_df(n=60):
    times = pd.date_range('2020-01-01', periods=n, freq='h')
    base = 200.0
    close = [base + 0.2 * i for i in range(n)]
    df = pd.DataFrame({
        'time': times.astype(str),
        'open': close,
        'high': [c + 0.5 for c in close],
        'low': [c - 0.5 for c in close],
        'close': close,
        'volume': [1000 + i for i in range(n)],
    })
    return df


def test_tradingview_wrapper_runs(tmp_path):
    df = make_synthetic_df(40)
    # write CSV to data/ETHUSDT_15.csv (wrapper will use this file)
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'ETHUSDT_15.csv')
    df.to_csv(csv_path, index=False)

    try:
        results, df_obj = run_backtests(csv_path=csv_path, save_csv=False)
        assert isinstance(results, list)
        # df_obj may be a pandas DataFrame or None
    finally:
        try:
            os.remove(csv_path)
        except Exception:
            pass
