import pandas as pd
import numpy as np
import importlib


def make_synthetic_df(n=60):
    # simple synthetic increasing price series
    times = pd.date_range('2020-01-01', periods=n, freq='h')
    close = np.linspace(100, 200, n)
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(100, 1000, size=n)
    df = pd.DataFrame({'time': times, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    return df


def test_run_rma_fix_with_synthetic(monkeypatch):
    mod = importlib.import_module('scripts.run_rma_fix')

    # monkeypatch pandas.read_csv used inside the runner
    synthetic = make_synthetic_df(120)

    monkeypatch.setattr('pandas.read_csv', lambda *args, **kwargs: synthetic)

    res = mod.main()
    assert res is not None, "run_rma_fix.main() returned None"
    df_old, df_new = res
    assert hasattr(df_old, 'shape') and hasattr(df_new, 'shape')
    assert df_old.shape[0] == df_new.shape[0]
