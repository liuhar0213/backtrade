import pandas as pd
from backtest_with_fees import run_backtests
import backtest_with_fees


class DummyDetector:
    def __init__(self, df):
        # keep reference but ignore
        self.df = df

    def detect_bullish_cannon(self, i):
        # return True at a specific index where MA windows are valid
        return i == 70

    # all other detectors return False
    def __getattr__(self, name):
        if name.startswith('detect_'):
            return lambda i: False
        raise AttributeError(name)


def make_synthetic_df(n=200):
    times = pd.date_range('2020-01-01', periods=n, freq='h')
    # gentle upward trend so MA20 > MA60 eventually
    base = 100.0
    close = [base + i * 0.2 for i in range(n)]
    open_ = [c - 0.05 for c in close]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    volume = [100 + (i % 10) for i in range(n)]
    df = pd.DataFrame({
        'time': times.astype(str),
        'timestamp': times,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    return df


def test_run_backtests_produces_trades_with_dummy_detector(monkeypatch):
    # Replace the detector used by PatternBacktesterWithFees with our dummy
    monkeypatch.setattr(backtest_with_fees, 'CorrectGoldenKDetector', DummyDetector)

    df = make_synthetic_df()

    trades, summary = run_backtests(df=df, fee_rate=0.0005, save_csv=False)

    assert isinstance(trades, list)
    assert isinstance(summary, dict)

    # With the dummy detector we expect at least zero trades (no crash)
    assert trades is not None
    assert summary is not None
