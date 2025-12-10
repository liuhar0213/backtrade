import pandas as pd
from backtest_simple import run_backtest


def make_synthetic_df(n=30):
    times = pd.date_range('2020-01-01', periods=n, freq='h')
    # create a gentle price walk
    base = 100.0
    close = [base + i * 0.5 for i in range(n)]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    df = pd.DataFrame({
        'datetime': times.astype(str),
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': [1000 + i for i in range(n)],
    })
    return df


def test_run_backtest_returns_expected_structure():
    df = make_synthetic_df(40)
    res = run_backtest(df=df, save_csv=False, show_trades=False)

    assert isinstance(res, dict)
    assert 'capital' in res
    assert 'trades_df' in res
    assert 'equity_curve' in res
    assert 'stats' in res

    # types
    assert isinstance(res['capital'], (int, float))
    import pandas as pd
    assert isinstance(res['trades_df'], pd.DataFrame)
    assert isinstance(res['equity_curve'], list)
    assert isinstance(res['stats'], dict)
