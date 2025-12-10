import importlib
import runpy


def test_backtest_wrappers_call_runpy(monkeypatch):
    modules = [
        'scripts.run_backtest_with_fees',
        'scripts.run_backtest_with_sessions',
        'scripts.run_backtest_tradingview_patterns',
        'scripts.run_backtest_vol_supertrend_sessions',
    ]
    called = {'count': 0}

    def fake_run_path(path, run_name=None):
        called['count'] += 1
        return {}

    monkeypatch.setattr('runpy.run_path', fake_run_path)

    for m in modules:
        mod = importlib.import_module(m)
        assert hasattr(mod, 'main')
        mod.main()

    assert called['count'] == len(modules)
