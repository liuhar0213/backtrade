import importlib


def test_runner_modules_importable():
    modules = [
        'scripts.run_quick_backtest',
        'scripts.run_tradingview_patterns',
        'scripts.run_supertrend',
        'scripts.run_rma_fix',
    ]

    for m in modules:
        mod = importlib.import_module(m)
        assert hasattr(mod, 'main'), f"{m} missing main()"
