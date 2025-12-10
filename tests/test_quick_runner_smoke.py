import importlib
import runpy
import sys


def test_quick_runner_calls_runpy(monkeypatch):
    # Ensure argparse in the runner doesn't accidentally parse pytest flags
    monkeypatch.setattr(sys, 'argv', ['run_quick_backtest'])

    mod = importlib.import_module('scripts.run_quick_backtest')

    called = {'ok': False}

    def fake_run_path(path, run_name=None):
        called['ok'] = True
        return {}

    monkeypatch.setattr('runpy.run_path', fake_run_path)

    # call main() and ensure it doesn't raise
    mod.main()
    assert called['ok'], 'runpy.run_path was not called by the quick runner'
