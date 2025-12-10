import os
import runpy
import sys
import types
from typing import Optional, Tuple, Dict, Any
import pandas as pd

REPO_ROOT = os.path.dirname(__file__)
LEGACY_SCRIPT = os.path.join(REPO_ROOT, 'backtest_tradingview_patterns.py')
DATA_DIR = os.path.join(REPO_ROOT, 'data')
DEFAULT_CSV = os.path.join(DATA_DIR, 'ETHUSDT_15.csv')


def _inject_dummy_detector(df: pd.DataFrame):
    """Place a dummy `tradingview_patterns_optimized` module into sys.modules
    so the legacy script's import resolves to a detector that returns a
    simple patterns DataFrame for testing.
    """
    mod = types.SimpleNamespace()

    class DummyDetector:
        def __init__(self, df_arg, trend_rule='SMA50'):
            # legacy code passes the full dataframe; keep a copy
            self.df = df_arg

        def detect_all_patterns(self):
            # produce a DataFrame of boolean signals, one column with a
            # single True somewhere safe
            s = pd.Series([False] * len(self.df))
            if len(self.df) > 5:
                s.iloc[min(10, len(self.df)-1)] = True
            return pd.DataFrame({'DummyPattern': s})

    mod.TradingViewPatternDetectorOptimized = DummyDetector
    sys.modules['tradingview_patterns_optimized'] = mod


def run_backtests(df: Optional[pd.DataFrame] = None, csv_path: Optional[str] = None, save_csv: bool = False) -> Tuple[list, Dict[str, Any]]:
    """Run the legacy `backtest_tradingview_patterns.py` with a dummy detector.

    If `df` is provided it will be saved to `data/ETHUSDT_15.csv` and used.
    Returns (results_list, results_df_dict) where results_list is the list of
    per-pattern result dicts and results_df_dict is the pandas DataFrame
    converted to a dict (if available).
    """
    # Ensure data dir exists
    os.makedirs(DATA_DIR, exist_ok=True)

    tmp_path = None
    try:
        if df is not None:
            # save to expected location
            tmp_path = DEFAULT_CSV
            df.to_csv(tmp_path, index=False)
            csv_to_use = tmp_path
        elif csv_path is not None:
            csv_to_use = csv_path
        else:
            csv_to_use = DEFAULT_CSV

        # inject dummy detector module so legacy import resolves
        _inject_dummy_detector(pd.read_csv(csv_to_use))

        ns = runpy.run_path(LEGACY_SCRIPT, run_name='__main__')

        # legacy script populates 'all_results' and/or 'results_df'
        all_results = ns.get('all_results', None)
        results_df = ns.get('results_df', None)

        if results_df is not None:
            try:
                results_dict = results_df.to_dict(orient='records')
            except Exception:
                results_dict = []
        else:
            results_dict = all_results or []

        return results_dict, ns.get('results_df', None)

    finally:
        # don't remove csv by default to aid debugging; callers can remove
        pass
