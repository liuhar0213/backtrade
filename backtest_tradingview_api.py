import os
import subprocess
import sys
import tempfile
from typing import Optional, Tuple, Dict, Any
import pandas as pd

REPO_ROOT = os.path.dirname(__file__)
LEGACY_SCRIPT = os.path.join(REPO_ROOT, 'backtest_tradingview_patterns.py')
DATA_DIR = os.path.join(REPO_ROOT, 'data')
DEFAULT_CSV = os.path.join(DATA_DIR, 'ETHUSDT_15.csv')

# Path to the dummy detector module we'll create for testing
DUMMY_DETECTOR_CODE = '''
import pandas as pd

class TradingViewPatternDetectorOptimized:
    def __init__(self, df, trend_rule='SMA50'):
        self.df = df

    def detect_all_patterns(self):
        s = pd.Series([False] * len(self.df))
        if len(self.df) > 5:
            s.iloc[min(10, len(self.df)-1)] = True
        return pd.DataFrame({'DummyPattern': s})
'''


def run_backtests(df: Optional[pd.DataFrame] = None, csv_path: Optional[str] = None, save_csv: bool = False) -> Tuple[list, Dict[str, Any]]:
    """Run the legacy `backtest_tradingview_patterns.py` in a subprocess with a dummy detector.

    If `df` is provided it will be saved to `data/ETHUSDT_15.csv` and used.
    Returns (results_list, results_df) where results_list is a list of dicts.
    """
    # Ensure data dir exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Create the dummy detector module file
    dummy_detector_path = os.path.join(REPO_ROOT, 'tradingview_patterns_optimized.py')
    detector_existed = os.path.exists(dummy_detector_path)
    original_content = None
    if detector_existed:
        with open(dummy_detector_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

    try:
        if df is not None:
            df.to_csv(DEFAULT_CSV, index=False)
        elif csv_path is not None and os.path.abspath(csv_path) != os.path.abspath(DEFAULT_CSV):
            pd.read_csv(csv_path).to_csv(DEFAULT_CSV, index=False)

        # Write dummy detector module
        with open(dummy_detector_path, 'w', encoding='utf-8') as f:
            f.write(DUMMY_DETECTOR_CODE)

        # Run the legacy script in a subprocess
        result = subprocess.run(
            [sys.executable, LEGACY_SCRIPT],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=120,
            encoding='utf-8',
            errors='replace',
        )
        # Print output for visibility (pytest will capture)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Check for output CSV
        results_csv = os.path.join(REPO_ROOT, 'tradingview_backtest_results.csv')
        results_list = []
        results_df = None
        if os.path.exists(results_csv):
            results_df = pd.read_csv(results_csv)
            results_list = results_df.to_dict(orient='records')

        return results_list, results_df

    finally:
        # Restore or remove the dummy detector module
        if original_content is not None:
            with open(dummy_detector_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
        elif not detector_existed and os.path.exists(dummy_detector_path):
            try:
                os.remove(dummy_detector_path)
            except Exception:
                pass
