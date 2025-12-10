import os
import sys
import subprocess
from typing import Optional, Tuple, Dict, Any
import pandas as pd

REPO_ROOT = os.path.dirname(__file__)
LEGACY_SCRIPT = os.path.join(REPO_ROOT, 'backtest_vol_supertrend_sessions.py')
EXPECTED_CSV = os.path.join(REPO_ROOT, 'LINKUSDT_8h_2020-01-01_to_2025-11-09.csv')


def run_backtests(df: Optional[pd.DataFrame] = None, csv_path: Optional[str] = None, cleanup: bool = True) -> Tuple[list, Dict[str, Any]]:
    """Run the legacy vol_supertrend sessions script in a subprocess.

    If `df` is provided it will be saved to the expected filename used by the legacy script.
    Returns (trades_list, extra_results).
    """
    try:
        if df is not None:
            df.to_csv(EXPECTED_CSV, index=False)
        elif csv_path is not None:
            if os.path.abspath(csv_path) != os.path.abspath(EXPECTED_CSV):
                pd.read_csv(csv_path).to_csv(EXPECTED_CSV, index=False)

        # Run the legacy script in a subprocess to isolate stdout reassignment
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

        # Check for any output CSV the script might have saved
        trades_list = []
        for csv_name in ['backtest_vol_supertrend_sessions.csv', 'trades.csv']:
            output_csv = os.path.join(REPO_ROOT, csv_name)
            if os.path.exists(output_csv):
                trades_df = pd.read_csv(output_csv)
                trades_list = trades_df.to_dict(orient='records')
                break

        extra = {'returncode': result.returncode}
        return trades_list, extra
    finally:
        if cleanup:
            for f in [EXPECTED_CSV]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception:
                    pass
