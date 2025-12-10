import os
import sys
import subprocess
import tempfile
import pandas as pd
from typing import Optional, Tuple, Dict, Any

REPO_ROOT = os.path.dirname(__file__)
LEGACY_SCRIPT = os.path.join(REPO_ROOT, 'backtest_with_sessions.py')


def run_backtests(df: Optional[pd.DataFrame] = None, csv_path: Optional[str] = None, cleanup: bool = True) -> Tuple[list, Dict[str, Any]]:
    """Run the legacy `backtest_with_sessions.py` in a subprocess.

    If `df` is provided, it will be written to a temporary CSV and that path
    will be passed to the script. If `csv_path` is provided it will be used
    instead. Returns (trades_list, summary_dict).
    """
    tmp_file = None
    expected_csv = os.path.join(REPO_ROOT, 'LINKUSDT_8h_2020-01-01_to_2025-11-09.csv')
    try:
        if df is not None:
            df.to_csv(expected_csv, index=False)
        elif csv_path is not None and os.path.abspath(csv_path) != os.path.abspath(expected_csv):
            pd.read_csv(csv_path).to_csv(expected_csv, index=False)

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

        # Try to read the output CSV if script saved one
        output_csv = os.path.join(REPO_ROOT, 'backtest_with_sessions.csv')
        trades_list = []
        if os.path.exists(output_csv):
            trades_df = pd.read_csv(output_csv)
            trades_list = trades_df.to_dict(orient='records')

        summary = {'returncode': result.returncode}
        return trades_list, summary

    finally:
        if cleanup:
            for f in [expected_csv, os.path.join(REPO_ROOT, 'backtest_with_sessions.csv')]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception:
                    pass
