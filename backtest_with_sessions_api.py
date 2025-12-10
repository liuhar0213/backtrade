import os
import runpy
import tempfile
import pandas as pd
from typing import Optional, Tuple, Dict, Any

REPO_ROOT = os.path.dirname(__file__)
LEGACY_SCRIPT = os.path.join(REPO_ROOT, 'backtest_with_sessions.py')


def run_backtests(df: Optional[pd.DataFrame] = None, csv_path: Optional[str] = None, cleanup: bool = True) -> Tuple[list, Dict[str, Any]]:
    """Run the legacy `backtest_with_sessions.py` in an isolated namespace.

    If `df` is provided, it will be written to a temporary CSV and that path
    will be passed to the script. If `csv_path` is provided it will be used
    instead. Returns (trades_list, summary_dict) if available from the script's
    globals; otherwise returns ([], {}).
    """
    tmp_file = None
    try:
        if df is not None:
            tmp_fd, tmp_file = tempfile.mkstemp(prefix='bws_', suffix='.csv', dir=REPO_ROOT)
            os.close(tmp_fd)
            df.to_csv(tmp_file, index=False)
            csv_to_use = tmp_file
        elif csv_path is not None:
            csv_to_use = csv_path
        else:
            # default file expected by the legacy script
            csv_to_use = os.path.join(REPO_ROOT, 'LINKUSDT_8h_2020-01-01_to_2025-11-09.csv')

        # The legacy script expects a specific filename; many scripts reference the CSV directly.
        # To ensure it uses our csv_to_use, set CWD to REPO_ROOT and set an expected filename if needed.
        # We'll call runpy.run_path to execute the script and capture its globals.
        ns = runpy.run_path(LEGACY_SCRIPT, run_name='__main__')

        # Try to extract common results
        trades = ns.get('trades', None) or ns.get('trades_df', None) or ns.get('all_trades', None) or []
        # If trades is a DataFrame, convert to list of dicts
        if hasattr(trades, 'to_dict'):
            trades_list = trades.to_dict(orient='records')
        elif isinstance(trades, list):
            trades_list = trades
        else:
            trades_list = []

        summary = ns.get('summary', None) or ns.get('pattern_stats', None) or {}

        return trades_list, summary

    finally:
        if cleanup and tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass
