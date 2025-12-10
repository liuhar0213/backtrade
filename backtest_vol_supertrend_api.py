import os
import runpy
from typing import Optional, Tuple, Dict, Any
import pandas as pd

REPO_ROOT = os.path.dirname(__file__)
LEGACY_SCRIPT = os.path.join(REPO_ROOT, 'backtest_vol_supertrend_sessions.py')
EXPECTED_CSV = os.path.join(REPO_ROOT, 'LINKUSDT_8h_2020-01-01_to_2025-11-09.csv')


def run_backtests(df: Optional[pd.DataFrame] = None, csv_path: Optional[str] = None, cleanup: bool = True) -> Tuple[list, Dict[str, Any]]:
    """Run the legacy vol_supertrend sessions script using a provided DataFrame or CSV.

    If `df` is provided it will be saved to the expected filename used by the legacy script.
    Returns (trades_list, extra_results) if available from the script's globals.
    """
    tmp_file = None
    try:
        if df is not None:
            df.to_csv(EXPECTED_CSV, index=False)
            csv_to_use = EXPECTED_CSV
        elif csv_path is not None:
            # copy or ensure path matches expected name
            if os.path.abspath(csv_path) != os.path.abspath(EXPECTED_CSV):
                # copy to expected location
                pd.read_csv(csv_path).to_csv(EXPECTED_CSV, index=False)
            csv_to_use = EXPECTED_CSV
        else:
            csv_to_use = EXPECTED_CSV

        ns = runpy.run_path(LEGACY_SCRIPT, run_name='__main__')

        # try to extract typical variables
        trades = ns.get('trades', None) or ns.get('trades_df', None) or ns.get('trades', [])
        equity_curve = ns.get('equity_curve', None)

        # normalize trades to list of dicts
        if hasattr(trades, 'to_dict'):
            trades_list = trades.to_dict(orient='records')
        elif isinstance(trades, list):
            trades_list = trades
        else:
            trades_list = []

        extra = {}
        if equity_curve is not None:
            extra['equity_curve'] = equity_curve

        return trades_list, extra
    finally:
        if cleanup:
            try:
                if os.path.exists(EXPECTED_CSV):
                    os.remove(EXPECTED_CSV)
            except Exception:
                pass
