#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare Supertrend implementations (RMA vs SMA) — logging runner
"""

import sys
import logging
import pandas as pd

# ensure repo root on sys.path
sys.path.insert(0, '.')

from strategies.supertrend_strategy_fixed import SupertrendStrategyFixed
from strategies.supertrend_strategy import SupertrendStrategy


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("scripts.run_rma_fix")

    data_file = 'data/BINANCE_ETHUSDT.P, 60.csv'
    try:
        df = pd.read_csv(data_file)
    except Exception:
        logger.exception("Failed to read data file %s", data_file)
        return None

    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    strategy_old = SupertrendStrategy(atr_period=14, factor=3.0)
    strategy_new = SupertrendStrategyFixed(atr_period=14, factor=3.0)

    df_old = strategy_old.calculate_supertrend(df.copy())
    df_new = strategy_new.calculate_supertrend(df.copy())

    logger.info("Calculated supertrend for old/new implementations")

    return df_old, df_new


if __name__ == '__main__':
    main()
