#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Supertrend backtest (light runner)

This script provides a lightweight runner entrypoint that uses the
project's Supertrend strategy class. It avoids any top-level output or
stdout rewrapping so it is safe to import in tests.
"""

import sys
import logging
from strategies.supertrend_strategy import SupertrendStrategy
from engine.backtest import SimpleBacktester
from engine.costing import CostEngine
from engine.allocator import PositionAllocator
from utils import data_loader

# ensure repo root on sys.path
sys.path.insert(0, '.')


def main(symbol: str = 'BTC', timeframe: str = '60'):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("scripts.run_supertrend")

    logger.info("Supertrend runner start: %s %s", symbol, timeframe)

    data_file = f"data/BINANCE_{symbol}USDT.P, {timeframe}.csv"
    try:
        df = data_loader.load_csv(data_file)
    except Exception:
        logger.exception("Failed to load data %s", data_file)
        return None

    # initialize strategy and backtester
    strategy = SupertrendStrategy(atr_period=10, factor=3.0)
    df_test = strategy.calculate_supertrend(df.copy())
    df_test = strategy.generate_signals(df_test)

    entries = strategy.get_trade_entries(df_test)
    logger.info("Generated %d entries", len(entries))

    if len(entries) == 0:
        logger.warning("No entries generated, exiting")
        return None

    entries_list = []
    for entry in entries:
        entries_list.append({
            'bar_idx': entry['bar_index'],
            'side': entry['side'],
            'entry_price': entry['entry_price'],
            'stop_loss': entry.get('stop_loss'),
            'edge': 'Supertrend'
        })

    cost_engine = CostEngine()
    allocator = PositionAllocator()
    backtester = SimpleBacktester(cost_engine, allocator, initial_equity=10000.0)
    backtester.run(df_test, entries_list, symbol=symbol, timeframe=timeframe)

    logger.info("Backtest finished for %s", symbol)
    return backtester


if __name__ == '__main__':
    main('BTC', '60')
