#!/usr/bin/env python3
"""scripts/run_orchestrator_baseline.py

Runnable baseline smoke script for quick local CI validation. Uses
`logging` instead of prints and creates a small dataset for the test.
"""
import sys
import logging
import pandas as pd

# ensure repo root on sys.path for local imports
sys.path.insert(0, '.')

from orchestrator import ABCDEOrchestrator


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("run_orchestrator_baseline")

    try:
        logger.info("准备测试数据...")
        df = pd.read_csv('data/BTCUSDT_15.csv')
        df_small = df.head(1000)
        df_small.to_csv('data/BTCUSDT_15_test.csv', index=False)
        logger.info("测试数据已创建: %d bars", len(df_small))

        logger.info("开始基线模式回测测试")
        orch = ABCDEOrchestrator(mode='baseline')
        results = orch.run_backtest(
            data_path='data/BTCUSDT_15_test.csv',
            output_dir='results/test_baseline'
        )

        logger.info("✓ 基线模式回测测试成功!")
        logger.info("  总交易: %s", results.get('total_trades', 0))
        logger.info("  总收益: %.2f%%", results.get('total_return', 0))
        logger.info("  胜率: %.2f%%", results.get('win_rate', 0))
        logger.info("  Sharpe: %.2f", results.get('sharpe_ratio', 0))

    except Exception:
        logger.exception("✗ 基线模式测试失败")


if __name__ == '__main__':
    main()
