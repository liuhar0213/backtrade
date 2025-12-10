#!/usr/bin/env python3
"""scripts/run_orchestrator_adaptive.py

Runnable adaptive-mode smoke script for local CI. Uses logging and
creates a larger dataset to trigger supervision logic.
"""
import sys
import logging
import pandas as pd

# ensure repo root on sys.path for local imports
sys.path.insert(0, '.')

from orchestrator import ABCDEOrchestrator


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("run_orchestrator_adaptive")

    try:
        logger.info("准备测试数据...")
        df = pd.read_csv('data/BTCUSDT_15.csv')
        df_test = df.head(2000)  # 使用2000 bars以覆盖多个监督周期
        df_test.to_csv('data/BTCUSDT_15_adaptive_test.csv', index=False)
        logger.info("测试数据已创建: %d bars", len(df_test))
        logger.info("将触发监督检查: %d 次（每50 bars）", len(df_test) // 50)

        logger.info("开始自适应模式回测测试")
        orch = ABCDEOrchestrator(mode='adaptive')
        results = orch.run_backtest(
            data_path='data/BTCUSDT_15_adaptive_test.csv',
            output_dir='results/test_adaptive'
        )

        logger.info("✓ 自适应模式回测测试成功!")
        logger.info("  总交易: %s", results.get('total_trades', 0))
        logger.info("  总收益: %.2f%%", results.get('total_return', 0))
        logger.info("  胜率: %.2f%%", results.get('win_rate', 0))
        logger.info("  Sharpe: %.2f", results.get('sharpe_ratio', 0))

        # 检查监督历史
        supervision_history = results.get('supervision_history', [])
        logger.info("监督检查: 触发次数=%d", len(supervision_history))

        if len(supervision_history) > 0:
            logger.info("首次检查: bar %s", supervision_history[0]['bar'])
            logger.info("末次检查: bar %s", supervision_history[-1]['bar'])
            total_suggestions = sum(s['suggestions'] for s in supervision_history)
            logger.info("总建议数: %d", total_suggestions)

        # 检查版本演化
        from Genome.version_tracker import VersionTracker
        tracker = VersionTracker()
        versions = tracker.version_tree.get('versions', [])
        logger.info("参数演化: 版本数=%d", len(versions))
        if len(versions) > 1:
            logger.info("初始版本: %s", versions[0]['version'])
            logger.info("最新版本: %s", versions[-1]['version'])

    except Exception:
        logger.exception("✗ 自适应模式测试失败")


if __name__ == '__main__':
    main()
