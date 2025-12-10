#!/usr/bin/env python3
"""scripts/run_orchestrator_init.py

Runnable smoke script that initializes the orchestrator in both
`baseline` and `adaptive` modes. Uses logging instead of prints so
outputs are CI-friendly.
"""
import sys
import logging

# ensure repo root on sys.path for local imports
sys.path.insert(0, '.')

from orchestrator import ABCDEOrchestrator


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("run_orchestrator_init")

    try:
        logger.info("测试基线模式初始化...")
        baseline_orch = ABCDEOrchestrator(mode='baseline')
        logger.info("✓ 基线模式初始化成功")

        logger.info("测试自适应模式初始化...")
        adaptive_orch = ABCDEOrchestrator(mode='adaptive')
        logger.info("✓ 自适应模式初始化成功")

        logger.info("所有测试通过!")

    except Exception:
        logger.exception("✗ 初始化失败")


if __name__ == "__main__":
    main()
