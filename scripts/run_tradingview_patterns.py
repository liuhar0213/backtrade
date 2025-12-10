#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试TradingView形态检测器
"""

import sys
import logging
import pandas as pd
from tradingview_patterns import TradingViewPatternDetector

# ensure repo root on sys.path
sys.path.insert(0, '.')


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("run_tradingview_patterns")

    logger.info("TradingView标准K线形态检测器 - 测试")

    # 加载数据
    try:
        df = pd.read_csv('data/ETHUSDT_15.csv')
        logger.info("数据加载成功: %d 根K线", len(df))
    except Exception as e:
        logger.exception("数据加载失败")
        return

    # 创建检测器
    logger.info("初始化检测器（使用SMA50趋势判断）...")
    detector = TradingViewPatternDetector(df, trend_rule='SMA50')
    logger.info("初始化完成！")

    # 检测所有形态
    logger.info("正在检测所有42个形态...")
    patterns = detector.detect_all_patterns()
    summary = detector.get_pattern_summary()

    logger.info("检测完成！共发现 %d 种形态，总计 %d 次出现", len(summary), sum(summary.values()))

    # 分类统计
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    bullish_keywords = ['Bullish', 'Hammer', 'Piercing', 'Morning', 'White_Soldiers', 'Bottom', 'Rising', 'Upside', 'Dragonfly']
    bearish_keywords = ['Bearish', 'Shooting', 'Hanging', 'Dark_Cloud', 'Evening', 'Black_Crows', 'Top', 'Falling', 'Downside', 'Gravestone']

    bullish_patterns = []
    bearish_patterns = []
    neutral_patterns = []

    for pattern, count in summary.items():
        is_bullish = any(keyword in pattern for keyword in bullish_keywords)
        is_bearish = any(keyword in pattern for keyword in bearish_keywords) and not is_bullish

        if is_bullish:
            bullish_patterns.append((pattern, count))
            bullish_count += count
        elif is_bearish:
            bearish_patterns.append((pattern, count))
            bearish_count += count
        else:
            neutral_patterns.append((pattern, count))
            neutral_count += count

    # 展示结果 (日志输出)
    logger.info("【看涨形态】 共 %d 种，%d 次出现", len(bullish_patterns), bullish_count)
    for pattern, count in sorted(bullish_patterns, key=lambda x: -x[1]):
        logger.info("  %s : %d 次", pattern, count)

    logger.info("【看跌形态】 共 %d 种，%d 次出现", len(bearish_patterns), bearish_count)
    for pattern, count in sorted(bearish_patterns, key=lambda x: -x[1]):
        logger.info("  %s : %d 次", pattern, count)

    logger.info("【中性/延续形态】 共 %d 种，%d 次出现", len(neutral_patterns), neutral_count)
    for pattern, count in sorted(neutral_patterns, key=lambda x: -x[1]):
        logger.info("  %s : %d 次", pattern, count)

    # 对比原有实现
    key_patterns = {
        'Hammer': '锤形线',
        'Shooting_Star': '射击之星',
        'Piercing': '刺透',
        'Dark_Cloud_Cover': '乌云盖顶',
        'Morning_Star': '早晨之星',
        'Evening_Star': '黄昏之星',
        'Three_White_Soldiers': '三白兵',
        'Three_Black_Crows': '三只乌鸦'
    }

    for eng, chn in key_patterns.items():
        count = summary.get(eng, 0)
        logger.info("%s (%s) : %d 次", chn, eng, count)


if __name__ == '__main__':
    main()
