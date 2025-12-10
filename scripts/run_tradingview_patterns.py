#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试TradingView形态检测器
"""

import sys
import pandas as pd
from tradingview_patterns import TradingViewPatternDetector

# 设置UTF-8输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():
    print("=" * 80)
    print("TradingView标准K线形态检测器 - 测试")
    print("=" * 80)
    print()

    # 加载数据
    try:
        df = pd.read_csv('data/ETHUSDT_15.csv')
        print(f"数据加载成功: {len(df)} 根K线")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 创建检测器
    print("初始化检测器（使用SMA50趋势判断）...")
    detector = TradingViewPatternDetector(df, trend_rule='SMA50')
    print("初始化完成！")
    print()

    # 检测所有形态
    print("正在检测所有42个形态...")
    patterns = detector.detect_all_patterns()
    summary = detector.get_pattern_summary()

    print()
    print("=" * 80)
    print(f"检测完成！共发现 {len(summary)} 种形态，总计 {sum(summary.values())} 次出现")
    print("=" * 80)
    print()

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

    # 展示结果
    print("【看涨形态】")
    print(f"共 {len(bullish_patterns)} 种，{bullish_count} 次出现")
    print("-" * 80)
    for pattern, count in sorted(bullish_patterns, key=lambda x: -x[1]):
        print(f"  {pattern:35s} : {count:5d} 次")
    print()

    print("【看跌形态】")
    print(f"共 {len(bearish_patterns)} 种，{bearish_count} 次出现")
    print("-" * 80)
    for pattern, count in sorted(bearish_patterns, key=lambda x: -x[1]):
        print(f"  {pattern:35s} : {count:5d} 次")
    print()

    print("【中性/延续形态】")
    print(f"共 {len(neutral_patterns)} 种，{neutral_count} 次出现")
    print("-" * 80)
    for pattern, count in sorted(neutral_patterns, key=lambda x: -x[1]):
        print(f"  {pattern:35s} : {count:5d} 次")
    print()

    # 对比原有实现
    print("=" * 80)
    print("与原有实现对比（关键形态）")
    print("=" * 80)

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
        print(f"{chn:15s} ({eng:25s}) : {count:5d} 次")

    print()
    print("=" * 80)
    print("检测完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
