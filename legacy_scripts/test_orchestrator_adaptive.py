#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试orchestrator自适应模式回测
验证D层监督和Genome层参数演化
"""
import sys
import io
import pandas as pd

from orchestrator import ABCDEOrchestrator

# 设置UTF-8编码（在导入之后）
if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# 创建测试数据（更大数据集以触发监督）
print("准备测试数据...")
df = pd.read_csv('data/BTCUSDT_15.csv')
df_test = df.head(2000)  # 使用2000 bars以覆盖多个监督周期
df_test.to_csv('data/BTCUSDT_15_adaptive_test.csv', index=False)
print(f"测试数据已创建: {len(df_test)} bars")
print(f"  将触发监督检查: {len(df_test) // 50} 次（每50 bars）")

# 运行自适应模式
print("\n" + "=" * 80)
print("开始自适应模式回测测试")
print("=" * 80)

try:
    orch = ABCDEOrchestrator(mode='adaptive')
    results = orch.run_backtest(
        data_path='data/BTCUSDT_15_adaptive_test.csv',
        output_dir='results/test_adaptive'
    )

    print("\n✓ 自适应模式回测测试成功!")
    print(f"  总交易: {results.get('total_trades', 0)}")
    print(f"  总收益: {results.get('total_return', 0):.2f}%")
    print(f"  胜率: {results.get('win_rate', 0):.2f}%")
    print(f"  Sharpe: {results.get('sharpe_ratio', 0):.2f}")

    # 检查监督历史
    supervision_history = results.get('supervision_history', [])
    print(f"\n监督检查:")
    print(f"  触发次数: {len(supervision_history)}")

    if len(supervision_history) > 0:
        print(f"  首次检查: bar {supervision_history[0]['bar']}")
        print(f"  末次检查: bar {supervision_history[-1]['bar']}")

        # 统计建议数量
        total_suggestions = sum(s['suggestions'] for s in supervision_history)
        print(f"  总建议数: {total_suggestions}")

    # 检查版本演化
    from Genome.version_tracker import VersionTracker
    tracker = VersionTracker()
    versions = tracker.version_tree['versions']
    print(f"\n参数演化:")
    print(f"  版本数: {len(versions)}")
    if len(versions) > 1:
        print(f"  初始版本: {versions[0]['version']}")
        print(f"  最新版本: {versions[-1]['version']}")

except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
