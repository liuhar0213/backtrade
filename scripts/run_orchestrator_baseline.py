#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试orchestrator基线模式回测
使用较小的数据集进行快速验证
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

# 创建小数据集（前1000行）
print("准备测试数据...")
df = pd.read_csv('data/BTCUSDT_15.csv')
df_small = df.head(1000)
df_small.to_csv('data/BTCUSDT_15_test.csv', index=False)
print(f"测试数据已创建: {len(df_small)} bars")

# 运行基线模式
print("\n" + "=" * 80)
print("开始基线模式回测测试")
print("=" * 80)

try:
    orch = ABCDEOrchestrator(mode='baseline')
    results = orch.run_backtest(
        data_path='data/BTCUSDT_15_test.csv',
        output_dir='results/test_baseline'
    )

    print("\n✓ 基线模式回测测试成功!")
    print(f"  总交易: {results.get('total_trades', 0)}")
    print(f"  总收益: {results.get('total_return', 0):.2f}%")
    print(f"  胜率: {results.get('win_rate', 0):.2f}%")
    print(f"  Sharpe: {results.get('sharpe_ratio', 0):.2f}")

except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
