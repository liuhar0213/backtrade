"""
测试改进后的锤形线和上吊线检测
对比优化前后的效果
"""

import pandas as pd
from find_correct_patterns import CorrectGoldenKDetector

# 读取数据
df = pd.read_csv('data/ETHUSDT_15.csv')

# 转换时间列（保持与backtest_with_fees.py一致）
df['timestamp'] = pd.to_datetime(df['time'])

# 筛选2025-05-01之后的数据（保持与backtest一致）
df = df[df['timestamp'] >= '2025-05-01'].reset_index(drop=True)

# 创建检测器
detector = CorrectGoldenKDetector(df)

# 检测锤形线
hammer_signals = []
for i in range(len(detector.df)):
    if detector.detect_hammer(i):
        hammer_signals.append({
            'index': i,
            'time': detector.df.iloc[i]['datetime'],
            'close': detector.df.iloc[i]['close'],
            'ma20': detector.df.iloc[i]['ma20'],
            'ma60': detector.df.iloc[i]['ma60'],
            'body_ratio': detector.df.iloc[i]['body_ratio'],
            'lower_shadow': detector.df.iloc[i]['lower_shadow'],
            'upper_shadow': detector.df.iloc[i]['upper_shadow']
        })

# 检测上吊线
hanging_man_signals = []
for i in range(len(detector.df)):
    if detector.detect_hanging_man(i):
        hanging_man_signals.append({
            'index': i,
            'time': detector.df.iloc[i]['datetime'],
            'close': detector.df.iloc[i]['close'],
            'ma20': detector.df.iloc[i]['ma20'],
            'ma60': detector.df.iloc[i]['ma60'],
            'body_ratio': detector.df.iloc[i]['body_ratio'],
            'lower_shadow': detector.df.iloc[i]['lower_shadow'],
            'upper_shadow': detector.df.iloc[i]['upper_shadow']
        })

print("="*100)
print("改进后的锤形线和上吊线检测结果")
print("="*100)

print(f"\n锤形线（Hammer）检测结果:")
print(f"  检测到 {len(hammer_signals)} 个信号")
print(f"\n前10个锤形线信号:")
print(f"{'时间':<20} {'收盘价':<10} {'MA20':<10} {'MA60':<10} {'实体占比':<10} {'趋势确认':<10}")
print("-"*80)

for signal in hammer_signals[:10]:
    price = signal['close']
    ma20 = signal['ma20']
    ma60 = signal['ma60']

    # 验证趋势（应该是下跌趋势）
    trend_ok = "OK" if (price < ma20 and ma20 < ma60) else "ERROR"

    print(f"{signal['time']:<20} {price:<10.2f} {ma20:<10.2f} {ma60:<10.2f} "
          f"{signal['body_ratio']*100:<10.1f}% {trend_ok:<10}")

print(f"\n\n上吊线（Hanging Man）检测结果:")
print(f"  检测到 {len(hanging_man_signals)} 个信号")
print(f"\n前10个上吊线信号:")
print(f"{'时间':<20} {'收盘价':<10} {'MA20':<10} {'MA60':<10} {'实体占比':<10} {'趋势确认':<10}")
print("-"*80)

for signal in hanging_man_signals[:10]:
    price = signal['close']
    ma20 = signal['ma20']
    ma60 = signal['ma60']

    # 验证趋势（应该是上涨趋势）
    trend_ok = "OK" if (price > ma20 and ma20 > ma60) else "ERROR"

    print(f"{signal['time']:<20} {price:<10.2f} {ma20:<10.2f} {ma60:<10.2f} "
          f"{signal['body_ratio']*100:<10.1f}% {trend_ok:<10}")

print(f"\n\n改进点总结:")
print(f"  1. 加入MA20/MA60趋势判断")
print(f"  2. 加入实体位置验证（实体下沿>K线中点）")
print(f"  3. 更严格的上影线要求（从10%改为5%）")

print(f"\n预期效果:")
print(f"  - 信号数量减少（提高质量）")
print(f"  - 胜率提高")
print(f"  - 锤形线预期从-1.50%改善到+1.5%~+3.0%")
print(f"  - 参考射击之星（镜像形态）：+4.21%，66.67%胜率")

print(f"\n下一步:")
print(f"  运行: python backtest_with_fees.py")
print(f"  对比优化前后的完整回测结果")
