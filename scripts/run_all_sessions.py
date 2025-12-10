#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同时段配置，找到匹配TV的设置
"""

import pandas as pd
import numpy as np
import sys
import io

# Avoid reassigning stdout at import time (breaks pytest terminal I/O).
if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_backtest(session_config_name, session_func):
    """运行回测"""
    # 加载数据
    df = pd.read_csv('LINKUSDT_8h_2020-01-01_to_2025-11-09.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['hour'] = df['datetime'].dt.hour

    # ATR和Supertrend
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ))
    df['atr'] = df['tr'].rolling(window=10).mean()
    df['upper_band'] = (df['high'] + df['low']) / 2 + 2.5 * df['atr']
    df['lower_band'] = (df['high'] + df['low']) / 2 - 2.5 * df['atr']
    df['direction'] = 1

    for i in range(1, len(df)):
        if df['close'].iloc[i-1] <= df['upper_band'].iloc[i-1]:
            df.loc[df.index[i], 'upper_band'] = min(df['upper_band'].iloc[i], df['upper_band'].iloc[i-1])
        if df['close'].iloc[i-1] >= df['lower_band'].iloc[i-1]:
            df.loc[df.index[i], 'lower_band'] = max(df['lower_band'].iloc[i], df['lower_band'].iloc[i-1])

        if df['close'].iloc[i] > df['upper_band'].iloc[i-1]:
            df.loc[df.index[i], 'direction'] = -1
        elif df['close'].iloc[i] < df['lower_band'].iloc[i-1]:
            df.loc[df.index[i], 'direction'] = 1
        else:
            df.loc[df.index[i], 'direction'] = df['direction'].iloc[i-1]

    # 时段过滤
    df['in_session'] = df['hour'].apply(session_func)

    # 信号
    df['signal'] = 0
    df.loc[(df['direction'] < 0) & (df['direction'].shift(1) > 0) & df['in_session'], 'signal'] = 1
    df.loc[(df['direction'] > 0) & (df['direction'].shift(1) < 0) & df['in_session'], 'signal'] = -1

    # 回测
    capital = 10000
    position = 0
    entry_price = 0
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]

        if position == 1:
            stop = entry_price - 1.8 * df.iloc[entry_i]['atr']
            take = entry_price + 2 * (entry_price - stop)

            if row['low'] <= stop:
                pnl_pct = (stop / entry_price - 1) * 100
                capital *= (1 + pnl_pct / 100 * 0.5)
                trades.append(pnl_pct)
                position = 0
            elif row['high'] >= take:
                pnl_pct = (take / entry_price - 1) * 100
                capital *= (1 + pnl_pct / 100 * 0.5)
                trades.append(pnl_pct)
                position = 0

        elif position == -1:
            stop = entry_price + 1.8 * df.iloc[entry_i]['atr']
            take = entry_price - 2 * (stop - entry_price)

            if row['high'] >= stop:
                pnl_pct = (entry_price / stop - 1) * 100
                capital *= (1 + pnl_pct / 100 * 0.5)
                trades.append(pnl_pct)
                position = 0
            elif row['low'] <= take:
                pnl_pct = (entry_price / take - 1) * 100
                capital *= (1 + pnl_pct / 100 * 0.5)
                trades.append(pnl_pct)
                position = 0

        if position == 0:
            if row['signal'] == 1:
                position = 1
                entry_price = row['close']
                entry_i = i
            elif row['signal'] == -1:
                position = -1
                entry_price = row['close']
                entry_i = i

    if position != 0:
        exit_price = df.iloc[-1]['close']
        if position == 1:
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100
        capital *= (1 + pnl_pct / 100 * 0.5)
        trades.append(pnl_pct)

    return_pct = (capital / 10000 - 1) * 100
    num_trades = len(trades)

    return return_pct, num_trades

# 测试各种配置
print("="*60)
print("测试不同时段配置")
print("="*60)

configs = [
    ("无时段过滤", lambda h: True),
    ("UTC 13-22 & 7-16", lambda h: (13 <= h < 22) or (7 <= h < 16)),
    ("UTC 5-14 & 23-8", lambda h: (5 <= h < 14) or (h >= 23 or h < 8)),
    ("仅UTC 13-22", lambda h: 13 <= h < 22),
    ("仅UTC 7-16", lambda h: 7 <= h < 16),
    ("UTC 0-9 & 13-22", lambda h: (0 <= h < 9) or (13 <= h < 22)),
    ("UTC 0-8", lambda h: 0 <= h < 8),
    ("UTC 8-16", lambda h: 8 <= h < 16),
    ("UTC 16-24", lambda h: 16 <= h < 24),
]

results = []
for name, func in configs:
    ret, trades = run_backtest(name, func)
    results.append((name, ret, trades))
    print(f"{name:20s} | 收益: {ret:7.2f}% | 交易: {trades:3d}笔")

print("\n" + "="*60)
print("匹配TV结果 (74.65%, 129笔):")
print("="*60)

for name, ret, trades in results:
    if 70 <= ret <= 80 and 125 <= trades <= 135:
        print(f"✓ {name}: {ret:.2f}%, {trades}笔 [匹配!]")

print("\n最接近的配置:")
best_match = min(results, key=lambda x: abs(x[1] - 74.65) + abs(x[2] - 129) * 0.5)
print(f"→ {best_match[0]}: {best_match[1]:.2f}%, {best_match[2]}笔")
