#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume + Supertrend 回测 - 完全匹配TV的Session逻辑
1. 只允许在session时段内开仓
2. 离开session时段时强制平仓所有持仓
"""

import pandas as pd
import numpy as np
import sys
import io

# Avoid reassigning stdout at import time (breaks pytest terminal I/O).
if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 参数
ATR_PERIOD = 10
ST_FACTOR = 2.5
STOP_LOSS_ATR = 1.8
TAKE_PROFIT_RR = 2.0
INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 0.5

# Session设置（北京时间）
TRADE_LONDON_SESSION = True   # 北京7-16点 = UTC 23-8点
TRADE_NY_SESSION = True        # 北京13-22点 = UTC 5-14点

print("[*] 加载数据...")
df = pd.read_csv('LINKUSDT_8h_2020-01-01_to_2025-11-09.csv')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df['hour'] = df['datetime'].dt.hour
print(f"[+] 数据加载成功! 共 {len(df)} 根K线\n")

print("[*] Session配置（TV逻辑）:")
print(f"    伦敦时段: 北京7-16点 = UTC 23-8点  {'[启用]' if TRADE_LONDON_SESSION else '[禁用]'}")
print(f"    纽约时段: 北京13-22点 = UTC 5-14点 {'[启用]' if TRADE_NY_SESSION else '[禁用]'}")
print(f"    ⚡ 离开session时段会强制平仓所有持仓")
print()

# 计算ATR
print("[*] 开始计算指标...")
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()

# 计算Supertrend
df['upper_band'] = (df['high'] + df['low']) / 2 + ST_FACTOR * df['atr']
df['lower_band'] = (df['high'] + df['low']) / 2 - ST_FACTOR * df['atr']
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

# Session过滤
def is_in_session(hour):
    """检查是否在交易时段内"""
    in_session = False

    # 伦敦时段: UTC 23-8点（跨午夜）
    if TRADE_LONDON_SESSION:
        if hour >= 23 or hour < 8:
            in_session = True

    # 纽约时段: UTC 5-14点
    if TRADE_NY_SESSION:
        if 5 <= hour < 14:
            in_session = True

    return in_session

df['in_session'] = df['hour'].apply(is_in_session)

# 生成信号（只在session内）
df['signal'] = 0
df.loc[(df['direction'] < 0) & (df['direction'].shift(1) > 0) & df['in_session'], 'signal'] = 1
df.loc[(df['direction'] > 0) & (df['direction'].shift(1) < 0) & df['in_session'], 'signal'] = -1

print("[*] 开始回测...")

# 回测逻辑 - TV SESSION逻辑
capital = INITIAL_CAPITAL
position = 0
entry_price = 0
stop_loss = 0
take_profit = 0
entry_time = None
entry_i = 0

trades = []
session_forced_exits = 0

for i in range(1, len(df)):
    row = df.iloc[i]
    in_session_now = row['in_session']

    # ⚡ 关键：如果当前不在session内，且持有仓位，立即强制平仓
    if position != 0 and not in_session_now:
        exit_price = row['close']
        if position == 1:
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / exit_price - 1) * 100

        capital *= (1 + pnl_pct / 100 * POSITION_SIZE_PCT)

        trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': row['datetime'],
            'exit_price': exit_price,
            'type': 'LONG' if position == 1 else 'SHORT',
            'exit_reason': 'SESSION',
            'pnl_pct': pnl_pct
        })

        session_forced_exits += 1
        position = 0
        continue

    # 检查止损/止盈出场（只有在session内才检查）
    if position != 0 and in_session_now:
        if position == 1:  # 持有多头
            if row['low'] <= stop_loss:
                exit_price = stop_loss
                pnl_pct = (exit_price / entry_price - 1) * 100
                capital *= (1 + pnl_pct / 100 * POSITION_SIZE_PCT)

                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': row['datetime'],
                    'exit_price': exit_price,
                    'type': 'LONG',
                    'exit_reason': 'STOP',
                    'pnl_pct': pnl_pct
                })
                position = 0

            elif row['high'] >= take_profit:
                exit_price = take_profit
                pnl_pct = (exit_price / entry_price - 1) * 100
                capital *= (1 + pnl_pct / 100 * POSITION_SIZE_PCT)

                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': row['datetime'],
                    'exit_price': exit_price,
                    'type': 'LONG',
                    'exit_reason': 'TAKE',
                    'pnl_pct': pnl_pct
                })
                position = 0

        elif position == -1:  # 持有空头
            if row['high'] >= stop_loss:
                exit_price = stop_loss
                pnl_pct = (entry_price / exit_price - 1) * 100
                capital *= (1 + pnl_pct / 100 * POSITION_SIZE_PCT)

                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': row['datetime'],
                    'exit_price': exit_price,
                    'type': 'SHORT',
                    'exit_reason': 'STOP',
                    'pnl_pct': pnl_pct
                })
                position = 0

            elif row['low'] <= take_profit:
                exit_price = take_profit
                pnl_pct = (entry_price / exit_price - 1) * 100
                capital *= (1 + pnl_pct / 100 * POSITION_SIZE_PCT)

                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': row['datetime'],
                    'exit_price': exit_price,
                    'type': 'SHORT',
                    'exit_reason': 'TAKE',
                    'pnl_pct': pnl_pct
                })
                position = 0

    # 检查入场（只在session内）
    if position == 0 and in_session_now:
        if row['signal'] == 1:  # 做多
            position = 1
            entry_price = row['close']
            entry_time = row['datetime']
            entry_i = i
            atr = row['atr']

            stop_loss = entry_price - (atr * STOP_LOSS_ATR)
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * TAKE_PROFIT_RR)

        elif row['signal'] == -1:  # 做空
            position = -1
            entry_price = row['close']
            entry_time = row['datetime']
            entry_i = i
            atr = row['atr']

            stop_loss = entry_price + (atr * STOP_LOSS_ATR)
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * TAKE_PROFIT_RR)

# 平掉剩余仓位
if position != 0:
    exit_price = df.iloc[-1]['close']
    if position == 1:
        pnl_pct = (exit_price / entry_price - 1) * 100
    else:
        pnl_pct = (entry_price / exit_price - 1) * 100

    capital *= (1 + pnl_pct / 100 * POSITION_SIZE_PCT)

    trades.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': df.iloc[-1]['datetime'],
        'exit_price': exit_price,
        'type': 'LONG' if position == 1 else 'SHORT',
        'exit_reason': 'CLOSE',
        'pnl_pct': pnl_pct
    })

# 分析结果
print("[+] 回测完成!\n")
trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
    losing_trades = len(trades_df[trades_df['pnl_pct'] < 0])
    win_rate = winning_trades / total_trades * 100

    total_return = (capital / INITIAL_CAPITAL - 1) * 100
    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0

    print("="*60)
    print("回测结果 - TV Session逻辑（强制平仓）")
    print("="*60)
    print(f"初始资金:      ${INITIAL_CAPITAL:,.2f}")
    print(f"最终权益:      ${capital:,.2f}")
    print(f"总收益率:      {total_return:.2f}%")
    print("="*60)
    print(f"总交易次数:    {total_trades}")
    print(f"  SESSION强制平仓: {session_forced_exits} 笔")
    print(f"  止损出场:       {len(trades_df[trades_df['exit_reason']=='STOP'])} 笔")
    print(f"  止盈出场:       {len(trades_df[trades_df['exit_reason']=='TAKE'])} 笔")
    print(f"盈利次数:      {winning_trades}")
    print(f"亏损次数:      {losing_trades}")
    print(f"胜率:          {win_rate:.2f}%")
    if avg_loss != 0:
        print(f"盈亏比:        {abs(avg_win/avg_loss):.2f}")
    print("="*60)
    print(f"平均盈利:      {avg_win:.2f}%")
    print(f"平均亏损:      {avg_loss:.2f}%")
    print("="*60)

    print(f"\n与TV对比:")
    print(f"  TV结果:   74.65%, 129笔（其中21笔SESSION强制平仓）")
    print(f"  本次回测: {total_return:.2f}%, {total_trades}笔（其中{session_forced_exits}笔SESSION强制平仓）")
    print(f"  差异:     {total_return-74.65:.2f}%, {total_trades-129}笔")
    print()

    # 保存交易记录
    try:
        trades_df.to_csv('backtest_trades_tv_session_logic.csv', index=False)
        print(f"[+] 交易记录已保存: backtest_trades_tv_session_logic.csv\n")
    except:
        pass

    # 显示所有交易
    print(f"所有 {len(trades_df)} 笔交易:")
    print("="*120)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120)
    print(trades_df[['entry_time', 'entry_price', 'exit_time', 'exit_price', 'type', 'exit_reason', 'pnl_pct']].to_string(index=True))

else:
    print("[!] 没有交易记录")
