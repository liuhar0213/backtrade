#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume + Supertrend + Sessions 回测
完全按照TradingView参数配置
"""

import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ========== 参数配置 (完全对应TradingView) ==========
ATR_PERIOD = 10
ST_FACTOR = 2.5
STOP_LOSS_ATR = 1.8
TAKE_PROFIT_RR = 2.0

# 交易时段 (按用户TV设置，北京时间转UTC)
# 北京时间13-22点 = UTC 5-14点
# 北京时间7-16点 = UTC 23-8点（前一天23点到当天8点）
TRADE_NY_SESSION = True      # "纽约": UTC 5:00-14:00 (北京13-22)
TRADE_LONDON_SESSION = True  # "伦敦": UTC 23:00-08:00 (北京7-16，跨天)
TRADE_TOKYO_SESSION = False
TRADE_SYDNEY_SESSION = False
USE_SESSION_FILTER = True

# 资金管理
INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 0.5  # 50%权益
COMMISSION = 0.001       # 0.1%
SLIPPAGE = 0.0002        # 2 标记号 ≈ 0.02%

# ========== 加载数据 ==========
print("[*] 加载数据...")
df = pd.read_csv('LINKUSDT_8h_2020-01-01_to_2025-11-09.csv')
print(f"[+] 数据加载成功! 共 {len(df)} 根K线\n")

# 转换时间为UTC
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df['hour'] = df['datetime'].dt.hour

print("="*60)
print("回测配置")
print("="*60)
print(f"交易对:        LINKUSDT永续合约")
print(f"时间周期:      8小时")
print(f"数据范围:      {df['datetime'].iloc[0]} 至 {df['datetime'].iloc[-1]}")
print(f"总K线数:       {len(df)}")
print(f"初始资金:      ${INITIAL_CAPITAL:,.0f}")
print(f"仓位大小:      {POSITION_SIZE_PCT*100}% 权益")
print(f"手续费:        {COMMISSION*100}%")
print(f"滑点:          {SLIPPAGE*100}%")
print("="*60)
print("Supertrend参数:")
print(f"  ATR Length:  {ATR_PERIOD}")
print(f"  ATR Factor:  {ST_FACTOR}")
print("="*60)
print("止损止盈:")
print(f"  止损类型:    ATR")
print(f"  止损倍数:    {STOP_LOSS_ATR}x ATR")
print(f"  止盈比例:    {TAKE_PROFIT_RR}:1")
print("="*60)
print("交易时段过滤 (按北京时间设置):")
print(f"  启用过滤:    {USE_SESSION_FILTER}")
print(f"  '纽约'时段:  {TRADE_NY_SESSION} (北京13-22点 = UTC 5-14点)")
print(f"  '伦敦'时段:  {TRADE_LONDON_SESSION} (北京7-16点 = UTC 23-8点)")
print(f"  东京时段:    {TRADE_TOKYO_SESSION}")
print(f"  悉尼时段:    {TRADE_SYDNEY_SESSION}")
print("="*60)

# ========== 计算交易时段 ==========
print("\n[*] 计算交易时段...")

def is_in_session(hour):
    """判断是否在允许交易的时段（按北京时间设置转换为UTC）"""
    if not USE_SESSION_FILTER:
        return True

    in_session = False

    # "纽约"时段: 北京13-22点 = UTC 5-14点
    if TRADE_NY_SESSION and (5 <= hour < 14):
        in_session = True

    # "伦敦"时段: 北京7-16点 = UTC 23-8点（跨天）
    if TRADE_LONDON_SESSION and (hour >= 23 or hour < 8):
        in_session = True

    # 东京时段（未使用）
    if TRADE_TOKYO_SESSION and (0 <= hour < 9):
        in_session = True

    # 悉尼时段（未使用）
    if TRADE_SYDNEY_SESSION and (hour >= 21 or hour < 6):
        in_session = True

    return in_session

df['in_session'] = df['hour'].apply(is_in_session)

# 统计时段覆盖率
total_bars = len(df)
session_bars = df['in_session'].sum()
session_coverage = session_bars / total_bars * 100

print(f"[+] 时段过滤完成")
print(f"    总K线数:           {total_bars}")
print(f"    时段内K线:         {session_bars}")
print(f"    时段覆盖率:        {session_coverage:.2f}%")

# ========== 计算Supertrend指标 ==========
print("\n[*] 计算Supertrend指标...")

# 计算ATR
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
df['supertrend'] = 0.0
df['direction'] = 1

for i in range(1, len(df)):
    # 更新上轨
    if df['close'].iloc[i-1] <= df['upper_band'].iloc[i-1]:
        df.loc[df.index[i], 'upper_band'] = min(df['upper_band'].iloc[i], df['upper_band'].iloc[i-1])

    # 更新下轨
    if df['close'].iloc[i-1] >= df['lower_band'].iloc[i-1]:
        df.loc[df.index[i], 'lower_band'] = max(df['lower_band'].iloc[i], df['lower_band'].iloc[i-1])

    # 确定方向
    if df['close'].iloc[i] > df['upper_band'].iloc[i-1]:
        df.loc[df.index[i], 'direction'] = -1  # 上升趋势
    elif df['close'].iloc[i] < df['lower_band'].iloc[i-1]:
        df.loc[df.index[i], 'direction'] = 1   # 下降趋势
    else:
        df.loc[df.index[i], 'direction'] = df['direction'].iloc[i-1]

    # 设置Supertrend值
    if df['direction'].iloc[i] == -1:
        df.loc[df.index[i], 'supertrend'] = df['lower_band'].iloc[i]
    else:
        df.loc[df.index[i], 'supertrend'] = df['upper_band'].iloc[i]

print("[+] Supertrend计算完成")

# ========== 生成交易信号 ==========
print("\n[*] 生成交易信号...")

# Supertrend反转 + 时段过滤
df['st_to_uptrend'] = (df['direction'] < 0) & (df['direction'].shift(1) > 0)
df['st_to_downtrend'] = (df['direction'] > 0) & (df['direction'].shift(1) < 0)

# 多头信号 = Supertrend转多 + 在交易时段
df['long_signal'] = df['st_to_uptrend'] & df['in_session']

# 空头信号 = Supertrend转空 + 在交易时段
df['short_signal'] = df['st_to_downtrend'] & df['in_session']

total_long_signals = df['long_signal'].sum()
total_short_signals = df['short_signal'].sum()

print(f"[+] 信号生成完成")
print(f"    多头信号:          {total_long_signals}")
print(f"    空头信号:          {total_short_signals}")
print(f"    总信号:            {total_long_signals + total_short_signals}")

# ========== 回测逻辑 ==========
print("\n[*] 开始回测...")
print("="*60)

capital = INITIAL_CAPITAL
position = 0  # 0: 无仓位, 1: 多头, -1: 空头
entry_price = 0
stop_loss = 0
take_profit = 0
entry_time = None
entry_bar = 0

trades = []
equity_curve = [capital]

for i in range(1, len(df)):
    row = df.iloc[i]

    # ========== 检查出场条件 ==========
    if position == 1:  # 持有多头
        # 检查止损
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
                'exit_reason': 'STOP LOSS',
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_bar
            })
            position = 0

        # 检查止盈
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
                'exit_reason': 'TAKE PROFIT',
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_bar
            })
            position = 0

    elif position == -1:  # 持有空头
        # 检查止损
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
                'exit_reason': 'STOP LOSS',
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_bar
            })
            position = 0

        # 检查止盈
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
                'exit_reason': 'TAKE PROFIT',
                'pnl_pct': pnl_pct,
                'bars_held': i - entry_bar
            })
            position = 0

    # ========== 检查入场条件 ==========
    if position == 0:
        # 做多信号
        if row['long_signal']:
            position = 1
            entry_price = row['close'] * (1 + SLIPPAGE)
            entry_time = row['datetime']
            entry_bar = i
            atr = row['atr']

            # 计算止损止盈
            stop_loss = entry_price - (atr * STOP_LOSS_ATR)
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * TAKE_PROFIT_RR)

        # 做空信号
        elif row['short_signal']:
            position = -1
            entry_price = row['close'] * (1 - SLIPPAGE)
            entry_time = row['datetime']
            entry_bar = i
            atr = row['atr']

            # 计算止损止盈
            stop_loss = entry_price + (atr * STOP_LOSS_ATR)
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * TAKE_PROFIT_RR)

    # 记录权益曲线
    equity_curve.append(capital)

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
        'exit_reason': 'FINAL CLOSE',
        'pnl_pct': pnl_pct,
        'bars_held': len(df) - 1 - entry_bar
    })

print("[+] 回测完成!\n")

# ========== 分析结果 ==========
if len(trades) == 0:
    print("[!] 没有交易记录!")
    sys.exit(0)

trades_df = pd.DataFrame(trades)

# 基础统计
total_trades = len(trades_df)
winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
losing_trades = len(trades_df[trades_df['pnl_pct'] < 0])
win_rate = winning_trades / total_trades * 100

total_return = (capital / INITIAL_CAPITAL - 1) * 100
avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0

# 利润因子
total_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
total_loss = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

# 最大回撤
equity_series = pd.Series(equity_curve)
running_max = equity_series.expanding().max()
drawdown = (equity_series - running_max) / running_max * 100
max_drawdown = drawdown.min()

# 持仓时长统计
avg_bars_held = trades_df['bars_held'].mean()
avg_hours_held = avg_bars_held * 8  # 8小时周期

# 多空统计
long_trades = trades_df[trades_df['type'] == 'LONG']
short_trades = trades_df[trades_df['type'] == 'SHORT']

# 计算年化收益
start_date = df['datetime'].iloc[0]
end_date = df['datetime'].iloc[-1]
days = (end_date - start_date).days
years = days / 365.25
annualized_return = ((capital / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

# 输出结果
print("="*60)
print("回测结果汇总")
print("="*60)
print(f"数据范围:      {start_date} 至 {end_date}")
print(f"回测天数:      {days}天 ({years:.2f}年)")
print("="*60)
print(f"初始资金:      ${INITIAL_CAPITAL:,.2f}")
print(f"最终权益:      ${capital:,.2f}")
print(f"净盈亏:        ${capital - INITIAL_CAPITAL:,.2f}")
print(f"总收益率:      {total_return:+.2f}%")
print(f"年化收益率:    {annualized_return:.2f}%")
print(f"最大回撤:      {max_drawdown:.2f}%")
print("="*60)
print(f"总交易次数:    {total_trades}")
print(f"盈利次数:      {winning_trades} ({winning_trades/total_trades*100:.1f}%)")
print(f"亏损次数:      {losing_trades} ({losing_trades/total_trades*100:.1f}%)")
print(f"胜率:          {win_rate:.2f}%")
print(f"盈亏比:        {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "盈亏比:        N/A")
print(f"利润因子:      {profit_factor:.2f}")
print("="*60)
print(f"平均盈利:      +{avg_win:.2f}%")
print(f"平均亏损:      {avg_loss:.2f}%")
print(f"最大单笔盈利:  +{trades_df['pnl_pct'].max():.2f}%")
print(f"最大单笔亏损:  {trades_df['pnl_pct'].min():.2f}%")
print("="*60)
print(f"多头交易:      {len(long_trades)}笔")
if len(long_trades) > 0:
    print(f"  胜率:        {len(long_trades[long_trades['pnl_pct'] > 0]) / len(long_trades) * 100:.1f}%")
    print(f"  平均收益:    {long_trades['pnl_pct'].mean():.2f}%")
else:
    print("  胜率:        N/A")
    print("  平均收益:    N/A")

print(f"空头交易:      {len(short_trades)}笔")
if len(short_trades) > 0:
    print(f"  胜率:        {len(short_trades[short_trades['pnl_pct'] > 0]) / len(short_trades) * 100:.1f}%")
    print(f"  平均收益:    {short_trades['pnl_pct'].mean():.2f}%")
else:
    print("  胜率:        N/A")
    print("  平均收益:    N/A")
print("="*60)
print(f"平均持仓时长:  {avg_bars_held:.1f}根K线 ({avg_hours_held:.0f}小时)")
print("="*60)

# 按年份统计
trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
yearly_stats = trades_df.groupby('year').agg({
    'pnl_pct': ['count', 'mean', 'sum']
}).round(2)
yearly_stats.columns = ['交易次数', '平均收益%', '累计收益%']

print("\n年度统计:")
print(yearly_stats)

# 保存结果
trades_df.to_csv('backtest_with_sessions.csv', index=False)
print(f"\n[+] 交易记录已保存: backtest_with_sessions.csv")

# 显示最近10笔交易
print(f"\n最近10笔交易:")
print(trades_df[['entry_time', 'entry_price', 'exit_time', 'exit_price', 'type', 'exit_reason', 'pnl_pct', 'bars_held']].tail(10).to_string(index=False))

print("\n" + "="*60)
print("[SUCCESS] 回测完成!")
print("="*60)
