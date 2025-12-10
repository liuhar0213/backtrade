#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume + Supertrend 简化回测 (可调用版)

This file preserves the original script behavior when run as a script,
and also exposes a `run_backtest()` function suitable for unit tests.
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import logging


# Default parameters (match original script)
ATR_PERIOD = 10
ST_FACTOR = 2.5
STOP_LOSS_ATR = 1.8
TAKE_PROFIT_RR = 2.0
INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 0.5
COMMISSION = 0.001


def run_backtest(df: Optional[pd.DataFrame] = None,
                 csv_file: str = 'LINKUSDT_8h_2020-01-01_to_2025-11-09.csv',
                 save_csv: bool = True,
                 show_trades: bool = False) -> Dict[str, Any]:
    """Run the backtest and return results.

    If `df` is provided, it will be used directly. Otherwise the CSV at
    `csv_file` will be read.
    Returns a dictionary with keys: `capital`, `trades_df`, `equity_curve`, `stats`.
    """
    log = logging.getLogger('backtest_simple')

    if df is None:
        log.info("[*] 加载数据... %s", csv_file)
        df = pd.read_csv(csv_file)
        log.info("[+] 数据加载成功! 共 %d 根K线", len(df))

    # compute indicators
    log.info("[*] 开始计算指标...")
    df = df.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()

    df['upper_band'] = (df['high'] + df['low']) / 2 + ST_FACTOR * df['atr']
    df['lower_band'] = (df['high'] + df['low']) / 2 - ST_FACTOR * df['atr']
    df['supertrend'] = 0.0
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

        if df['direction'].iloc[i] == -1:
            df.loc[df.index[i], 'supertrend'] = df['lower_band'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend'] = df['upper_band'].iloc[i]

    df['signal'] = 0
    df.loc[(df['direction'] < 0) & (df['direction'].shift(1) > 0), 'signal'] = 1
    df.loc[(df['direction'] > 0) & (df['direction'].shift(1) < 0), 'signal'] = -1

    # backtest loop
    log.info("[*] 开始回测...")
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_time = None

    trades = []
    equity_curve = [capital]

    for i in range(1, len(df)):
        row = df.iloc[i]

        # exits
        if position == 1:
            if row['low'] <= stop_loss:
                exit_price = stop_loss
                pnl_pct = (exit_price / entry_price - 1) * 100
                capital *= (1 + pnl_pct / 100 * POSITION_SIZE_PCT)

                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': row.get('datetime', None),
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
                    'exit_time': row.get('datetime', None),
                    'exit_price': exit_price,
                    'type': 'LONG',
                    'exit_reason': 'TAKE',
                    'pnl_pct': pnl_pct
                })
                position = 0

        elif position == -1:
            if row['high'] >= stop_loss:
                exit_price = stop_loss
                pnl_pct = (entry_price / exit_price - 1) * 100
                capital *= (1 + pnl_pct / 100 * POSITION_SIZE_PCT)

                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': row.get('datetime', None),
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
                    'exit_time': row.get('datetime', None),
                    'exit_price': exit_price,
                    'type': 'SHORT',
                    'exit_reason': 'TAKE',
                    'pnl_pct': pnl_pct
                })
                position = 0

        # entries
        if position == 0:
            if row['signal'] == 1:
                position = 1
                entry_price = row['close']
                entry_time = row.get('datetime', None)
                atr = row['atr']

                stop_loss = entry_price - (atr * STOP_LOSS_ATR)
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * TAKE_PROFIT_RR)

            elif row['signal'] == -1:
                position = -1
                entry_price = row['close']
                entry_time = row.get('datetime', None)
                atr = row['atr']

                stop_loss = entry_price + (atr * STOP_LOSS_ATR)
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * TAKE_PROFIT_RR)

        equity_curve.append(capital)

    # close remaining position
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
            'exit_time': df.iloc[-1].get('datetime', None),
            'exit_price': exit_price,
            'type': 'LONG' if position == 1 else 'SHORT',
            'exit_reason': 'CLOSE',
            'pnl_pct': pnl_pct
        })

    trades_df = pd.DataFrame(trades)

    stats = {}
    if len(trades_df) > 0:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_pct'] < 0])
        win_rate = winning_trades / total_trades * 100

        total_return = (capital / INITIAL_CAPITAL - 1) * 100
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0

        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        stats = {
            'initial_capital': INITIAL_CAPITAL,
            'final_capital': capital,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
        }

        # optionally save CSV
        if save_csv:
            try:
                trades_df.to_csv('backtest_trades_simple.csv', index=False)
                log.info("[+] 交易记录已保存: backtest_trades_simple.csv")
            except PermissionError:
                log.warning('[!] 无法保存CSV (文件被占用)')

        if show_trades:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', 120)
            print(trades_df[['entry_time', 'entry_price', 'exit_time', 'exit_price', 'type', 'exit_reason', 'pnl_pct']].to_string(index=True))

    else:
        log.info("[!] 没有交易记录")

    return {
        'capital': capital,
        'trades_df': trades_df,
        'equity_curve': equity_curve,
        'stats': stats,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run simple Volume+Supertrend backtest')
    parser.add_argument('--csv', '-c', help='CSV file to load', default='LINKUSDT_8h_2020-01-01_to_2025-11-09.csv')
    parser.add_argument('--no-save', dest='save', action='store_false', help='Do not save trades CSV')
    parser.add_argument('--show-trades', dest='show_trades', action='store_true', help='Print all trades to stdout')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    res = run_backtest(df=None, csv_file=args.csv, save_csv=args.save, show_trades=args.show_trades)

    # print summary similar to original script
    stats = res.get('stats', {})
    if stats:
        print('='*60)
        print('回测结果')
        print('='*60)
        print(f"初始资金:      ${stats['initial_capital']:,.2f}")
        print(f"最终权益:      ${stats['final_capital']:,.2f}")
        print(f"总收益率:      {stats['total_return_pct']:.2f}%")
        print(f"最大回撤:      {stats['max_drawdown_pct']:.2f}%")
        print('='*60)
        print(f"总交易次数:    {stats['total_trades']}")
        print(f"盈利次数:      {stats['winning_trades']}")
        print(f"亏损次数:      {stats['losing_trades']}")
        print(f"胜率:          {stats['win_rate_pct']:.2f}%")
        if stats['avg_loss_pct'] != 0:
            print(f"盈亏比:        {abs(stats['avg_win_pct']/stats['avg_loss_pct']):.2f}")
        print('='*60)
        print(f"平均盈利:      {stats['avg_win_pct']:.2f}%")
        print(f"平均亏损:      {stats['avg_loss_pct']:.2f}%")
        print('='*60)


if __name__ == '__main__':
    main()
