#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试单独的Supertrend策略（不加Trendlines过滤）
"""

import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from strategies.supertrend_strategy import SupertrendStrategy
from engine.backtest import SimpleBacktester
from engine.costing import CostEngine
from engine.allocator import PositionAllocator


def test_supertrend_only(symbol='ETH', atr_period=10, factor=3.0):
    """测试纯Supertrend策略"""

    print("=" * 120)
    print(f"纯Supertrend策略测试 - {symbol}")
    print(f"参数: ATR={atr_period}, Factor={factor}")
    print("=" * 120)

    # 加载数据
    data_files = {
        'SOL': 'data/BINANCE_SOLUSDT.P, 60.csv',
        'BNB': 'data/BINANCE_BNBUSDT.P, 60.csv',
        'ETH': 'data/BINANCE_ETHUSDT.P, 60.csv',
        'BTC': 'data/BINANCE_BTCUSDT.P, 60.csv'
    }

    df = pd.read_csv(data_files[symbol])
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df.columns = df.columns.str.lower()

    print(f"\n[1] 加载数据: {len(df)}根K线")

    # 计算Supertrend
    strategy = SupertrendStrategy(atr_period=atr_period, factor=factor)
    df = strategy.calculate_supertrend(df.copy())
    df = strategy.generate_signals(df)

    print(f"[2] 计算Supertrend完成")

    # 获取入场点
    entries = strategy.get_trade_entries(df)
    print(f"[3] 生成{len(entries)}个入场信号")

    # 转换为回测格式
    entries_list = []
    for entry in entries:
        entries_list.append({
            'bar_idx': entry['bar_index'],
            'side': entry['side'],
            'entry_price': entry['entry_price'],
            'stop_loss': entry['stop_loss'],
            'edge': 'Supertrend',
            'take_profit': None
        })

    # 运行回测
    cost_engine = CostEngine()
    allocator = PositionAllocator()
    backtester = SimpleBacktester(cost_engine, allocator, initial_equity=10000.0)
    backtester.run(df, entries_list, symbol=symbol, timeframe='1h')

    trades = backtester.trades
    print(f"[4] 执行{len(trades)}笔交易")

    # 统计
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['net_pnl'] > 0)
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    total_pnl = sum(t['net_pnl'] for t in trades)
    final_equity = backtester.equity

    avg_win = sum(t['net_pnl'] for t in trades if t['net_pnl'] > 0) / winning_trades if winning_trades > 0 else 0
    avg_loss = sum(t['net_pnl'] for t in trades if t['net_pnl'] <= 0) / losing_trades if losing_trades > 0 else 0

    print(f"\n{'=' * 120}")
    print("交易统计")
    print(f"{'=' * 120}")
    print(f"\n总交易数: {total_trades}")
    print(f"  盈利: {winning_trades} ({win_rate:.2f}%)")
    print(f"  亏损: {losing_trades} ({100-win_rate:.2f}%)")
    print(f"\n平均盈利: ${avg_win:.2f}")
    print(f"平均亏损: ${avg_loss:.2f}")
    print(f"盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "盈亏比: N/A")
    print(f"\n总盈亏: ${total_pnl:.2f}")
    print(f"最终权益: ${final_equity:.2f}")
    print(f"收益率: {(final_equity - 10000) / 10000 * 100:.2f}%")

    # 出场原因统计
    exit_reasons = {}
    for t in trades:
        reason = t['exit_reason']
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    print(f"\n出场原因统计:")
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/total_trades*100:.1f}%)")

    print("=" * 120)

    return {
        'total_pnl': total_pnl,
        'final_equity': final_equity,
        'win_rate': win_rate,
        'total_trades': total_trades
    }


if __name__ == '__main__':
    print("\n测试1: TradingView默认参数 (ATR=10, Factor=3.0)")
    print("-" * 120)
    test_supertrend_only('ETH', atr_period=10, factor=3.0)

    print("\n\n测试2: 之前使用的参数 (ATR=14, Factor=3.0)")
    print("-" * 120)
    test_supertrend_only('ETH', atr_period=14, factor=3.0)

    print("\n\n测试3: 更保守的参数 (ATR=10, Factor=2.0)")
    print("-" * 120)
    test_supertrend_only('ETH', atr_period=10, factor=2.0)
