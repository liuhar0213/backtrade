#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supertrend策略三版本对比测试

测试市场: ETH/BNB/SOL
测试版本:
  1. 原版Supertrend
  2. Supertrend + ADX过滤
  3. Supertrend完整优化版
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path

from strategies.supertrend_strategy import (
    SupertrendStrategy,
    SupertrendADXStrategy,
    SupertrendFullStrategy
)
from engine.backtest import SimpleBacktester
from engine.costing import CostEngine
from engine.allocator import PositionAllocator


def load_data(symbol: str, timeframe: str = '60') -> pd.DataFrame:
    """加载数据"""
    print(f"\n[LOAD] Loading {symbol} {timeframe}h data...")
    data_file = f"data/BINANCE_{symbol}USDT.P, {timeframe}.csv"
    df = pd.read_csv(data_file)

    # 标准化列名
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df.columns = df.columns.str.lower()

    print(f"[OK] Loaded {len(df)} bars from {df['time'].min()} to {df['time'].max()}")
    return df


def run_strategy_test(df: pd.DataFrame, strategy, strategy_name: str, symbol: str) -> dict:
    """运行单个策略测试"""
    print(f"\n{'='*80}")
    print(f"Testing: {strategy_name}")
    print(f"{'='*80}")

    # 计算Supertrend
    df_test = strategy.calculate_supertrend(df.copy())

    # 计算ADX(如果策略需要)
    if hasattr(strategy, 'calculate_adx'):
        df_test = strategy.calculate_adx(df_test)

    # 计算成交量(如果策略需要)
    if hasattr(strategy, 'calculate_volume_filter'):
        df_test = strategy.calculate_volume_filter(df_test)

    # 生成信号
    df_test = strategy.generate_signals(df_test)

    # 获取入场点
    entries = strategy.get_trade_entries(df_test)

    print(f"[SIGNAL] Generated {len(entries)} entry signals")

    if len(entries) < 10:
        return {
            'strategy': strategy_name,
            'symbol': symbol,
            'trades': len(entries),
            'status': 'insufficient_signals'
        }

    # 转换为backtest格式
    entries_list = []
    for entry in entries:
        entry_dict = {
            'bar_idx': entry['bar_index'],
            'side': entry['side'],
            'entry_price': entry['entry_price'],
            'stop_loss': entry['stop_loss'],
            'edge': 'Supertrend'
        }
        # 添加止盈和时间止损(如果有)
        if entry.get('take_profit'):
            entry_dict['take_profit'] = entry['take_profit']
        else:
            # 原版: 使用较大的止盈(基于ATR)
            if entry['side'] == 1:
                entry_dict['take_profit'] = entry['entry_price'] * 1.10  # 10%止盈
            else:
                entry_dict['take_profit'] = entry['entry_price'] * 0.90

        if entry.get('time_stop_bars'):
            entry_dict['time_stop_bars'] = entry['time_stop_bars']

        entries_list.append(entry_dict)

    # 运行回测
    cost_engine = CostEngine()
    allocator = PositionAllocator()
    backtester = SimpleBacktester(cost_engine, allocator, initial_equity=10000.0)

    results = backtester.run(df_test, entries_list, symbol=symbol, timeframe='1h')

    # 获取交易列表
    trades = backtester.trades

    if len(trades) < 10:
        return {
            'strategy': strategy_name,
            'symbol': symbol,
            'trades': len(trades),
            'status': 'insufficient_trades'
        }

    # 计算指标
    trades_df = pd.DataFrame(trades)

    win_rate = (trades_df['net_pnl'] > 0).sum() / len(trades_df)
    total_gross_pnl = trades_df['gross_pnl'].sum()
    total_cost = trades_df['total_cost'].sum()
    total_net_pnl = trades_df['net_pnl'].sum()

    final_equity = backtester.equity
    total_return = (final_equity - 10000) / 10000

    # 计算年化收益
    time_range = pd.to_datetime(df['time'].max()) - pd.to_datetime(df['time'].min())
    years = time_range.days / 365.25
    annual_return = (total_return / years) if years > 0 else 0

    # 计算最大回撤和Calmar
    equity_curve = backtester.equity_curve
    if len(equity_curve) > 0:
        equity_series = pd.Series([e['equity'] for e in equity_curve])
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
    else:
        max_drawdown = 0
        calmar = 0

    # 计算J值
    j_value = 0.6 * annual_return + 0.25 * calmar + 0.15 * win_rate

    result = {
        'strategy': strategy_name,
        'symbol': symbol,
        'trades': len(trades_df),
        'win_rate': win_rate,
        'total_gross_pnl': total_gross_pnl,
        'total_cost': total_cost,
        'total_net_pnl': total_net_pnl,
        'final_equity': final_equity,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'j_value': j_value,
        'years': years,
        'status': 'success'
    }

    # 打印结果
    print(f"\n{'-'*80}")
    print(f"RESULTS: {strategy_name} on {symbol}")
    print(f"{'-'*80}")
    print(f"Total Trades: {result['trades']}")
    print(f"Win Rate: {result['win_rate']*100:.2f}%")
    print(f"Gross PnL: ${result['total_gross_pnl']:.2f}")
    print(f"Total Cost: ${result['total_cost']:.2f}")
    print(f"Net PnL: ${result['total_net_pnl']:.2f}")
    print(f"Final Equity: ${result['final_equity']:.2f}")
    print(f"Annual Return: {result['annual_return']*100:.2f}%")
    print(f"Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"Calmar Ratio: {result['calmar']:.2f}")
    print(f"J-Value: {result['j_value']:.4f}")

    if result['j_value'] > 0:
        print(f"\n[SUCCESS] Strategy is profitable! J={result['j_value']:.4f}")
    else:
        print(f"\n[FAIL] Strategy unprofitable. J={result['j_value']:.4f}")

    return result


def main():
    print("="*80)
    print("SUPERTREND STRATEGY - THREE VERSION COMPARISON")
    print("="*80)
    print("\nTesting 3 versions on ETH/BNB/SOL:")
    print("  v1. Original Supertrend (no filters)")
    print("  v2. Supertrend + ADX filter (ADX>25)")
    print("  v3. Supertrend Full (ADX + SL/TP + Volume)")
    print("\n" + "="*80)

    # 测试市场
    symbols = ['ETH', 'BNB', 'SOL']

    # 初始化三个策略
    strategies = [
        (SupertrendStrategy(atr_period=10, factor=3.0), "v1_Original"),
        (SupertrendADXStrategy(atr_period=10, factor=3.0, adx_period=14, adx_threshold=25.0), "v2_ADX_Filter"),
        (SupertrendFullStrategy(
            atr_period=10, factor=3.0,
            adx_period=14, adx_threshold=25.0,
            stop_loss_pct=0.02, take_profit_pct=0.05,
            volume_threshold=1.2
        ), "v3_Full_Optimized")
    ]

    all_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING SYMBOL: {symbol}USDT")
        print(f"{'='*80}")

        # 加载数据
        df = load_data(symbol, timeframe='60')

        # 测试每个策略
        for strategy, name in strategies:
            result = run_strategy_test(df, strategy, name, symbol)
            all_results.append(result)

    # 汇总结果
    print(f"\n{'='*80}")
    print("SUMMARY: ALL RESULTS")
    print(f"{'='*80}")

    results_df = pd.DataFrame(all_results)

    # 只显示成功的结果
    success_results = results_df[results_df['status'] == 'success']

    if len(success_results) > 0:
        print("\nProfitable Strategies (J > 0):")
        profitable = success_results[success_results['j_value'] > 0]
        if len(profitable) > 0:
            for _, row in profitable.iterrows():
                print(f"  [{row['strategy']}] {row['symbol']}: J={row['j_value']:.4f}, "
                      f"WinRate={row['win_rate']*100:.1f}%, Return={row['annual_return']*100:.1f}%/year")
        else:
            print("  [NONE] No profitable strategies found.")

        print("\nTop 5 Strategies by J-Value:")
        top_5 = success_results.nlargest(5, 'j_value')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            status = "[PROFIT]" if row['j_value'] > 0 else "[LOSS]"
            print(f"  {i}. {status} [{row['strategy']}] {row['symbol']}: J={row['j_value']:.4f}, "
                  f"WinRate={row['win_rate']*100:.1f}%, NetPnL=${row['total_net_pnl']:.2f}")

        # 策略版本对比
        print("\nStrategy Version Comparison:")
        for strategy_name in ['v1_Original', 'v2_ADX_Filter', 'v3_Full_Optimized']:
            strategy_results = success_results[success_results['strategy'] == strategy_name]
            if len(strategy_results) > 0:
                avg_j = strategy_results['j_value'].mean()
                avg_wr = strategy_results['win_rate'].mean()
                profitable_count = (strategy_results['j_value'] > 0).sum()
                total_count = len(strategy_results)
                print(f"  {strategy_name}: Avg J={avg_j:.4f}, Avg WinRate={avg_wr*100:.1f}%, "
                      f"Profitable: {profitable_count}/{total_count}")

    # 保存结果
    results_df.to_csv('supertrend_comparison_results.csv', index=False)
    print(f"\n[SAVE] Results saved to supertrend_comparison_results.csv")

    # 最终结论
    print(f"\n{'='*80}")
    print("FINAL CONCLUSION")
    print(f"{'='*80}")

    if len(success_results) > 0:
        best_result = success_results.loc[success_results['j_value'].idxmax()]

        if best_result['j_value'] > 0:
            print(f"\n[SUCCESS] Best strategy found!")
            print(f"  Strategy: {best_result['strategy']}")
            print(f"  Market: {best_result['symbol']}")
            print(f"  J-Value: {best_result['j_value']:.4f}")
            print(f"  Win Rate: {best_result['win_rate']*100:.2f}%")
            print(f"  Net PnL: ${best_result['total_net_pnl']:.2f}")
            print(f"  Annual Return: {best_result['annual_return']*100:.2f}%")
        else:
            print(f"\n[FAIL] All strategies unprofitable")
            print(f"  Best J-Value: {best_result['j_value']:.4f} (still negative)")
            print(f"  Best Strategy: {best_result['strategy']} on {best_result['symbol']}")
            print(f"\nConclusion:")
            print(f"  - Supertrend strategy FAILS on all tested markets")
            print(f"  - Even with ADX filter and full optimization")
            print(f"  - Recommendation: Abandon classic technical strategies")
    else:
        print("\n[FAIL] All tests produced insufficient signals/trades")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
