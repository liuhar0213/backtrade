#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Supertrend + Williams Vix Fix组合策略
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from strategies.supertrend_strategy import SupertrendStrategy
from strategies.supertrend_vix_strategy import (
    SupertrendVixStrategy,
    SupertrendVixStrategyV2,
    SupertrendVixStrategyV3
)
from engine.backtest import SimpleBacktester
from engine.costing import CostEngine
from engine.allocator import PositionAllocator


def run_vix_test(symbol='SOL', initial_equity=10000.0):
    """
    运行Vix Fix策略测试

    Args:
        symbol: 交易品种
        initial_equity: 初始资金
    """
    print("="*80)
    print(f"SUPERTREND + VIX FIX STRATEGY TEST - {symbol}")
    print("="*80)

    # 加载数据
    data_files = {
        'SOL': 'data/BINANCE_SOLUSDT.P, 60.csv',
        'BNB': 'data/BINANCE_BNBUSDT.P, 60.csv',
        'ETH': 'data/BINANCE_ETHUSDT.P, 60.csv'
    }

    print(f"\n[LOAD] Loading {symbol} 1h data...")
    df = pd.read_csv(data_files[symbol])
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df.columns = df.columns.str.lower()
    print(f"[OK] Loaded {len(df)} bars from {df['time'].min()} to {df['time'].max()}")

    # 准备策略列表
    strategies = [
        (SupertrendStrategy(atr_period=10, factor=3.0),
         "Baseline",
         "原版Supertrend (无VIX)"),

        (SupertrendVixStrategy(atr_period=10, factor=3.0, long_only=True),
         "VIX_LongOnly",
         "只在恐慌时做多"),

        (SupertrendVixStrategyV2(atr_period=10, factor=3.0),
         "VIX_AvoidShort",
         "恐慌时避免做空"),

        (SupertrendVixStrategyV3(atr_period=10, factor=3.0),
         "VIX_Balanced",
         "平衡版"),
    ]

    results_summary = []

    for strategy, strategy_name, description in strategies:
        print(f"\n{'='*80}")
        print(f"Testing: {strategy_name} - {description}")
        print(f"{'='*80}")

        # 计算指标
        print("[CALC] Calculating indicators...")
        df_test = strategy.calculate_supertrend(df.copy())

        # 如果是Vix策略，计算Vix Fix
        if 'VIX' in strategy_name or 'Vix' in strategy_name:
            df_test = strategy.calculate_vix_fix(df_test)

        df_test = strategy.generate_signals(df_test)

        # 获取入场点
        entries = strategy.get_trade_entries(df_test)
        print(f"[SIGNAL] Generated {len(entries)} entry signals")

        # 转换为回测格式
        entries_list = []
        for entry in entries:
            entry_dict = {
                'bar_idx': entry['bar_index'],
                'side': entry['side'],
                'entry_price': entry['entry_price'],
                'stop_loss': entry['stop_loss'],
                'edge': 'Supertrend',
                'take_profit': entry['entry_price'] * 1.10 if entry['side'] == 1 else entry['entry_price'] * 0.90
            }
            entries_list.append(entry_dict)

        if len(entries_list) < 10:
            print(f"[WARN] Too few signals ({len(entries_list)}), skipping...")
            results_summary.append({
                'strategy': strategy_name,
                'description': description,
                'signals': len(entries_list),
                'trades': 0,
                'j_value': 0,
                'annual_return': 0,
                'win_rate': 0,
                'net_pnl': 0,
                'max_drawdown': 0
            })
            continue

        # 运行回测
        print("[BACKTEST] Running backtest...")
        cost_engine = CostEngine()
        allocator = PositionAllocator()
        backtester = SimpleBacktester(cost_engine, allocator, initial_equity=initial_equity)

        backtester.run(df_test, entries_list, symbol=symbol, timeframe='1h')

        # 计算指标
        trades = backtester.trades
        if len(trades) == 0:
            print(f"[WARN] No trades executed, skipping...")
            results_summary.append({
                'strategy': strategy_name,
                'description': description,
                'signals': len(entries_list),
                'trades': 0,
                'j_value': 0,
                'annual_return': 0,
                'win_rate': 0,
                'net_pnl': 0,
                'max_drawdown': 0
            })
            continue

        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['net_pnl'] > 0).sum() / len(trades_df)
        total_net_pnl = trades_df['net_pnl'].sum()
        final_equity = backtester.equity
        total_return = (final_equity - initial_equity) / initial_equity

        # 计算年化收益
        time_range = pd.to_datetime(df['time'].max()) - pd.to_datetime(df['time'].min())
        years = time_range.days / 365.25
        annual_return = (total_return / years) if years > 0 else 0

        # 计算最大回撤
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

        # 显示结果
        print(f"\n{'-'*80}")
        print(f"RESULTS: {strategy_name}")
        print(f"{'-'*80}")
        print(f"Description:     {description}")
        print(f"Signals:         {len(entries_list)}")
        print(f"Total Trades:    {len(trades_df)}")
        print(f"Win Rate:        {win_rate*100:.2f}%")
        print(f"Net PnL:         ${total_net_pnl:,.2f}")
        print(f"Final Equity:    ${final_equity:,.2f}")
        print(f"Annual Return:   {annual_return*100:.2f}%")
        print(f"Max Drawdown:    {max_drawdown*100:.2f}%")
        print(f"Calmar Ratio:    {calmar:.2f}")
        print(f"J-Value:         {j_value:.4f}")

        if j_value > 0:
            print(f"Status:          ✅ Profitable")
        else:
            print(f"Status:          ❌ Unprofitable")

        # 计算信号统计
        if 'VIX' in strategy_name or 'Vix' in strategy_name:
            panic_signals = sum(1 for e in entries if e.get('vix_panic', False))
            print(f"\nVIX Statistics:")
            print(f"  Panic Signals: {panic_signals}/{len(entries_list)} ({panic_signals/len(entries_list)*100:.1f}%)")

        results_summary.append({
            'strategy': strategy_name,
            'description': description,
            'signals': len(entries_list),
            'trades': len(trades_df),
            'j_value': j_value,
            'annual_return': annual_return,
            'win_rate': win_rate,
            'net_pnl': total_net_pnl,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'final_equity': final_equity
        })

    # 总结对比
    print(f"\n{'='*80}")
    print(f"SUMMARY - {symbol}")
    print(f"{'='*80}\n")

    print(f"{'Strategy':<20} {'Signals':>8} {'Trades':>8} {'J-Value':>10} {'Return':>12} {'WinRate':>10} {'Drawdown':>10} {'Net PnL':>15}")
    print("-"*110)

    for r in sorted(results_summary, key=lambda x: x['j_value'], reverse=True):
        print(f"{r['strategy']:<20} {r['signals']:>8} {r['trades']:>8} {r['j_value']:>10.4f} {r['annual_return']*100:>11.2f}% {r['win_rate']*100:>9.2f}% {r['max_drawdown']*100:>9.2f}% ${r['net_pnl']:>13,.2f}")

    print("\n" + "="*80)

    # 性能对比
    baseline = next((r for r in results_summary if r['strategy'] == 'Baseline'), None)
    if baseline and baseline['j_value'] > 0:
        print(f"\nVS BASELINE (Original Supertrend):")
        print("-"*80)
        for r in results_summary:
            if r['strategy'] != 'Baseline':
                j_change = ((r['j_value'] - baseline['j_value']) / baseline['j_value'] * 100) if baseline['j_value'] != 0 else 0
                pnl_change = ((r['net_pnl'] - baseline['net_pnl']) / baseline['net_pnl'] * 100) if baseline['net_pnl'] != 0 else 0
                signal_change = ((r['signals'] - baseline['signals']) / baseline['signals'] * 100) if baseline['signals'] != 0 else 0

                status = "📈 Better" if j_change > 0 else "📉 Worse" if j_change < 0 else "➡️ Same"

                print(f"\n{r['strategy']} ({r['description']}):")
                print(f"  J-Value:  {r['j_value']:.4f} ({j_change:+.1f}%) {status}")
                print(f"  Net PnL:  ${r['net_pnl']:,.2f} ({pnl_change:+.1f}%)")
                print(f"  Signals:  {r['signals']} ({signal_change:+.1f}%)")

    print("\n" + "="*80)

    return results_summary


def main():
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*15 + "SUPERTREND + WILLIAMS VIX FIX STRATEGY TEST" + " "*21 + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")

    # 测试SOL
    print("\n📊 Testing SOL (Best performing symbol for Supertrend)")
    sol_results = run_vix_test('SOL')

    # 测试BNB
    print("\n\n📊 Testing BNB")
    bnb_results = run_vix_test('BNB')

    # 最终总结
    print("\n\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*30 + "FINAL CONCLUSION" + " "*32 + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")

    print("💡 KEY FINDINGS:\n")

    # SOL最佳策略
    sol_best = max(sol_results, key=lambda x: x['j_value'])
    print(f"SOL Best Strategy: {sol_best['strategy']}")
    print(f"  J-Value: {sol_best['j_value']:.4f}")
    print(f"  Net PnL: ${sol_best['net_pnl']:,.2f}")
    print(f"  Annual Return: {sol_best['annual_return']*100:.2f}%\n")

    # BNB最佳策略
    bnb_best = max(bnb_results, key=lambda x: x['j_value'])
    print(f"BNB Best Strategy: {bnb_best['strategy']}")
    print(f"  J-Value: {bnb_best['j_value']:.4f}")
    print(f"  Net PnL: ${bnb_best['net_pnl']:,.2f}")
    print(f"  Annual Return: {bnb_best['annual_return']*100:.2f}%\n")

    print("="*80)
    print("\n📌 RECOMMENDATION:")

    # 判断是否有改进
    sol_baseline = next((r for r in sol_results if r['strategy'] == 'Baseline'), None)
    bnb_baseline = next((r for r in bnb_results if r['strategy'] == 'Baseline'), None)

    sol_improved = sol_best['j_value'] > sol_baseline['j_value'] if sol_baseline else False
    bnb_improved = bnb_best['j_value'] > bnb_baseline['j_value'] if bnb_baseline else False

    if sol_improved or bnb_improved:
        print("✅ VIX Fix filter shows improvement on some symbols!")
        if sol_improved:
            improvement = (sol_best['j_value'] - sol_baseline['j_value']) / sol_baseline['j_value'] * 100
            print(f"   - SOL: +{improvement:.1f}% improvement with {sol_best['strategy']}")
        if bnb_improved:
            improvement = (bnb_best['j_value'] - bnb_baseline['j_value']) / bnb_baseline['j_value'] * 100
            print(f"   - BNB: +{improvement:.1f}% improvement with {bnb_best['strategy']}")
    else:
        print("⚠️  VIX Fix filter did NOT improve performance.")
        print("    Original Supertrend strategy performs better.")
        print("    Stick with the baseline Supertrend v1/v2 strategy.")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
