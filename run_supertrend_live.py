#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supertrend策略 - 快速启动脚本
用于回测和未来实盘接口
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Live Supertrend runner (logging friendly)
"""

import sys
import logging
import pandas as pd
from strategies.supertrend_strategy import SupertrendStrategy, SupertrendADXStrategy
from engine.backtest import SimpleBacktester
from engine.costing import CostEngine
from engine.allocator import PositionAllocator

# ensure repo root on sys.path
sys.path.insert(0, '.')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_supertrend_live")


def run_single_strategy(symbol='SOL', strategy_type='v1', initial_equity=10000.0):
    """
    运行单个策略

    Args:
        symbol: 交易品种 ('SOL', 'BNB', 'ETH')
        strategy_type: 策略版本 ('v1'=原版, 'v2'=ADX过滤)
        initial_equity: 初始资金
    """
    logger.info("SUPERTREND %s - %s", strategy_type.upper(), symbol)

    # 加载数据
    data_files = {
        'SOL': 'data/BINANCE_SOLUSDT.P, 60.csv',
        'BNB': 'data/BINANCE_BNBUSDT.P, 60.csv',
        'ETH': 'data/BINANCE_ETHUSDT.P, 60.csv'
    }

    logger.info("Loading %s 1h data...", symbol)
    df = pd.read_csv(data_files[symbol])
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df.columns = df.columns.str.lower()
    logger.info("Loaded %d bars", len(df))

    # 初始化策略
    if strategy_type == 'v1':
        strategy = SupertrendStrategy(atr_period=10, factor=3.0)
        strategy_name = "Original Supertrend"
    elif strategy_type == 'v2':
        strategy = SupertrendADXStrategy(atr_period=10, factor=3.0, adx_threshold=25.0)
        strategy_name = "Supertrend + ADX Filter"
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    logger.info("Strategy: %s", strategy_name)

    # 计算指标
    logger.info("Calculating indicators...")
    df_test = strategy.calculate_supertrend(df.copy())
    if strategy_type == 'v2':
        df_test = strategy.calculate_adx(df_test)
    df_test = strategy.generate_signals(df_test)

    # 获取入场点
    entries = strategy.get_trade_entries(df_test)
    logger.info("Generated %d entry signals", len(entries))

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

    # 运行回测
    logger.info("Running backtest...")
    cost_engine = CostEngine()
    allocator = PositionAllocator()
    backtester = SimpleBacktester(cost_engine, allocator, initial_equity=initial_equity)

    backtester.run(df_test, entries_list, symbol=symbol, timeframe='1h')

    # 获取交易列表
    trades = backtester.trades

    if len(trades) == 0:
        logger.error("No trades executed")
        return None

    # 计算指标
    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['net_pnl'] > 0).sum() / len(trades_df)
    total_net_pnl = trades_df['net_pnl'].sum()
    final_equity = backtester.equity
    total_return = (final_equity - initial_equity) / initial_equity

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

    # 计算平均盈亏
    winning_trades = trades_df[trades_df['net_pnl'] > 0]
    losing_trades = trades_df[trades_df['net_pnl'] <= 0]
    avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0

    # 显示结果
    logger.info("BACKTEST RESULTS: Initial Equity=%s Final Equity=%s NetPnL=%s Return=%.2f%%", 
                f"${initial_equity:,.2f}", f"${final_equity:,.2f}", f"${total_net_pnl:,.2f}", total_return*100)

    logger.info("Trades: total=%d win_rate=%.2f%% avg_win=%s avg_loss=%s", 
                len(trades_df), win_rate*100, f"${avg_win:.2f}", f"${avg_loss:.2f}")

    logger.info("Risk: max_drawdown=%.2f%% calmar=%.2f", max_drawdown*100, calmar)

    logger.info("Score: J-value=%.4f", j_value)


    # 返回结果
    results = {
        'j_value': j_value,
        'annual_return': annual_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'net_pnl': total_net_pnl
    }

    return results


def run_recommended_strategies():
    """运行所有推荐策略"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "SUPERTREND STRATEGY - LIVE BACKTEST" + " "*23 + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")

    strategies = [
        ('SOL', 'v1', "🥇 SOL Supertrend v1 (冠军)"),
        ('BNB', 'v2', "🥈 BNB Supertrend v2 (稳健)"),
        ('BNB', 'v1', "🥉 BNB Supertrend v1"),
    ]

    results_summary = []

    for symbol, strategy_type, name in strategies:
        logger.info("Running recommended: %s", name)
        print(f"{'='*80}\n")

        results = run_single_strategy(symbol, strategy_type)
        results_summary.append({
            'name': name,
            'symbol': symbol,
            'type': strategy_type,
            'j_value': results['j_value'],
            'return': results['annual_return'],
            'win_rate': results['win_rate'],
            'drawdown': results['max_drawdown'],
            'net_pnl': results['net_pnl']
        })

        print("\n")

    # 总结
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*28 + "SUMMARY COMPARISON" + " "*32 + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")

    print(f"{'Strategy':<30} {'J-Value':>10} {'Return':>12} {'WinRate':>10} {'Drawdown':>12} {'Net PnL':>15}")
    print("-"*95)

    for r in sorted(results_summary, key=lambda x: x['j_value'], reverse=True):
        print(f"{r['name']:<30} {r['j_value']:>10.4f} {r['return']:>11.2f}% {r['win_rate']:>9.2f}% {r['drawdown']:>11.2f}% ${r['net_pnl']:>13,.2f}")

    print("\n" + "="*80)
    print("✅ All recommended strategies completed!")
    print("="*80)

    # 建议
    best = max(results_summary, key=lambda x: x['j_value'])
    print(f"\n🎯 BEST STRATEGY: {best['name']}")
    print(f"   J-Value: {best['j_value']:.4f}")
    print(f"   Annual Return: {best['return']:.2f}%")
    print(f"   Net PnL: ${best['net_pnl']:,.2f}")

    print("\n💡 RECOMMENDATION:")
    print("   Start with $1000-2000 for validation")
    print("   Use 0.5% risk per trade initially")
    print("   Expect 50-60% drawdown")
    print("   Expect 28-42% win rate")
    print("   Strictly follow all signals")
    print("\n   See SUPERTREND_STRATEGY_FINAL.md for full details")
    print("="*80 + "\n")


def quick_test(symbol='SOL', strategy_type='v1'):
    """快速测试单个策略"""
    return run_single_strategy(symbol, strategy_type)


if __name__ == '__main__':
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == 'quick':
            # 快速测试: python run_supertrend_live.py quick SOL v1
            symbol = sys.argv[2] if len(sys.argv) > 2 else 'SOL'
            strategy_type = sys.argv[3] if len(sys.argv) > 3 else 'v1'
            quick_test(symbol, strategy_type)
        else:
            print("Unknown command. Use:")
            print("  python run_supertrend_live.py              - Run all recommended strategies")
            print("  python run_supertrend_live.py quick SOL v1 - Quick test single strategy")
    else:
        # 运行所有推荐策略
        run_recommended_strategies()
