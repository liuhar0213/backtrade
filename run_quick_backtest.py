#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速回测验证 - 测试Edge A/B/C默认参数

目标：
1. 快速验证3个Edge是否有盈利潜力
2. 查看关键指标：胜率、Calmar、J值
3. 决定下一步：继续WFO或调整策略

测试数据：
- ETH/USDT 1分钟，前50万根K线
- 默认参数（未优化）
"""

import sys
import argparse
import sys
import logging
from backtest_simple import QuickBacktester

# ensure repo root on sys.path
sys.path.insert(0, '.')


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("run_quick_backtest")

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='ETHUSDT')
    parser.add_argument('--bars', type=int, default=500)
    args = parser.parse_args()

    logger.info("Quick backtest: %s bars=%d", args.symbol, args.bars)
    bt = QuickBacktester(symbol=args.symbol)
    df = bt.load_sample_data(args.bars)
    logger.info("Loaded sample data: %d bars", len(df))
    try:
        stats = bt.run(df)
        logger.info("Backtest finished")
        logger.info("Stats: %s", stats)
    except Exception:
        logger.exception("Backtest execution failed")


if __name__ == '__main__':
    main()
    for data_file in data_files:
        if data_file.exists():
            print(f"\n加载数据: {data_file}")
            df = pd.read_csv(data_file, nrows=max_bars)

            # 提取币种名称
            symbol = data_file.stem.replace('_1m_merged', '')

            print(f"[OK] 币种: {symbol}")
            print(f"[OK] K线数: {len(df):,}")
            print(f"[OK] 时间范围: {df['time'].iloc[0]} 到 {df['time'].iloc[-1]}")

            return df, symbol

    print("[ERROR] 找不到任何数据文件")
    return None, None


def generate_all_signals(df):
    """生成所有Edge信号"""
    print("\n" + "="*80)
    print("生成Edge信号")
    print("="*80)

    all_signals = []

    # Edge A
    print("\n[1/3] Edge A: 趋势回撤延续")
    edge_a = EdgeA({'bandwidth': 0.20, 'risk_reward': 2.0, 'time_stop_bars': 30})
    df_a = edge_a.process(df.copy())
    entries_a_df = edge_a.get_trade_entries(df_a)

    # 转换为字典列表
    if len(entries_a_df) > 0:
        entries_a = entries_a_df.to_dict('records')
        for entry in entries_a:
            entry['edge'] = 'Edge_A'
            entry['bar_idx'] = entry.pop('bar_index')
            entry['side'] = entry.pop('direction')
            entry['take_profit'] = entry.pop('target')
            entry['use_retracement_stop'] = False
        all_signals.extend(entries_a)

    print(f"  信号数: {len(entries_a_df)}")

    # Edge B
    print("\n[2/3] Edge B: NR7压缩突破")
    edge_b = EdgeB({'body_quality_threshold': 1.3, 'risk_reward': 1.5, 'retracement_stop': 0.5})
    df_b = edge_b.process(df.copy())
    entries_b_df = edge_b.get_trade_entries(df_b)

    # 转换为字典列表
    if len(entries_b_df) > 0:
        entries_b = entries_b_df.to_dict('records')
        for entry in entries_b:
            entry['edge'] = 'Edge_B'
            entry['bar_idx'] = entry.pop('bar_index')
            entry['side'] = entry.pop('direction')
            entry['take_profit'] = entry.pop('target')
            entry['use_retracement_stop'] = True
            entry['time_stop_bars'] = None  # Edge B用回吐止损
        all_signals.extend(entries_b)

    print(f"  信号数: {len(entries_b_df)}")

    # Edge C
    print("\n[3/3] Edge C: 假突破反转")
    edge_c = EdgeC({'density_percentile': 70, 'risk_reward': 1.5, 'time_stop_bars': 20})
    df_c = edge_c.process(df.copy())
    entries_c_df = edge_c.get_trade_entries(df_c)

    # 转换为字典列表
    if len(entries_c_df) > 0:
        entries_c = entries_c_df.to_dict('records')
        for entry in entries_c:
            entry['edge'] = 'Edge_C'
            entry['bar_idx'] = entry.pop('bar_index')
            entry['side'] = entry.pop('direction')
            entry['take_profit'] = entry.pop('target')
            entry['use_retracement_stop'] = False
        all_signals.extend(entries_c)

    print(f"  信号数: {len(entries_c_df)}")

    print(f"\n[OK] 总信号数: {len(all_signals)}")

    # 按bar_idx排序
    all_signals = sorted(all_signals, key=lambda x: x['bar_idx'])

    return all_signals, df_a  # 返回df_a因为它包含ATR列


def run_backtest(df, signals, symbol):
    """运行回测"""
    print("\n" + "="*80)
    print("运行回测")
    print("="*80)

    # 初始化引擎
    cost_engine = CostEngine()
    allocator = PositionAllocator()
    backtester = SimpleBacktester(cost_engine, allocator, initial_equity=10000.0)

    # 运行
    print(f"\n初始权益: $10,000")
    print(f"时间周期: 1分钟")
    print(f"币种: {symbol}")
    print(f"\n开始回测...")

    results = backtester.run(df, signals, timeframe='1m', symbol=symbol)

    return results


def analyze_results(results):
    """分析回测结果"""
    print("\n" + "="*80)
    print("回测结果汇总")
    print("="*80)

    # 基础指标
    print(f"\n[基础指标]")
    print(f"  总交易数: {results['total_trades']}")
    print(f"  胜率: {results['win_rate']:.2%}")
    print(f"  平均盈亏: ${results['avg_pnl']:.2f}")
    print(f"  总盈亏: ${results['total_pnl']:.2f}")
    print(f"  最终权益: ${results['equity_curve'][-1]['equity']:.2f}")

    # 风险指标
    print(f"\n[风险指标]")
    print(f"  最大回撤: {results['max_dd']:.2%}")
    print(f"  年化收益: {results['annual_return']:.2%}")
    print(f"  Calmar比率: {results['calmar']:.2f}")

    # 目标函数
    print(f"\n[目标函数]")
    print(f"  J值: {results['J']:.4f}")

    # 判断达标
    if results['J'] > 0.3:
        print(f"  状态: [OK] J > 0.3，达标！")
    else:
        print(f"  状态: [WARN] J < 0.3，未达标")

    # 成本分析
    if len(results['trades']) > 0:
        trades_df = pd.DataFrame(results['trades'])

        print(f"\n[成本分析]（4列分离）")
        print(f"  总入场费: ${trades_df['fee_entry'].sum():.2f}")
        print(f"  总出场费: ${trades_df['fee_exit'].sum():.2f}")
        print(f"  总资金费: ${trades_df['funding'].sum():.2f}")
        print(f"  总滑点: ${trades_df['slippage'].sum():.2f}")
        print(f"  总成本: ${trades_df['total_cost'].sum():.2f}")

        # 成本占比
        gross_sum = trades_df['gross_pnl'].sum()
        if abs(gross_sum) > 0:
            cost_ratio = trades_df['total_cost'].sum() / abs(gross_sum)
            print(f"  成本占毛利: {cost_ratio:.2%}")

        # 分Edge统计
        print(f"\n[分Edge统计]")
        for edge_name in ['Edge_A', 'Edge_B', 'Edge_C']:
            edge_trades = trades_df[trades_df['edge'] == edge_name]
            if len(edge_trades) > 0:
                edge_wins = (edge_trades['net_pnl'] > 0).sum()
                edge_wr = edge_wins / len(edge_trades)
                edge_pnl = edge_trades['net_pnl'].sum()

                print(f"  {edge_name}:")
                print(f"    交易数: {len(edge_trades)}")
                print(f"    胜率: {edge_wr:.2%}")
                print(f"    总盈亏: ${edge_pnl:.2f}")
                print(f"    平均: ${edge_trades['net_pnl'].mean():.2f}")

    # 显示部分交易
    if len(results['trades']) > 0:
        print(f"\n[交易样本]（前10笔）")
        trades_df = pd.DataFrame(results['trades'])
        display_cols = ['edge', 'side', 'entry_price', 'exit_price', 'net_pnl', 'exit_reason']
        print(trades_df[display_cols].head(10).to_string(index=False))


def generate_summary(results, symbol):
    """生成总结建议"""
    print("\n" + "="*80)
    print("结论与建议")
    print("="*80)

    J = results['J']
    wr = results['win_rate']
    calmar = results['calmar']
    total_trades = results['total_trades']

    print(f"\n测试币种: {symbol}")
    print(f"测试参数: 默认参数（未优化）")

    # 判断
    if J > 0.3 and calmar > 0.5:
        print(f"\n[结论] 系统有盈利潜力！")
        print(f"  - J值 {J:.4f} > 0.3 达标")
        print(f"  - Calmar {calmar:.2f} > 0.5 良好")
        print(f"\n[建议] 下一步：")
        print(f"  1. 运行完整WFO（搜索440万参数组合）")
        print(f"  2. 寻找平台稳区（跨币种/周期稳定参数）")
        print(f"  3. 生成完整报告")
    elif J > 0.15:
        print(f"\n[结论] 系统有潜力，但需优化")
        print(f"  - J值 {J:.4f} 接近达标")
        print(f"  - 胜率 {wr:.2%}")
        print(f"  - Calmar {calmar:.2f}")
        print(f"\n[建议] 下一步：")
        print(f"  1. 手动调整参数（带宽、RR等）")
        print(f"  2. 或运行部分WFO（测试关键参数）")
        print(f"  3. 测试其他币种（BNB/SOL）")
    else:
        print(f"\n[结论] 当前参数效果不佳")
        print(f"  - J值 {J:.4f} < 0.15")
        print(f"  - 可能原因：参数未调优、市场不适配")
        print(f"\n[建议] 下一步：")
        print(f"  1. 引入TradingView指标（RSI、MACD等）")
        print(f"  2. 或大幅调整Edge参数")
        print(f"  3. 或测试更长周期（15m/1h）")

    if total_trades < 100:
        print(f"\n[注意] 交易数较少（{total_trades}笔），样本可能不足")
        print(f"  建议增加数据量或降低Edge阈值")


def main():
    """主流程"""
    print("\n" + "="*100)
    print("快速回测验证 - Edge A/B/C默认参数")
    print("="*100)

    # 1. 加载数据
    df, symbol = load_data(max_bars=500000)

    if df is None:
        print("\n[ERROR] 数据加载失败，测试中止")
        return

    # 2. 生成信号
    signals, df_processed = generate_all_signals(df)

    if len(signals) == 0:
        print("\n[WARN] 没有生成任何信号，检查Edge参数")
        return

    # 3. 运行回测
    results = run_backtest(df_processed, signals, symbol)

    # 4. 分析结果
    analyze_results(results)

    # 5. 生成建议
    generate_summary(results, symbol)

    # 6. 保存结果（可选）
    print("\n" + "="*100)
    print("[OK] 快速验证完成！")
    print("="*100)


if __name__ == '__main__':
    main()
