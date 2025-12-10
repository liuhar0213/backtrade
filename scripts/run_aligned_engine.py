#!/usr/bin/env python3
"""
测试完全对齐的引擎

验证是否能达到原系统性能：
- 总收益：+64.60%
- Sharpe：0.2831
- 交易数：66笔
"""
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from E_layer.pipeline.loader_csv import load_csv_pandas
from E_layer.signals.bias_calculator import compute_atr
from E_layer.backtest.event_engine_aligned import EventEngineAligned
from E_layer.backtest.metrics import print_metrics_summary

from core.feature_engine import compute_features
from core.strategy_pool_extended import generate_strategy_scores
from core.feature_mixer import FeatureMixer

def test_aligned_engine():
    """测试完全对齐的引擎"""
    print("=" * 80)
    print("测试完全对齐原系统的引擎")
    print("=" * 80)

    # 使用原系统的"高盈亏比型"配置
    config = {
        'bias_threshold': 0.45,
        'atr_multiplier': 3.5,
        'min_stop_loss': 0.015,
        'min_take_profit': 0.05,
        'trailing_activation': 0.03,
        'trailing_distance': 0.01,
        'use_atr': True,
        'dynamic_trailing': True,
        'commission': 0.0004,
        'slippage': 0.0001
    }

    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n关键对齐:")
    print("  [ALIGNED] 全仓交易（10000元固定）")
    print("  [ALIGNED] 动态止损使用max()（可收紧可放宽）")
    print("  [ALIGNED] 固定费用10元/笔")
    print("  [ALIGNED] 追踪止盈逻辑100%一致")

    # 加载数据
    print("\n[1/4] 加载数据...")
    data_path = "data/BINANCE_BTCUSDT.P, 15.csv"
    bars = load_csv_pandas(data_path, symbol="BTCUSDT", tf="15m")
    print(f"   加载 {len(bars)} 根K线")

    # 计算特征
    print("\n[2/4] 计算特征...")
    df = pd.DataFrame([{
        'timestamp': b.ts,
        'open': b.open,
        'high': b.high,
        'low': b.low,
        'close': b.close,
        'volume': b.volume
    } for b in bars])

    df = compute_features(df, win_short=8, win_long=34)
    df = generate_strategy_scores(df)
    df = FeatureMixer().mix(df)

    atr_values = []
    for i in range(len(bars)):
        window_start = max(0, i - 14)
        atr = compute_atr(bars[window_start:i+1], period=14)
        atr_values.append(atr)
    df['atr'] = atr_values

    # 运行回测
    print("\n[3/4] 运行回测（完全对齐版本）...")
    engine = EventEngineAligned(config, initial_capital=10000.0)

    warmup = 50
    for i in range(warmup, len(bars)):
        bar = bars[i]
        bias = df.iloc[i]['bias']
        atr = df.iloc[i]['atr']
        engine.on_bar(bar, bias, atr)

    results = engine.get_results()
    trades = results['trades']
    metrics = results['metrics']

    # 打印结果
    print_metrics_summary(metrics)

    # 详细分析
    print("\n" + "=" * 80)
    print("详细性能分析")
    print("=" * 80)

    # 交易统计
    df_trades = pd.DataFrame([{
        'side': t.side,
        'net': t.net,
        'exit_reason': t.exit_reason,
        'gross': t.gross,
        'fees': t.fees,
        'bars_held': t.tags.get('bars_held', 0)
    } for t in trades])

    print("\n持仓时长统计:")
    print(f"  平均持仓: {df_trades['bars_held'].mean():.1f}根K线")
    print(f"  最长持仓: {df_trades['bars_held'].max():.0f}根K线")
    print(f"  最短持仓: {df_trades['bars_held'].min():.0f}根K线")
    avg_hours = df_trades['bars_held'].mean() * 15 / 60
    avg_days = avg_hours / 24
    print(f"  平均持仓时间: {avg_hours:.1f}小时 ({avg_days:.2f}天)")

    print("\n多空分布:")
    side_counts = df_trades['side'].value_counts()
    for side, count in side_counts.items():
        print(f"  {side}: {count}笔 ({count/len(df_trades)*100:.1f}%)")

    print("\n出场原因:")
    exit_counts = df_trades['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        print(f"  {reason}: {count}笔 ({count/len(df_trades)*100:.1f}%)")

    # 盈亏分析
    wins = df_trades[df_trades['net'] > 0]
    losses = df_trades[df_trades['net'] <= 0]

    print("\n盈亏统计:")
    print(f"  盈利交易: {len(wins)}笔")
    print(f"  亏损交易: {len(losses)}笔")
    print(f"  胜率: {len(wins)/len(df_trades)*100:.2f}%")
    if len(wins) > 0:
        print(f"  平均盈利: ${wins['net'].mean():.2f}")
    if len(losses) > 0:
        print(f"  平均亏损: ${abs(losses['net'].mean()):.2f}")
    if len(wins) > 0 and len(losses) > 0:
        print(f"  盈亏比: {wins['net'].mean() / abs(losses['net'].mean()):.2f}")

    # 与原系统和之前版本对比
    print("\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)

    comparison_data = {
        '版本': ['原系统', 'E层v6(risk-based)', 'E层v7(对齐版)'],
        '总收益': ['+64.60%', '+7.73%', f"{metrics['total_return']*100:.2f}%"],
        'Sharpe': ['0.2831', '0.7433', f"{metrics['sharpe']:.4f}"],
        '交易数': ['66笔', '131笔', f"{metrics['total_trades']}笔"],
        '胜率': ['40.91%', '41.98%', f"{metrics['win_rate']*100:.2f}%"],
        '盈亏比': ['2.93', '1.11', f"{metrics['profit_factor']:.2f}"],
        '持仓': ['300根', '148根', f"{df_trades['bars_held'].mean():.0f}根"]
    }

    df_comparison = pd.DataFrame(comparison_data)
    print()
    print(df_comparison.to_string(index=False))

    # 达成率分析
    print("\n" + "=" * 80)
    print("原系统达成率")
    print("=" * 80)

    target_return = 64.60
    target_sharpe = 0.2831
    target_trades = 66
    target_holding = 300

    current_return = metrics['total_return'] * 100
    current_sharpe = metrics['sharpe']
    current_trades = metrics['total_trades']
    current_holding = df_trades['bars_held'].mean()

    print(f"\n总收益:")
    print(f"  目标: {target_return:.2f}%")
    print(f"  当前: {current_return:.2f}%")
    if current_return > 0:
        achievement = (current_return / target_return) * 100
        print(f"  达成率: {achievement:.1f}%")
    else:
        print(f"  状态: 仍为负值")

    print(f"\nSharpe:")
    print(f"  目标: {target_sharpe:.4f}")
    print(f"  当前: {current_sharpe:.4f}")
    if current_sharpe > 0:
        achievement = (current_sharpe / target_sharpe) * 100
        print(f"  达成率: {achievement:.1f}%")
        if achievement > 100:
            print(f"  [优秀] 超越原系统!")
    else:
        print(f"  状态: 仍为负值")

    print(f"\n交易数:")
    print(f"  目标: {target_trades}笔")
    print(f"  当前: {current_trades}笔")
    ratio = (current_trades / target_trades) * 100
    print(f"  比例: {ratio:.1f}%")

    print(f"\n持仓时长:")
    print(f"  目标: {target_holding:.0f}根")
    print(f"  当前: {current_holding:.0f}根")
    achievement = (current_holding / target_holding) * 100
    print(f"  达成率: {achievement:.1f}%")

    # 最终评价
    print("\n" + "=" * 80)
    print("对齐效果评估")
    print("=" * 80)

    if current_return >= target_return * 0.9:
        print("\n[成功] 完全对齐！收益达到原系统90%以上")
        print("  系统已准备好实盘部署")
    elif current_return >= target_return * 0.7:
        print("\n[良好] 基本对齐，收益达到原系统70%以上")
        print("  还有一些细节差异需要检查")
    elif current_return >= target_return * 0.5:
        print("\n[一般] 部分对齐，收益达到原系统50%以上")
        print("  仍有较大优化空间")
    elif current_return > 0:
        print("\n[偏低] 收益为正但远低于原系统")
        print("  需要深入分析差异原因")
    else:
        print("\n[未达标] 收益仍为负")
        print("  对齐失败，需要重新检查逻辑")

    # 输出关键差异点
    print("\n可能的差异点:")
    print("  1. 特征计算是否完全一致")
    print("  2. Bias信号计算是否完全一致")
    print("  3. ATR计算是否完全一致")
    print("  4. 时间戳对齐问题")
    print("  5. 浮点数精度问题")

    return results

if __name__ == "__main__":
    try:
        results = test_aligned_engine()

        print("\n" + "=" * 80)
        print("测试完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
