#!/usr/bin/env python3
"""
使用"平衡型"配置测试对齐引擎
原系统最优配置：bias_threshold=0.35
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

def test_all_configs():
    """测试所有三种配置"""
    print("=" * 80)
    print("测试对齐引擎 - 三种配置对比")
    print("=" * 80)

    # 三种配置（来自原系统）
    configs = {
        '平衡型': {
            'bias_threshold': 0.35,
            'atr_multiplier': 3.0,
            'min_stop_loss': 0.015,
            'min_take_profit': 0.03,
            'trailing_activation': 0.02,
            'trailing_distance': 0.005,
            'use_atr': True,
            'dynamic_trailing': True,
            'commission': 0.0004,
            'slippage': 0.0001
        },
        '高胜率型': {
            'bias_threshold': 0.45,
            'atr_multiplier': 3.0,
            'min_stop_loss': 0.02,
            'min_take_profit': 0.04,
            'trailing_activation': 0.015,
            'trailing_distance': 0.005,
            'use_atr': True,
            'dynamic_trailing': True,
            'commission': 0.0004,
            'slippage': 0.0001
        },
        '高盈亏比型': {
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
    }

    # 加载数据
    print("\n[1/3] 加载数据...")
    data_path = "data/BINANCE_BTCUSDT.P, 15.csv"
    bars = load_csv_pandas(data_path, symbol="BTCUSDT", tf="15m")
    print(f"   加载 {len(bars)} 根K线")

    # 计算特征
    print("\n[2/3] 计算特征...")
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

    # 运行所有配置
    print("\n[3/3] 运行回测...")
    results = {}

    for config_name, config in configs.items():
        print(f"\n  测试 {config_name}...")
        engine = EventEngineAligned(config, initial_capital=10000.0)

        warmup = 50
        for i in range(warmup, len(bars)):
            bar = bars[i]
            bias = df.iloc[i]['bias']
            atr = df.iloc[i]['atr']
            engine.on_bar(bar, bias, atr)

        result = engine.get_results()
        results[config_name] = result
        metrics = result['metrics']

        print(f"    收益: {metrics['total_return']*100:+.2f}%, "
              f"Sharpe: {metrics['sharpe']:.4f}, "
              f"交易: {metrics['total_trades']}笔, "
              f"胜率: {metrics['win_rate']*100:.1f}%")

    # 对比报告
    print("\n" + "=" * 80)
    print("三种配置对比")
    print("=" * 80)

    comparison_data = []
    for config_name in ['平衡型', '高胜率型', '高盈亏比型']:
        metrics = results[config_name]['metrics']
        trades = results[config_name]['trades']

        bars_held = [t.tags.get('bars_held', 0) for t in trades]
        avg_holding = sum(bars_held) / len(bars_held) if bars_held else 0

        comparison_data.append({
            '配置': config_name,
            '总收益': f"{metrics['total_return']*100:.2f}%",
            'Sharpe': f"{metrics['sharpe']:.4f}",
            '交易数': metrics['total_trades'],
            '胜率': f"{metrics['win_rate']*100:.1f}%",
            '盈亏比': f"{metrics['profit_factor']:.2f}",
            '平均盈利': f"${metrics['avg_win']:.2f}",
            '平均亏损': f"${metrics['avg_loss']:.2f}",
            '最大回撤': f"{metrics['max_drawdown']*100:.2f}%",
            '持仓': f"{avg_holding:.0f}根"
        })

    df_comparison = pd.DataFrame(comparison_data)
    print()
    print(df_comparison.to_string(index=False))

    # 与原系统对比
    print("\n" + "=" * 80)
    print("与原系统性能对比")
    print("=" * 80)

    target = {
        '总收益': 64.60,
        'Sharpe': 0.2831,
        '交易数': 66,
        '胜率': 40.91,
        '盈亏比': 2.93,
        '持仓': 300
    }

    print(f"\n原系统目标:")
    print(f"  总收益: {target['总收益']:.2f}%")
    print(f"  Sharpe: {target['Sharpe']:.4f}")
    print(f"  交易数: {target['交易数']}笔")
    print(f"  胜率: {target['胜率']:.2f}%")
    print(f"  盈亏比: {target['盈亏比']:.2f}")

    # 找出最佳配置
    best_return = None
    best_return_val = -999
    best_sharpe = None
    best_sharpe_val = -999

    for config_name in ['平衡型', '高胜率型', '高盈亏比型']:
        metrics = results[config_name]['metrics']
        if metrics['total_return'] > best_return_val:
            best_return_val = metrics['total_return']
            best_return = config_name
        if metrics['sharpe'] > best_sharpe_val:
            best_sharpe_val = metrics['sharpe']
            best_sharpe = config_name

    print(f"\n对齐引擎最佳:")
    print(f"  最佳收益: {best_return} ({best_return_val*100:.2f}%)")
    print(f"  最佳Sharpe: {best_sharpe} ({best_sharpe_val:.4f})")

    # 达成率
    if best_return_val > 0:
        achievement_return = (best_return_val * 100 / target['总收益']) * 100
        print(f"\n  收益达成率: {achievement_return:.1f}%")
    else:
        print(f"\n  收益: 仍为负值")

    if best_sharpe_val > 0:
        achievement_sharpe = (best_sharpe_val / target['Sharpe']) * 100
        print(f"  Sharpe达成率: {achievement_sharpe:.1f}%")
    else:
        print(f"  Sharpe: 仍为负值")

    # 详细分析最佳配置
    print("\n" + "=" * 80)
    print(f"最佳配置详细分析：{best_return}")
    print("=" * 80)

    best_metrics = results[best_return]['metrics']
    best_trades = results[best_return]['trades']

    df_trades = pd.DataFrame([{
        'side': t.side,
        'net': t.net,
        'exit_reason': t.exit_reason,
        'bars_held': t.tags.get('bars_held', 0)
    } for t in best_trades])

    print("\n多空分布:")
    side_counts = df_trades['side'].value_counts()
    for side, count in side_counts.items():
        print(f"  {side}: {count}笔 ({count/len(df_trades)*100:.1f}%)")

    print("\n出场原因:")
    exit_counts = df_trades['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        print(f"  {reason}: {count}笔 ({count/len(df_trades)*100:.1f}%)")

    print("\n持仓时长:")
    print(f"  平均: {df_trades['bars_held'].mean():.1f}根K线")
    print(f"  中位: {df_trades['bars_held'].median():.0f}根K线")
    print(f"  最长: {df_trades['bars_held'].max():.0f}根K线")

    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)

    if best_return_val >= target['总收益'] * 0.9:
        print("\n[成功] 完全对齐！已达到原系统90%以上性能")
    elif best_return_val >= target['总收益'] * 0.7:
        print("\n[良好] 基本对齐，已达到原系统70%以上性能")
    elif best_return_val >= target['总收益'] * 0.5:
        print("\n[一般] 部分对齐，已达到原系统50%以上性能")
    elif best_return_val > 0:
        print("\n[偏低] 收益为正但远低于原系统")
        print("  可能原因:")
        print("    1. 特征计算差异")
        print("    2. Bias信号质量差异")
        print("    3. 参数仍需优化")
    else:
        print("\n[失败] 收益为负，对齐失败")
        print("  需要检查:")
        print("    1. 基础逻辑是否正确")
        print("    2. 数据加载是否一致")
        print("    3. 特征计算是否完全对齐")

    return results

if __name__ == "__main__":
    try:
        results = test_all_configs()

        print("\n" + "=" * 80)
        print("测试完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
