#!/usr/bin/env python3
"""
测试最终优化配置
调整止损止盈参数以提高盈亏比
"""
import json
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from E_layer.pipeline.loader_csv import load_csv_pandas
from E_layer.signals.bias_calculator import compute_atr
from E_layer.backtest.event_engine_fixed import EventEngineFixed
from E_layer.backtest.metrics import print_metrics_summary
from E_layer.backtest.report_csv import export_trades_csv, export_equity_curve_csv

from core.feature_engine import compute_features
from core.strategy_pool_extended import generate_strategy_scores
from core.feature_mixer import FeatureMixer

def test_final_config():
    """测试最终优化配置"""
    print("=" * 80)
    print("测试最终优化配置 - 调整止损止盈参数")
    print("=" * 80)

    # 最终优化配置
    config = {
        'bias_threshold': 0.30,
        'atr_multiplier': 3.0,
        'min_stop_loss': 0.02,       # 1.5% -> 2.0%
        'min_take_profit': 0.04,     # 5.0% -> 4.0%
        'trailing_activation': 0.03,  # 2.5% -> 3.0%
        'trailing_distance': 0.015,   # 1.0% -> 1.5%
        'use_atr': True,
        'dynamic_trailing': True,
        'commission': 0.0004,
        'slippage': 0.0001,
        'risk_per_trade': 0.005      # 0.5%
    }

    print("\n配置参数对比:")
    print("  参数                  原配置    优化后    变化")
    print("  " + "-" * 60)
    print("  min_stop_loss         1.5%      2.0%     +33%")
    print("  min_take_profit       5.0%      4.0%     -20%")
    print("  trailing_activation   2.5%      3.0%     +20%")
    print("  trailing_distance     1.0%      1.5%     +50%")
    print("  risk_per_trade        1.5%      0.5%     -67%")

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
    print("\n[3/4] 运行回测...")
    engine = EventEngineFixed(config, initial_capital=10000.0)

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

    # 导出结果
    print("\n[4/4] 导出结果...")
    output_dir = Path("E_layer_results_final")
    output_dir.mkdir(exist_ok=True)

    export_trades_csv(str(output_dir / "trades.csv"), trades)

    # 详细分析
    print("\n" + "=" * 80)
    print("详细性能分析")
    print("=" * 80)

    # 多空分布
    df_trades = pd.DataFrame([{
        'side': t.side,
        'net': t.net,
        'exit_reason': t.exit_reason,
        'gross': t.gross,
        'fees': t.fees
    } for t in trades])

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
    print(f"  平均盈利: ${wins['net'].mean():.2f}")
    print(f"  平均亏损: ${abs(losses['net'].mean()):.2f}")
    print(f"  盈亏比: {wins['net'].mean() / abs(losses['net'].mean()):.2f}")

    # 与之前版本对比
    print("\n" + "=" * 80)
    print("版本对比")
    print("=" * 80)

    comparison_data = {
        '版本': ['原系统', 'E层(0.5%原参数)', 'E层(最终优化)'],
        '总收益': ['+64.60%', '-2.87%', f"{metrics['total_return']*100:.2f}%"],
        'Sharpe': ['0.2831', '-0.4650', f"{metrics['sharpe']:.4f}"],
        '交易数': ['66笔', '144笔', f"{metrics['total_trades']}笔"],
        '胜率': ['40.91%', '37.50%', f"{metrics['win_rate']*100:.2f}%"],
        '盈亏比': ['2.93', '0.94', f"{metrics['profit_factor']:.2f}"],
        '最大回撤': ['-6.60%', '-7.27%', f"{metrics['max_drawdown']*100:.2f}%"]
    }

    df_comparison = pd.DataFrame(comparison_data)
    print()
    print(df_comparison.to_string(index=False))

    # 改善评估
    print("\n" + "=" * 80)
    print("改善评估")
    print("=" * 80)

    if metrics['total_return'] > 0:
        print("\n[成功] 总收益转正!")
        print(f"  当前收益: {metrics['total_return']*100:.2f}%")
        print(f"  目标收益: +40-55%")
        if metrics['total_return'] > 0.40:
            print(f"  [优秀] 已达到目标收益范围")
        elif metrics['total_return'] > 0.20:
            print(f"  [良好] 接近目标，继续优化")
        else:
            print(f"  [一般] 需要进一步优化")
    else:
        print("\n[未达标] 总收益仍为负")
        print(f"  当前收益: {metrics['total_return']*100:.2f}%")

    if metrics['sharpe'] > 0:
        print("\n[成功] Sharpe转正!")
        print(f"  当前Sharpe: {metrics['sharpe']:.4f}")
        print(f"  目标Sharpe: 0.20-0.26")
        if metrics['sharpe'] >= 0.20:
            print(f"  [优秀] 已达到目标范围")
        elif metrics['sharpe'] >= 0.10:
            print(f"  [良好] 接近目标")
        else:
            print(f"  [一般] 需要继续优化")
    else:
        print("\n[未达标] Sharpe仍为负")
        print(f"  当前Sharpe: {metrics['sharpe']:.4f}")

    if metrics['profit_factor'] >= 1.5:
        print("\n[成功] 盈亏比显著改善!")
        print(f"  当前盈亏比: {metrics['profit_factor']:.2f}")
        print(f"  目标盈亏比: 2.0-2.5")
        if metrics['profit_factor'] >= 2.0:
            print(f"  [优秀] 已达到目标范围")
        else:
            print(f"  [良好] 接近目标")
    else:
        print("\n[需改善] 盈亏比仍偏低")
        print(f"  当前盈亏比: {metrics['profit_factor']:.2f}")

    # 下一步建议
    print("\n" + "=" * 80)
    print("下一步建议")
    print("=" * 80)

    if metrics['total_return'] > 0 and metrics['sharpe'] > 0 and metrics['profit_factor'] > 1.5:
        print("\n[优秀] 性能已显著改善!")
        print("  建议:")
        print("  1. 微调参数以进一步接近原系统")
        print("  2. 对比原系统的逐笔交易记录")
        print("  3. 准备实盘部署")
    elif metrics['total_return'] > 0:
        print("\n[进步] 收益转正，但仍需优化")
        print("  建议:")
        print("  1. 继续调整止损止盈参数")
        print("  2. 测试不同的trailing参数组合")
        print("  3. 考虑放宽TradeGuard限制")
    else:
        print("\n[需优化] 尚未达到预期")
        print("  建议:")
        print("  1. 检查ATR计算是否与原系统一致")
        print("  2. 对比原系统的开仓/出场逻辑")
        print("  3. 尝试更激进的参数（更宽止损、更低止盈）")

    return results

if __name__ == "__main__":
    try:
        results = test_final_config()

        print("\n" + "=" * 80)
        print("测试完成!")
        print("=" * 80)

        print("\n结果文件:")
        print("  E_layer_results_final/trades.csv")
        print("  E_layer_results_final/equity_curve.csv")

    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
