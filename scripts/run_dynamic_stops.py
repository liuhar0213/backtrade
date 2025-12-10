#!/usr/bin/env python3
"""
测试动态止损版本
验证是否能接近原系统性能
"""
import json
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from E_layer.pipeline.loader_csv import load_csv_pandas
from E_layer.signals.bias_calculator import compute_atr
from E_layer.backtest.event_engine_dynamic import EventEngineDynamic
from E_layer.backtest.metrics import print_metrics_summary
from E_layer.backtest.report_csv import export_trades_csv, export_equity_curve_csv

from core.feature_engine import compute_features
from core.strategy_pool_extended import generate_strategy_scores
from core.feature_mixer import FeatureMixer

def test_dynamic_stops():
    """测试动态止损版本"""
    print("=" * 80)
    print("测试动态止损版本 - 关键改进")
    print("=" * 80)

    # 使用最佳已知配置
    config = {
        'bias_threshold': 0.30,
        'atr_multiplier': 3.0,
        'min_stop_loss': 0.015,       # 1.5%
        'min_take_profit': 0.05,      # 5.0%
        'trailing_activation': 0.025,  # 2.5%
        'trailing_distance': 0.01,     # 1.0%
        'use_atr': True,
        'dynamic_trailing': True,
        'commission': 0.0004,
        'slippage': 0.0001,
        'risk_per_trade': 0.005       # 0.5% (最佳参数)
    }

    print("\n关键改进:")
    print("  [NEW] 动态止损调整：每根K线根据当前ATR更新止损")
    print("  [NEW] 止损只能放宽，不能收紧")
    print("  [NEW] 追踪持仓时长，用于分析")
    print("\n预期效果:")
    print("  持仓时长: ~50根 -> ~200-300根")
    print("  平均盈利: $79 -> $300-400")
    print("  盈亏比: 0.94 -> 2.0-2.5")
    print("  总收益: -2.87% -> +30-50%")

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
    print("\n[3/4] 运行回测（动态止损版本）...")
    engine = EventEngineDynamic(config, initial_capital=10000.0)

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
    output_dir = Path("E_layer_results_dynamic")
    output_dir.mkdir(exist_ok=True)

    export_trades_csv(str(output_dir / "trades.csv"), trades)

    # 详细分析
    print("\n" + "=" * 80)
    print("详细性能分析")
    print("=" * 80)

    # 持仓时长分析
    df_trades = pd.DataFrame([{
        'side': t.side,
        'net': t.net,
        'exit_reason': t.exit_reason,
        'gross': t.gross,
        'fees': t.fees,
        'bars_held': t.tags.get('bars_held', 0),
        'entry_ts': t.entry_ts,
        'exit_ts': t.exit_ts
    } for t in trades])

    print("\n持仓时长统计:")
    print(f"  平均持仓: {df_trades['bars_held'].mean():.1f}根K线")
    print(f"  最长持仓: {df_trades['bars_held'].max():.0f}根K线")
    print(f"  最短持仓: {df_trades['bars_held'].min():.0f}根K线")
    print(f"  中位持仓: {df_trades['bars_held'].median():.0f}根K线")

    # 转换为小时/天
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

    # 与之前版本对比
    print("\n" + "=" * 80)
    print("版本对比")
    print("=" * 80)

    comparison_data = {
        '版本': ['原系统', 'E层(0.5%固定)', 'E层(动态止损)'],
        '总收益': ['+64.60%', '-2.87%', f"{metrics['total_return']*100:.2f}%"],
        'Sharpe': ['0.2831', '-0.4650', f"{metrics['sharpe']:.4f}"],
        '交易数': ['66笔', '144笔', f"{metrics['total_trades']}笔"],
        '胜率': ['40.91%', '37.50%', f"{metrics['win_rate']*100:.2f}%"],
        '盈亏比': ['2.93', '0.94', f"{metrics['profit_factor']:.2f}"],
        '最大回撤': ['-6.60%', '-7.27%', f"{metrics['max_drawdown']*100:.2f}%"],
        '平均持仓': ['300根(~3.1天)', '~50根(估计)', f"{df_trades['bars_held'].mean():.0f}根({avg_days:.2f}天)"]
    }

    df_comparison = pd.DataFrame(comparison_data)
    print()
    print(df_comparison.to_string(index=False))

    # 改善评估
    print("\n" + "=" * 80)
    print("改善评估")
    print("=" * 80)

    prev_return = -2.87
    prev_sharpe = -0.4650
    prev_pf = 0.94
    prev_holding = 50

    improvement_return = ((metrics['total_return']*100 - prev_return) / abs(prev_return)) * 100
    improvement_sharpe = ((metrics['sharpe'] - prev_sharpe) / abs(prev_sharpe)) * 100
    improvement_pf = ((metrics['profit_factor'] - prev_pf) / prev_pf) * 100
    improvement_holding = ((df_trades['bars_held'].mean() - prev_holding) / prev_holding) * 100

    print("\n相比固定止损版本的改进:")
    print(f"  总收益改进: {improvement_return:+.1f}%")
    print(f"  Sharpe改进: {improvement_sharpe:+.1f}%")
    print(f"  盈亏比改进: {improvement_pf:+.1f}%")
    print(f"  持仓时长改进: {improvement_holding:+.1f}%")

    # 与原系统对比
    print("\n相比原系统的达成率:")
    target_return = 64.60
    target_sharpe = 0.2831
    target_pf = 2.93
    target_holding = 300

    if metrics['total_return'] > 0:
        achievement_return = (metrics['total_return']*100 / target_return) * 100
        print(f"  总收益达成: {achievement_return:.1f}%")
    else:
        print(f"  总收益: 仍为负值 ({metrics['total_return']*100:.2f}%)")

    if metrics['sharpe'] > 0:
        achievement_sharpe = (metrics['sharpe'] / target_sharpe) * 100
        print(f"  Sharpe达成: {achievement_sharpe:.1f}%")
    else:
        print(f"  Sharpe: 仍为负值 ({metrics['sharpe']:.4f})")

    achievement_pf = (metrics['profit_factor'] / target_pf) * 100
    print(f"  盈亏比达成: {achievement_pf:.1f}%")

    achievement_holding = (df_trades['bars_held'].mean() / target_holding) * 100
    print(f"  持仓时长达成: {achievement_holding:.1f}%")

    # 最终评价
    print("\n" + "=" * 80)
    print("最终评价")
    print("=" * 80)

    if metrics['total_return'] > 0.30 and metrics['sharpe'] > 0.20:
        print("\n[成功] 动态止损显著改善了系统性能!")
        print("  系统已达到实盘部署标准")
    elif metrics['total_return'] > 0 and metrics['sharpe'] > 0:
        print("\n[进步] 收益和Sharpe均转正!")
        print("  还需要微调参数以进一步接近原系统")
    elif metrics['total_return'] > prev_return * 2:  # 改善超过2倍
        print("\n[改善] 动态止损带来了明显改善")
        print("  需要继续优化其他参数")
    else:
        print("\n[有限改善] 动态止损有帮助但不够")
        print("  可能还需要检查其他逻辑差异")

    # 下一步建议
    print("\n下一步建议:")
    if metrics['total_return'] > 0.40:
        print("  1. 准备实盘部署")
        print("  2. 添加风控监控")
        print("  3. 建立交易日志系统")
    elif metrics['total_return'] > 0:
        print("  1. 微调ATR multiplier (尝试2.5, 3.5)")
        print("  2. 测试不同的trailing参数")
        print("  3. 对比原系统逐笔交易记录")
    else:
        print("  1. 对比原系统的完整代码逻辑")
        print("  2. 检查是否还有其他隐藏的动态调整")
        print("  3. 考虑添加更多诊断日志")

    return results

if __name__ == "__main__":
    try:
        results = test_dynamic_stops()

        print("\n" + "=" * 80)
        print("测试完成!")
        print("=" * 80)

        print("\n结果文件:")
        print("  E_layer_results_dynamic/trades.csv")

    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
