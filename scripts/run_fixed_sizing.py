#!/usr/bin/env python3
"""
测试修复后的Position Sizing
对比三个版本的性能
"""
import json
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from E_layer.pipeline.loader_csv import load_csv_pandas
from E_layer.signals.bias_calculator import compute_atr
from E_layer.backtest.event_engine_fixed import EventEngineFixed
from E_layer.backtest.metrics import print_metrics_summary, calculate_metrics
from E_layer.backtest.report_csv import export_trades_csv, export_equity_curve_csv

# 导入现有系统
from core.feature_engine import compute_features
from core.strategy_pool_extended import generate_strategy_scores
from core.feature_mixer import FeatureMixer

def run_with_fixed_sizing():
    """使用修复后的position sizing运行"""
    print("=" * 80)
    print("测试修复后的Position Sizing")
    print("=" * 80)

    # 配置
    config = {
        'bias_threshold': 0.30,      # 优化后的阈值
        'atr_multiplier': 3.0,
        'min_stop_loss': 0.015,
        'min_take_profit': 0.05,
        'trailing_activation': 0.025,
        'trailing_distance': 0.01,
        'use_atr': True,
        'dynamic_trailing': True,
        'commission': 0.0004,
        'slippage': 0.0001,
        'risk_per_trade': 0.015      # ✅ 新增：单笔风险1.5%
    }

    data_path = "data/BINANCE_BTCUSDT.P, 15.csv"

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    bars = load_csv_pandas(data_path, symbol="BTCUSDT", tf="15m")
    print(f"   加载 {len(bars)} 根K线")

    # 2. 计算特征
    print("\n[2/5] 计算特征和bias...")
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

    # 3. 计算ATR
    print("\n[3/5] 计算ATR...")
    atr_values = []
    for i in range(len(bars)):
        window_start = max(0, i - 14)
        atr = compute_atr(bars[window_start:i+1], period=14)
        atr_values.append(atr)
    df['atr'] = atr_values

    # 4. 运行回测（使用修复后的引擎）
    print("\n[4/5] 运行回测（修复后Position Sizing）...")
    engine = EventEngineFixed(config, initial_capital=10000.0)

    warmup = 50
    for i in range(warmup, len(bars)):
        bar = bars[i]
        bias = df.iloc[i]['bias']
        atr = df.iloc[i]['atr']
        engine.on_bar(bar, bias, atr)

    # 5. 获取结果
    results = engine.get_results()
    trades = results['trades']
    equity_curve = results['equity_curve']
    metrics = results['metrics']

    # 打印结果
    print_metrics_summary(metrics)

    # 6. 导出结果
    print("\n[5/5] 导出结果...")
    output_dir = Path("E_layer_results_fixed")
    output_dir.mkdir(exist_ok=True)

    export_trades_csv(str(output_dir / "trades.csv"), trades)
    export_equity_curve_csv(str(output_dir / "equity_curve.csv"), equity_curve)

    print(f"\n结果已保存到: {output_dir}/")

    # 7. 对比分析
    print("\n" + "=" * 80)
    print("三版本性能对比")
    print("=" * 80)

    # 读取其他版本结果
    try:
        df_orig = pd.read_csv("E_layer_results/trades.csv", encoding='utf-8')
        df_opt = pd.read_csv("E_layer_results_optimized/trades.csv", encoding='utf-8')
        df_fixed = pd.DataFrame([{
            'symbol': t.symbol,
            'side': t.side,
            'net': t.net,
            'exit_reason': t.exit_reason
        } for t in trades])

        print("\n交易次数对比:")
        print(f"  原始版本 (threshold=0.45): {len(df_orig)}笔")
        print(f"  优化版本 (threshold=0.30): {len(df_opt)}笔")
        print(f"  修复版本 (fixed sizing):  {len(df_fixed)}笔")

        print("\n多空分布对比:")
        print(f"  原始版本: LONG {(df_orig['side']=='LONG').sum()}笔, SHORT {(df_orig['side']=='SHORT').sum()}笔")
        print(f"  优化版本: LONG {(df_opt['side']=='LONG').sum()}笔, SHORT {(df_opt['side']=='SHORT').sum()}笔")
        print(f"  修复版本: LONG {(df_fixed['side']=='LONG').sum()}笔, SHORT {(df_fixed['side']=='SHORT').sum()}笔")

        print("\n总收益对比:")
        print(f"  原始版本: {df_orig['net'].sum():.2f} USD ({df_orig['net'].sum()/100:.2f}%)")
        print(f"  优化版本: {df_opt['net'].sum():.2f} USD ({df_opt['net'].sum()/100:.2f}%)")
        print(f"  修复版本: {sum(t.net for t in trades):.2f} USD ({sum(t.net for t in trades)/100:.2f}%)")

        print("\n胜率对比:")
        print(f"  原始版本: {(df_orig['net']>0).sum()/len(df_orig)*100:.2f}%")
        print(f"  优化版本: {(df_opt['net']>0).sum()/len(df_opt)*100:.2f}%")
        print(f"  修复版本: {len([t for t in trades if t.net>0])/len(trades)*100:.2f}%")

    except Exception as e:
        print(f"[警告] 无法加载对比数据: {e}")

    # 8. 关键改进说明
    print("\n" + "=" * 80)
    print("关键修复说明")
    print("=" * 80)

    print("\n✅ Position Sizing修复:")
    print("   原代码: notional = equity (全仓)")
    print("   修复后: qty = (equity × 1.5%) / stop_distance")
    print("\n   效果:")
    print("   - 单笔风险限制在1.5%")
    print("   - 根据止损距离动态调整仓位")
    print("   - 防止单笔大亏损")

    print("\n预期改进:")
    if metrics['total_return'] > 0:
        print(f"   ✅ 总收益转正: {metrics['total_return']*100:.2f}%")
    else:
        print(f"   ⚠️  总收益仍为负: {metrics['total_return']*100:.2f}%")

    if metrics['sharpe'] > 0:
        print(f"   ✅ Sharpe转正: {metrics['sharpe']:.4f}")
    else:
        print(f"   ⚠️  Sharpe仍为负: {metrics['sharpe']:.4f}")

    return results

if __name__ == "__main__":
    try:
        results = run_with_fixed_sizing()

        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)

        metrics = results['metrics']

        # 判断修复效果
        if metrics['total_return'] > 0 and metrics['sharpe'] > 0:
            print("\n🎉 修复成功！")
            print(f"   总收益: {metrics['total_return']*100:.2f}%")
            print(f"   Sharpe: {metrics['sharpe']:.4f}")
            print("\n下一步: 继续优化参数以接近原系统性能（64.60%, Sharpe 0.28）")

        elif metrics['total_return'] > 0:
            print("\n✅ 部分成功（收益转正）")
            print(f"   总收益: {metrics['total_return']*100:.2f}%")
            print(f"   Sharpe: {metrics['sharpe']:.4f}")
            print("\n下一步: 优化止损止盈参数以提高Sharpe")

        else:
            print("\n⚠️  仍需优化")
            print(f"   总收益: {metrics['total_return']*100:.2f}%")
            print(f"   Sharpe: {metrics['sharpe']:.4f}")
            print("\n可能原因:")
            print("   1. 止损止盈参数需要调整")
            print("   2. TradeGuard限制过严")
            print("   3. 费率或滑点模型需要优化")

    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
