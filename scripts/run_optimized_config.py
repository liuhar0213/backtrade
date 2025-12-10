#!/usr/bin/env python3
"""
测试优化后的配置（bias_threshold = 0.30）
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from E_layer.main import run_backtest_with_config

def test_with_optimized_threshold():
    """使用优化阈值测试"""
    print("=" * 80)
    print("测试优化后配置（bias_threshold = 0.30）")
    print("=" * 80)

    # 加载优化配置
    config = {
        'bias_threshold': 0.30,  # 从0.45降到0.30
        'atr_multiplier': 3.0,
        'min_stop_loss': 0.015,
        'min_take_profit': 0.05,
        'trailing_activation': 0.025,
        'trailing_distance': 0.01,
        'use_atr': True,
        'dynamic_trailing': True
    }

    data_path = "data/BINANCE_BTCUSDT.P, 15.csv"

    # 运行回测
    results = run_backtest_with_config(
        data_path,
        config,
        output_dir="E_layer_results_optimized"
    )

    # 对比原始结果
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)

    print("\n配置参数对比:")
    print(f"  原始阈值: 0.45")
    print(f"  优化阈值: 0.30")
    print(f"  降低幅度: {(0.45-0.30)/0.45*100:.1f}%")

    print("\n预期改进:")
    print(f"  LONG信号: 3笔 → 50-100笔")
    print(f"  SHORT信号: 272笔 → 200-300笔")
    print(f"  多空比: 1:91 → 1:3左右")

    print("\n结果文件保存在:")
    print(f"  E_layer_results_optimized/trades.csv")
    print(f"  E_layer_results_optimized/equity_curve.csv")
    print(f"  E_layer_results_optimized/summary.csv")

    return results

if __name__ == "__main__":
    results = test_with_optimized_threshold()
