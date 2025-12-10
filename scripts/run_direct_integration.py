#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_direct_integration.py
测试ABCDEF直接集成模式

对比：
1. 传统模式（文件I/O）：D层 → 文件 → DF → 文件 → F层
2. 直接集成模式：D层 → 内存 → DF → 内存 → F层

预期改善：
- 延迟: 20-30ms → 2-3ms (10x)
- 数据一致性: 提升
- 实时响应: 提升
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

# 测试配置
CYCLES = 5
USE_DIRECT_INTEGRATION = True


def generate_mock_data():
    """生成模拟数据"""
    features_data = pd.DataFrame({
        'trend_score': np.random.randn(100),
        'pattern_score': np.random.randn(100),
        'volatility': np.random.randn(100),
        'volume_signal': np.random.randn(100)
    })

    trades_data = pd.DataFrame({
        'signal_strength': np.random.rand(20),
        'pnl': np.random.randn(20) * 100,
        'confidence': np.random.rand(20),
        'outcome': np.random.choice(['win', 'loss'], 20)
    })

    performance_data = {
        'total_return': 0.15 + np.random.rand() * 0.1,
        'sharpe_ratio': 1.0 + np.random.rand() * 0.5,
        'max_drawdown': 0.08 + np.random.rand() * 0.05,
        'win_rate': 0.50 + np.random.rand() * 0.1,
        'trades_count': 50,
        'profit_factor': 1.5 + np.random.rand() * 0.5
    }

    return features_data, trades_data, performance_data


def test_traditional_mode():
    """测试传统文件I/O模式"""
    print("\n" + "="*80)
    print("测试传统模式（文件I/O）")
    print("="*80)

    from DF_fusion.df_coordinator import DFCoordinator

    # 创建协调器（不传入orchestrator）
    coordinator = DFCoordinator(base_dir=".")
    coordinator.print_integration_mode()

    total_time = 0

    for cycle in range(1, CYCLES + 1):
        start = time.perf_counter()

        # 模拟D层建议
        d_suggestions = [
            {
                'target': 'bias_threshold',
                'delta': np.random.uniform(-0.05, 0.05),
                'scope': 'signal',
                'confidence': 0.70 + np.random.rand() * 0.2,
                'reason': 'drift_detected'
            }
        ]

        # 模拟F层输出
        f_output = {
            'suggestions': [
                {
                    'target': 'bias_threshold',
                    'delta': np.random.uniform(-0.03, 0.03),
                    'scope': 'signal',
                    'confidence': 0.60 + np.random.rand() * 0.2,
                    'reason': 'exploration'
                }
            ],
            'insights': {
                'key_findings': [
                    {'type': 'importance', 'finding': '最重要参数: bias_threshold'}
                ],
                'exploration_status': {'success_rate': 0.55}
            }
        }

        # 运行协调（传入数据）
        result = coordinator.run_coordination(
            cycle=cycle,
            d_suggestions=d_suggestions,
            f_output=f_output
        )

        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        print(f"\n  周期 #{cycle} 延迟: {elapsed:.2f} ms")

    avg_time = total_time / CYCLES
    print(f"\n传统模式平均延迟: {avg_time:.2f} ms")

    return avg_time


def test_direct_integration_mode():
    """测试直接集成模式"""
    print("\n" + "="*80)
    print("测试直接集成模式（零文件I/O）")
    print("="*80)

    try:
        from abcdef_orchestrator import ABCDEFOrchestrator

        # 创建协调器
        orchestrator = ABCDEFOrchestrator(
            config_path="abcdef_config.yaml",
            base_dir="."
        )

        # 打印集成模式
        orchestrator.df_coordinator.print_integration_mode()

        total_time = 0
        total_d_time = 0
        total_f_time = 0
        total_df_time = 0

        for cycle in range(1, CYCLES + 1):
            print(f"\n{'='*60}")
            print(f"周期 #{cycle}")
            print(f"{'='*60}")

            # 生成模拟数据
            features_data, trades_data, performance_data = generate_mock_data()

            # 测量D层延迟
            d_start = time.perf_counter()
            d_output = orchestrator.run_d_layer(
                cycle=cycle,
                features_data=features_data,
                trades_data=trades_data,
                performance_data=performance_data,
                prev_f_infinity=None
            )
            d_elapsed = (time.perf_counter() - d_start) * 1000
            total_d_time += d_elapsed

            # 测量F层延迟（使用模拟路径）
            f_start = time.perf_counter()
            # 注意：实际运行需要真实文件，这里仅测试接口
            # f_output = orchestrator.run_f_layer(...)
            # 使用模拟F输出
            orchestrator.state['last_f_output'] = {
                'suggestions': [
                    {
                        'target': 'bias_threshold',
                        'delta': np.random.uniform(-0.03, 0.03),
                        'scope': 'signal',
                        'confidence': 0.65,
                        'reason': 'exploration'
                    }
                ],
                'insights': {
                    'key_findings': [],
                    'exploration_status': {'success_rate': 0.57}
                }
            }
            f_elapsed = (time.perf_counter() - f_start) * 1000
            total_f_time += f_elapsed

            # 测量DF融合延迟（直接集成）
            df_start = time.perf_counter()
            df_result = orchestrator.run_df_fusion(cycle=cycle, use_cached=True)
            df_elapsed = (time.perf_counter() - df_start) * 1000
            total_df_time += df_elapsed

            total_elapsed = d_elapsed + f_elapsed + df_elapsed
            total_time += total_elapsed

            print(f"\n  延迟分解:")
            print(f"    D层: {d_elapsed:.2f} ms")
            print(f"    F层: {f_elapsed:.2f} ms")
            print(f"    DF融合: {df_elapsed:.2f} ms")
            print(f"    总计: {total_elapsed:.2f} ms")

        avg_total = total_time / CYCLES
        avg_d = total_d_time / CYCLES
        avg_f = total_f_time / CYCLES
        avg_df = total_df_time / CYCLES

        print(f"\n{'='*60}")
        print("直接集成模式平均延迟:")
        print(f"  D层: {avg_d:.2f} ms")
        print(f"  F层: {avg_f:.2f} ms")
        print(f"  DF融合: {avg_df:.2f} ms")
        print(f"  总计: {avg_total:.2f} ms")
        print(f"{'='*60}")

        return avg_df  # 返回DF融合延迟用于对比

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主测试流程"""
    print("\n" + "#"*80)
    print("# ABCDEF直接集成模式性能测试")
    print("#"*80)
    print(f"\n测试周期数: {CYCLES}")
    print(f"测试项目: DF协调器延迟对比")

    # 测试1: 传统模式
    traditional_latency = test_traditional_mode()

    # 测试2: 直接集成模式
    direct_latency = test_direct_integration_mode()

    # 对比结果
    if direct_latency is not None:
        print("\n" + "#"*80)
        print("# 性能对比结果")
        print("#"*80)
        print(f"\n传统模式（文件I/O）延迟: {traditional_latency:.2f} ms")
        print(f"直接集成模式延迟: {direct_latency:.2f} ms")

        speedup = traditional_latency / direct_latency if direct_latency > 0 else 0
        improvement = ((traditional_latency - direct_latency) / traditional_latency * 100) if traditional_latency > 0 else 0

        print(f"\n性能提升:")
        print(f"  加速比: {speedup:.1f}x")
        print(f"  延迟降低: {improvement:.1f}%")
        print(f"  绝对降低: {traditional_latency - direct_latency:.2f} ms")

        if speedup >= 5:
            print(f"\n✓ 目标达成! 延迟降低 {speedup:.1f}x (目标: ≥5x)")
        else:
            print(f"\n× 未达目标，仅降低 {speedup:.1f}x (目标: ≥5x)")

    print("\n" + "#"*80)
    print("# 测试完成")
    print("#"*80)


if __name__ == '__main__':
    main()
