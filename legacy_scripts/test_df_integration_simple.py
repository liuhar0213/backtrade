#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_df_integration_simple.py
简化版DF集成测试

专注测试DF协调器的两种模式：
1. 文件I/O模式（传统）
2. 直接集成模式（优化）
"""

import time
import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))


def test_file_io_mode():
    """测试文件I/O模式"""
    print("\n" + "="*80)
    print("测试模式1: 文件I/O（传统模式）")
    print("="*80)

    from DF_fusion.df_coordinator import DFCoordinator

    # 创建协调器（不传入orchestrator）
    coordinator = DFCoordinator(base_dir=".")
    coordinator.print_integration_mode()

    # 模拟数据
    d_suggestions = [
        {
            'target': 'bias_threshold',
            'delta': 0.05,
            'scope': 'signal',
            'confidence': 0.85,
            'reason': 'drift_detected'
        },
        {
            'target': 'atr_multiplier',
            'delta': 0.1,
            'scope': 'risk',
            'confidence': 0.75,
            'reason': 'coherence_low'
        }
    ]

    f_output = {
        'suggestions': [
            {
                'target': 'bias_threshold',
                'delta': 0.03,
                'scope': 'signal',
                'confidence': 0.70,
                'reason': 'exploration'
            }
        ],
        'insights': {
            'key_findings': [
                {'type': 'importance', 'finding': '最重要参数: bias_threshold'}
            ],
            'exploration_status': {'success_rate': 0.57}
        }
    }

    # 运行多次测试
    cycles = 10
    latencies = []

    for cycle in range(1, cycles + 1):
        start = time.perf_counter()

        result = coordinator.run_coordination(
            cycle=cycle,
            d_suggestions=d_suggestions,
            f_output=f_output
        )

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        print(f"  周期 #{cycle}: {elapsed:.2f} ms")

    avg = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    print(f"\n文件I/O模式性能:")
    print(f"  平均延迟: {avg:.2f} ms")
    print(f"  P50: {p50:.2f} ms")
    print(f"  P95: {p95:.2f} ms")

    return avg


def test_direct_integration_mode():
    """测试直接集成模式"""
    print("\n" + "="*80)
    print("测试模式2: 直接集成（优化模式）")
    print("="*80)

    # 模拟一个简单的orchestrator
    class MockOrchestrator:
        def __init__(self):
            self.state = {
                'last_d_suggestions': [
                    {
                        'target': 'bias_threshold',
                        'delta': 0.05,
                        'scope': 'signal',
                        'confidence': 0.85,
                        'reason': 'drift_detected'
                    },
                    {
                        'target': 'atr_multiplier',
                        'delta': 0.1,
                        'scope': 'risk',
                        'confidence': 0.75,
                        'reason': 'coherence_low'
                    }
                ],
                'last_f_output': {
                    'suggestions': [
                        {
                            'target': 'bias_threshold',
                            'delta': 0.03,
                            'scope': 'signal',
                            'confidence': 0.70,
                            'reason': 'exploration'
                        }
                    ],
                    'insights': {
                        'key_findings': [
                            {'type': 'importance', 'finding': '最重要参数: bias_threshold'}
                        ],
                        'exploration_status': {'success_rate': 0.57}
                    }
                }
            }

    from DF_fusion.df_coordinator import DFCoordinator

    # 创建协调器（传入orchestrator）
    mock_orch = MockOrchestrator()
    coordinator = DFCoordinator(base_dir=".", orchestrator=mock_orch)
    coordinator.print_integration_mode()

    # 运行多次测试
    cycles = 10
    latencies = []

    for cycle in range(1, cycles + 1):
        start = time.perf_counter()

        # 直接调用（不传入数据，会自动从orchestrator获取）
        result = coordinator.run_coordination(cycle=cycle)

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        print(f"  周期 #{cycle}: {elapsed:.2f} ms")

    avg = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    print(f"\n直接集成模式性能:")
    print(f"  平均延迟: {avg:.2f} ms")
    print(f"  P50: {p50:.2f} ms")
    print(f"  P95: {p95:.2f} ms")

    return avg


def main():
    """主测试"""
    print("\n" + "#"*80)
    print("# DF协调器集成模式性能对比测试")
    print("#"*80)

    # 测试1: 文件I/O模式
    file_io_latency = test_file_io_mode()

    # 测试2: 直接集成模式
    direct_latency = test_direct_integration_mode()

    # 对比结果
    print("\n" + "#"*80)
    print("# 性能对比结果")
    print("#"*80)

    print(f"\n文件I/O模式: {file_io_latency:.2f} ms")
    print(f"直接集成模式: {direct_latency:.2f} ms")

    if direct_latency > 0:
        speedup = file_io_latency / direct_latency
        improvement = (file_io_latency - direct_latency) / file_io_latency * 100

        print(f"\n性能提升:")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  延迟降低: {improvement:.1f}%")
        print(f"  绝对降低: {file_io_latency - direct_latency:.2f} ms")

        # 评估
        print(f"\n评估:")
        if speedup >= 1.5:
            print(f"  ✓ 直接集成模式更快 ({speedup:.2f}x)")
        elif speedup >= 0.9:
            print(f"  = 两种模式性能相近 ({speedup:.2f}x)")
        else:
            print(f"  × 直接集成模式较慢 ({speedup:.2f}x)")

        print(f"\n结论:")
        print(f"  直接集成模式的主要优势:")
        print(f"  1. 零文件I/O延迟")
        print(f"  2. 数据一致性保证（内存直接访问）")
        print(f"  3. 实时响应能力（无需等待文件写入/读取）")
        print(f"  4. 更易于调试和监控")

    print("\n" + "#"*80)
    print("# 测试完成")
    print("#"*80)


if __name__ == '__main__':
    main()
