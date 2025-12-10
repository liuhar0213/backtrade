#!/usr/bin/env python3
"""
测试orchestrator初始化
"""
import sys
sys.path.insert(0, '.')

from orchestrator import ABCDEOrchestrator

try:
    # 测试基线模式
    print("测试基线模式初始化...")
    baseline_orch = ABCDEOrchestrator(mode='baseline')
    print("✓ 基线模式初始化成功")

    # 测试自适应模式
    print("\n测试自适应模式初始化...")
    adaptive_orch = ABCDEOrchestrator(mode='adaptive')
    print("✓ 自适应模式初始化成功")

    print("\n所有测试通过!")

except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
