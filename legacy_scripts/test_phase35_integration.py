"""
Phase 3.5 集成测试脚本

测试三个核心模块：
1. ParameterGenome - 参数版本管理
2. FeatureSemanticMonitor - 特征语义监控
3. SystemHealthDashboard - 系统健康仪表盘

Author: Phase 3.5 Team
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import Phase 3.5 modules
from F_intelligence.param_genome import ParameterGenome, create_parameter_genome
from F_intelligence.feature_semantic_monitor import FeatureSemanticMonitor, create_semantic_monitor
from F_intelligence.system_health_dashboard import SystemHealthDashboard

# Import existing Phase 3 modules (for integration)
try:
    from F_intelligence.bayesian_optimizer import BayesianOptimizer
    from F_intelligence.shap_feature_selector import SHAPFeatureSelector
    PHASE3_AVAILABLE = True
except ImportError:
    PHASE3_AVAILABLE = False
    print("[警告] Phase 3 模块未找到，将使用模拟数据")


def test_parameter_genome():
    """测试1: 参数基因组管理"""
    print("\n" + "="*80)
    print("测试 1: 参数基因组管理 (Parameter Genome)")
    print("="*80)

    # 创建参数基因组
    genome = create_parameter_genome(storage_path="F_intelligence/genome_test")

    # 模拟参数优化演化过程
    print("\n[场景] 模拟5代参数演化过程...")

    # 第0代: 初始参数（手动设置）
    gen0_params = {
        'ma_fast': 10,
        'ma_slow': 30,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }

    v0 = genome.create_version(
        params=gen0_params,
        source='manual',
        metadata={'generation': 0, 'description': '初始手动参数'}
    )

    # 模拟回测性能
    genome.update_metrics(v0, {
        'F_infinity': 0.45,
        'sharpe': 0.8,
        'total_return': 0.15,
        'max_drawdown': -0.12
    })

    # 第1代: Grid Search优化
    gen1_params = gen0_params.copy()
    gen1_params['ma_fast'] = 12
    gen1_params['rsi_period'] = 12

    v1 = genome.create_version(
        params=gen1_params,
        parent_version=v0,
        source='grid',
        metadata={'generation': 1, 'method': 'GridSearch'}
    )

    genome.update_metrics(v1, {
        'F_infinity': 0.52,
        'sharpe': 1.1,
        'total_return': 0.22,
        'max_drawdown': -0.10
    })

    # 第2代: Bayesian优化
    gen2_params = gen1_params.copy()
    gen2_params['ma_slow'] = 35
    gen2_params['rsi_overbought'] = 75

    v2 = genome.create_version(
        params=gen2_params,
        parent_version=v1,
        source='bayesian',
        metadata={'generation': 2, 'method': 'BayesianOptimization'}
    )

    genome.update_metrics(v2, {
        'F_infinity': 0.68,
        'sharpe': 1.5,
        'total_return': 0.35,
        'max_drawdown': -0.08
    })

    # 第3代: SHAP派生特征
    gen3_params = gen2_params.copy()
    gen3_params['volatility_window'] = 20  # 新特征
    gen3_params['volume_ma'] = 15  # 新特征

    v3 = genome.create_version(
        params=gen3_params,
        parent_version=v2,
        source='shap',
        metadata={'generation': 3, 'method': 'SHAP-derived'}
    )

    genome.update_metrics(v3, {
        'F_infinity': 0.75,
        'sharpe': 1.8,
        'total_return': 0.48,
        'max_drawdown': -0.06
    })

    # 第4代: 在线学习调整
    gen4_params = gen3_params.copy()
    gen4_params['ma_fast'] = 11
    gen4_params['rsi_oversold'] = 28

    v4 = genome.create_version(
        params=gen4_params,
        parent_version=v3,
        source='online_learning',
        metadata={'generation': 4, 'method': 'OnlineLearning'}
    )

    genome.update_metrics(v4, {
        'F_infinity': 0.82,
        'sharpe': 2.1,
        'total_return': 0.62,
        'max_drawdown': -0.05
    })

    # 激活第4代参数
    genome.activate_version(v4)

    # 测试参数依赖关系
    print("\n[测试] 添加参数依赖关系...")
    genome.add_dependency('ma_fast', 'ma_slow', 'derived_from', strength=0.8)
    genome.add_dependency('rsi_period', 'rsi_oversold', 'correlates_with', strength=0.6)
    genome.add_dependency('ma_fast', 'volatility_window', 'enhances', strength=0.5)

    # 导出依赖图
    graph_path = Path("F_intelligence/genome_test/dependency_graph.json")
    genome.export_dependency_graph(graph_path, format='json')

    # 生成血统报告
    print("\n[报告] 生成参数血统报告...")
    lineage_report = genome.generate_lineage_report(v4)
    print(lineage_report)

    # 获取最佳版本
    best_versions = genome.get_best_version(metric='F_infinity', top_n=3)
    print("\n[最佳版本] Top 3 by F_infinity:")
    for i, (vid, score) in enumerate(best_versions, 1):
        print(f"  {i}. {vid}: F_infinity = {score:.4f}")

    # 版本差异对比
    print("\n[版本对比] v0 vs v4:")
    diff = genome.get_version_diff(v0, v4)
    for param, changes in diff.items():
        print(f"  {param}: {changes['version_a']} → {changes['version_b']}", end="")
        if changes.get('pct_change'):
            print(f" ({changes['pct_change']:+.1f}%)")
        else:
            print()

    print("\n[SUCCESS] 参数基因组测试完成")
    return genome


def test_semantic_monitor():
    """测试2: 特征语义监控"""
    print("\n" + "="*80)
    print("测试 2: 特征语义监控 (Feature Semantic Monitor)")
    print("="*80)

    # 创建语义监控器
    monitor = create_semantic_monitor(
        window_size=30,
        storage_path="F_intelligence/semantic_monitor_test"
    )

    # 模拟特征演化过程（30天）
    print("\n[场景] 模拟30天特征演化...")

    features = ['MA_cross', 'RSI_signal', 'Volume_surge', 'MACD_divergence', 'BB_squeeze']

    # 生成模拟的特征嵌入和分布数据
    base_time = datetime.now() - timedelta(days=30)

    for day in range(30):
        timestamp = base_time + timedelta(days=day)

        for feature in features:
            # 模拟特征嵌入（64维向量）
            # 前15天稳定，后15天引入漂移
            if day < 15:
                # 稳定期：嵌入向量变化小
                base_embedding = np.random.randn(64)
                embedding = base_embedding + np.random.randn(64) * 0.05
            else:
                # 漂移期：某些特征开始漂移
                if feature in ['RSI_signal', 'Volume_surge']:
                    # 这两个特征出现语义漂移
                    drift_factor = (day - 15) / 15  # 0→1
                    base_embedding = np.random.randn(64)
                    embedding = base_embedding + np.random.randn(64) * (0.05 + drift_factor * 0.3)
                else:
                    # 其他特征保持稳定
                    base_embedding = np.random.randn(64)
                    embedding = base_embedding + np.random.randn(64) * 0.05

            monitor.update_embedding(feature, embedding, timestamp)

            # 模拟特征分布
            if day < 15:
                values = np.random.normal(0, 1, 1000)
            else:
                if feature == 'RSI_signal':
                    # RSI_signal分布漂移（均值偏移）
                    shift = (day - 15) / 15 * 0.5
                    values = np.random.normal(shift, 1, 1000)
                elif feature == 'Volume_surge':
                    # Volume_surge分布漂移（方差变化）
                    scale = 1 + (day - 15) / 15 * 0.8
                    values = np.random.normal(0, scale, 1000)
                else:
                    values = np.random.normal(0, 1, 1000)

            monitor.update_distribution(feature, values, timestamp)

            # 模拟SHAP重要性
            base_importance = {
                'MA_cross': 0.25,
                'RSI_signal': 0.20,
                'Volume_surge': 0.18,
                'MACD_divergence': 0.22,
                'BB_squeeze': 0.15
            }

            if day < 15:
                importance = base_importance[feature] + np.random.randn() * 0.02
            else:
                if feature == 'RSI_signal':
                    # RSI_signal重要性下降（不稳定）
                    importance = base_importance[feature] * (1 - (day - 15) / 15 * 0.3) + np.random.randn() * 0.05
                else:
                    importance = base_importance[feature] + np.random.randn() * 0.02

            monitor.update_importance(feature, max(0, importance), timestamp)

    # 计算每个特征的稳定性指标
    print("\n[稳定性指标]")
    for feature in features:
        f_semantic = monitor.compute_semantic_stability(feature)
        drift_metrics = monitor.compute_distribution_drift(feature, method='ks')
        importance_stability = monitor.compute_importance_stability(feature)

        print(f"\n{feature}:")
        print(f"  F_semantic (语义稳定性): {f_semantic:.4f}")
        print(f"  Distribution drift (分布漂移): {drift_metrics['drift']:.4f}")
        print(f"  Importance stability (重要性稳定性): {importance_stability:.4f}")

    # 检测语义漂移
    print("\n[漂移检测]")
    drifted_features = monitor.detect_semantic_drift(
        semantic_threshold=0.8,
        drift_threshold=0.3,
        importance_threshold=0.6
    )

    if drifted_features:
        print(f"检测到 {len(drifted_features)} 个特征出现漂移:")
        for drift_info in drifted_features:
            print(f"\n特征: {drift_info['feature']}")
            print(f"  问题数量: {len(drift_info['issues'])}")
            for issue in drift_info['issues']:
                print(f"  - {issue['type']}: {issue['value']:.4f} (阈值: {issue['threshold']}, 严重性: {issue['severity']})")
    else:
        print("未检测到特征漂移")

    # 生成修复建议
    if drifted_features:
        print("\n[修复建议]")
        suggestions = monitor.suggest_repairs(drifted_features)
        for suggestion in suggestions:
            print(f"\n特征: {suggestion['feature']}")
            for action in suggestion['actions']:
                print(f"  [{action['priority']}] {action['action']}: {action['reason']}")

    # 生成健康报告
    print("\n[健康报告]")
    health_report = monitor.get_health_report()
    print(f"监控特征数量: {health_report['monitored_features']}")
    print(f"平均语义稳定性: {health_report['average_semantic_stability']:.4f}")
    print(f"平均重要性稳定性: {health_report['average_importance_stability']:.4f}")
    print(f"漂移特征数量: {health_report['num_drifted']}")
    print(f"健康状态: {health_report['health_status']}")

    # 保存健康报告
    monitor.save_health_report(health_report)

    print("\n[SUCCESS] 特征语义监控测试完成")
    return monitor, health_report


def test_system_health_dashboard():
    """测试3: 系统健康仪表盘"""
    print("\n" + "="*80)
    print("测试 3: 系统健康仪表盘 (System Health Dashboard)")
    print("="*80)

    # 创建仪表盘
    dashboard = SystemHealthDashboard(storage_path="F_intelligence/health_dashboard_test")

    # 模拟7天的系统运行数据
    print("\n[场景] 模拟7天系统运行...")

    base_time = datetime.now() - timedelta(days=7)

    for day in range(7):
        timestamp = base_time + timedelta(days=day)

        # 模拟系统指标演化
        # Day 0-2: 系统稳定
        # Day 3-4: 系统性能下降
        # Day 5-6: 系统恢复

        if day <= 2:
            # 稳定期
            f_infinity = 0.75 + np.random.rand() * 0.1
            f_semantic = 0.90 + np.random.rand() * 0.05
            f_optimization = 0.85 + np.random.rand() * 0.08
            f_prediction = 0.80 + np.random.rand() * 0.1
            f_risk = 0.95 + np.random.rand() * 0.03
        elif day <= 4:
            # 下降期
            decline = (day - 2) * 0.15
            f_infinity = 0.75 - decline + np.random.rand() * 0.05
            f_semantic = 0.90 - decline * 0.8 + np.random.rand() * 0.05
            f_optimization = 0.85 - decline * 0.5 + np.random.rand() * 0.05
            f_prediction = 0.80 - decline * 0.6 + np.random.rand() * 0.05
            f_risk = 0.95 - decline * 0.3 + np.random.rand() * 0.03
        else:
            # 恢复期
            recovery = (day - 4) * 0.15
            f_infinity = 0.45 + recovery + np.random.rand() * 0.05
            f_semantic = 0.60 + recovery * 0.8 + np.random.rand() * 0.05
            f_optimization = 0.60 + recovery * 0.5 + np.random.rand() * 0.05
            f_prediction = 0.62 + recovery * 0.6 + np.random.rand() * 0.05
            f_risk = 0.80 + recovery * 0.3 + np.random.rand() * 0.03

        # 收集指标
        metrics = dashboard.collect_metrics(
            f_infinity=f_infinity,
            f_semantic=f_semantic,
            optimization_efficiency=f_optimization,
            prediction_accuracy=f_prediction,
            risk_health=f_risk,
            # 额外指标
            active_strategies=np.random.randint(5, 15),
            avg_sharpe=np.random.uniform(0.8, 2.5),
            total_return=np.random.uniform(0.1, 0.5)
        )

        print(f"\nDay {day}: F_eco = {metrics['F_eco']:.4f}")

    # 获取当前系统状态
    print("\n[当前状态]")
    status = dashboard.get_current_status()
    print(f"健康状态: {status['health_status']}")
    print(f"F_eco: {status['latest_f_eco']:.4f}")
    print(f"趋势: {status['trend']}")

    # 检测告警
    print("\n[告警检测]")
    alerts = dashboard.detect_alerts(
        f_eco_threshold=0.6,
        f_semantic_threshold=0.7,
        decline_threshold=-0.15
    )

    if alerts:
        print(f"检测到 {len(alerts)} 个告警:")
        for alert in alerts:
            severity_prefix = {
                'CRITICAL': '[!]',
                'WARNING': '[*]',
                'INFO': '[i]'
            }
            prefix = severity_prefix.get(alert['severity'], '[ ]')
            print(f"  {prefix} [{alert['severity']}] {alert['message']}")
    else:
        print("[OK] 无告警")

    # 生成健康报告
    print("\n[生成HTML仪表盘]")
    dashboard_path = dashboard.generate_dashboard(
        title="Phase 3.5 系统健康仪表盘",
        description="集成测试 - 7天运行数据"
    )
    print(f"仪表盘已生成: {dashboard_path}")

    # 生成文本摘要
    print("\n[系统摘要]")
    summary = dashboard.get_summary_text()
    print(summary)

    # 保存快照
    snapshot_path = dashboard.save_snapshot()
    print(f"\n快照已保存: {snapshot_path}")

    print("\n[SUCCESS] 系统健康仪表盘测试完成")
    return dashboard, status


def integration_test_with_phase3():
    """集成测试: Phase 3.5 + Phase 3"""
    print("\n" + "="*80)
    print("集成测试: Phase 3.5 与 Phase 3 协同工作")
    print("="*80)

    if not PHASE3_AVAILABLE:
        print("[跳过] Phase 3 模块未找到，跳过集成测试")
        return

    print("\n[场景] 完整工作流模拟...")

    # 1. 使用BayesianOptimizer优化参数
    print("\n步骤1: Bayesian优化")
    optimizer = BayesianOptimizer(
        param_space={
            'ma_fast': (5, 20),
            'ma_slow': (20, 50),
            'rsi_period': (10, 20)
        },
        n_iterations=5
    )

    # 创建参数基因组记录优化过程
    genome = create_parameter_genome(storage_path="F_intelligence/genome_integration")

    # 模拟优化过程
    for i in range(5):
        # Bayesian优化建议参数
        suggested_params = {
            'ma_fast': np.random.randint(5, 20),
            'ma_slow': np.random.randint(20, 50),
            'rsi_period': np.random.randint(10, 20)
        }

        # 创建参数版本
        version_id = genome.create_version(
            params=suggested_params,
            source='bayesian',
            metadata={'iteration': i}
        )

        # 模拟回测结果
        f_infinity = 0.5 + i * 0.05 + np.random.rand() * 0.1
        sharpe = 0.8 + i * 0.2 + np.random.rand() * 0.2

        genome.update_metrics(version_id, {
            'F_infinity': f_infinity,
            'sharpe': sharpe
        })

        print(f"  迭代 {i}: F_infinity = {f_infinity:.4f}, Sharpe = {sharpe:.4f}")

    # 2. 使用SHAP选择特征并监控语义
    print("\n步骤2: SHAP特征选择 + 语义监控")

    # 创建语义监控器
    monitor = create_semantic_monitor(storage_path="F_intelligence/semantic_monitor_integration")

    # 模拟特征选择过程
    features = ['MA_10', 'MA_30', 'RSI_14', 'MACD', 'BB_width', 'Volume_MA']

    for feature in features:
        # 模拟特征嵌入
        embedding = np.random.randn(64)
        monitor.update_embedding(feature, embedding)

        # 模拟特征重要性
        importance = np.random.rand()
        monitor.update_importance(feature, importance)

        print(f"  {feature}: 重要性 = {importance:.4f}")

    # 3. 系统健康监控
    print("\n步骤3: 系统健康监控")

    dashboard = SystemHealthDashboard(storage_path="F_intelligence/health_dashboard_integration")

    # 收集系统指标
    health_report = monitor.get_health_report()
    best_version = genome.get_best_version(metric='F_infinity', top_n=1)[0]

    metrics = dashboard.collect_metrics(
        f_infinity=best_version[1],  # 最佳F_infinity
        f_semantic=health_report['average_semantic_stability'],
        optimization_efficiency=0.85,
        prediction_accuracy=0.78,
        risk_health=0.92
    )

    print(f"  F_eco = {metrics['F_eco']:.4f}")

    # 生成综合报告
    print("\n步骤4: 生成综合报告")

    report = {
        'timestamp': datetime.now().isoformat(),
        'best_parameters': {
            'version_id': best_version[0],
            'F_infinity': best_version[1]
        },
        'feature_health': {
            'monitored_features': health_report['monitored_features'],
            'semantic_stability': health_report['average_semantic_stability'],
            'status': health_report['health_status']
        },
        'system_health': {
            'F_eco': metrics['F_eco'],
            'status': dashboard.get_current_status()['health_status']
        }
    }

    report_path = Path("F_intelligence/PHASE35_INTEGRATION_REPORT.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n综合报告已保存: {report_path}")
    print("\n[SUCCESS] Phase 3 + Phase 3.5 集成测试完成")


def main():
    """主测试函数"""
    print("="*80)
    print("Phase 3.5 集成测试套件")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # 测试1: 参数基因组
    try:
        genome = test_parameter_genome()
        results['parameter_genome'] = 'PASSED'
    except Exception as e:
        print(f"\n[ERROR] 参数基因组测试失败: {e}")
        results['parameter_genome'] = f'FAILED: {e}'

    # 测试2: 语义监控
    try:
        monitor, health_report = test_semantic_monitor()
        results['semantic_monitor'] = 'PASSED'
    except Exception as e:
        print(f"\n[ERROR] 语义监控测试失败: {e}")
        results['semantic_monitor'] = f'FAILED: {e}'

    # 测试3: 健康仪表盘
    try:
        dashboard, status = test_system_health_dashboard()
        results['health_dashboard'] = 'PASSED'
    except Exception as e:
        print(f"\n[ERROR] 健康仪表盘测试失败: {e}")
        results['health_dashboard'] = f'FAILED: {e}'

    # 集成测试
    try:
        integration_test_with_phase3()
        results['phase3_integration'] = 'PASSED' if PHASE3_AVAILABLE else 'SKIPPED'
    except Exception as e:
        print(f"\n[ERROR] Phase 3集成测试失败: {e}")
        results['phase3_integration'] = f'FAILED: {e}'

    # 生成测试报告
    print("\n" + "="*80)
    print("测试结果摘要")
    print("="*80)

    for test_name, result in results.items():
        status_mark = '[PASS]' if result == 'PASSED' else ('[SKIP]' if result == 'SKIPPED' else '[FAIL]')
        print(f"{status_mark} {test_name}: {result}")

    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 保存测试报告
    report_path = Path("F_intelligence/PHASE35_TEST_REPORT.json")
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)

    print(f"\n测试报告已保存: {report_path}")

    return results


if __name__ == "__main__":
    main()
