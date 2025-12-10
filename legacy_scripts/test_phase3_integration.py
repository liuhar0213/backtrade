"""
Phase 3集成测试 - 完整验证

测试内容:
1. F4: 贝叶斯参数优化
2. F5: SHAP特征选择
3. F6: Kelly风险管理
4. F7: 在线学习+Walk-Forward验证
5. 对比Phase 2 baseline

Author: Phase 3 Team
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# 导入配置和核心模块
from config_manager import ABCDEFConfigManager
from E_layer.backtest.event_engine import EventEngine
from E_layer.backtest.metrics import calculate_metrics

# 导入Phase 3模块
from F_intelligence.F4_adaptive_optimizer import AdaptiveParameterOptimizer
from F_intelligence.F5_dynamic_feature_selector import DynamicFeatureSelector
from F_intelligence.F6_risk_controller import create_risk_controller
from F_intelligence.F7_online_learner import create_online_learner


class Phase3IntegrationTester:
    """Phase 3集成测试器"""

    def __init__(self, data_path: str = 'data/BINANCE_BTCUSDT.P, 15.csv'):
        """
        初始化测试器

        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.config = ABCDEFConfigManager('abcdef_config.yaml')

        # 加载数据
        print("\n[1/7] 加载数据...")
        self.data = self._load_data()
        print(f"  数据范围: {self.data.index[0]} 至 {self.data.index[-1]}")
        print(f"  数据条数: {len(self.data)}")

        # 测试结果
        self.results = {}

    def _load_data(self) -> pd.DataFrame:
        """加载市场数据"""
        df = pd.read_csv(self.data_path)

        # 转换时间戳
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Shanghai')
            df.set_index('timestamp', inplace=True)
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

        # 标准化列名（全部小写）
        df.columns = [col.lower().strip() for col in df.columns]

        # 确保必要的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基础技术指标（用于测试）"""
        print("\n[2/7] 计算基础特征...")

        # 移动平均线
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()

        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std

        # 成交量特征
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # 价格动量
        df['momentum'] = df['close'].pct_change(10)
        df['roc'] = df['close'].pct_change(20) * 100

        # 删除NaN
        df = df.dropna()

        print(f"  特征数量: {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])}")
        print(f"  有效数据: {len(df)}")

        return df

    def test_baseline(self) -> Dict[str, Any]:
        """
        测试1: Phase 2 Baseline
        不使用Phase 3的任何功能
        """
        print("\n" + "="*60)
        print("测试1: Phase 2 Baseline (不使用Phase 3)")
        print("="*60)

        # 准备数据
        df = self._add_basic_features(self.data.copy())

        # 创建简单信号（基于SMA交叉）
        df['signal'] = 0
        df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1  # 多头
        df.loc[df['sma_20'] < df['sma_50'], 'signal'] = -1  # 空头

        # 简单回测
        initial_capital = 10000
        capital = initial_capital
        position = 0
        trades = []

        for i in range(1, len(df)):
            current_signal = df.iloc[i]['signal']
            prev_signal = df.iloc[i-1]['signal']
            current_price = df.iloc[i]['close']

            # 信号变化
            if current_signal != prev_signal:
                # 平仓
                if position != 0:
                    pnl = (current_price - entry_price) * position
                    capital += pnl
                    trades.append({
                        'exit_price': current_price,
                        'pnl': pnl,
                        'return': pnl / abs(entry_price * position)
                    })
                    position = 0

                # 开仓
                if current_signal != 0:
                    position = current_signal
                    entry_price = current_price

        # 计算指标
        if trades:
            returns = [t['return'] for t in trades]
            total_return = (capital - initial_capital) / initial_capital
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            avg_return = np.mean(returns)
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

            # 计算最大回撤
            equity_curve = [initial_capital]
            temp_capital = initial_capital
            for t in trades:
                temp_capital += t['pnl']
                equity_curve.append(temp_capital)
            peak = equity_curve[0]
            max_dd = 0
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            total_return = 0
            win_rate = 0
            avg_return = 0
            sharpe = 0
            max_dd = 0

        result = {
            'name': 'Phase 2 Baseline',
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'n_trades': len(trades),
            'avg_return': avg_return
        }

        self._print_results(result)
        self.results['baseline'] = result

        return result

    def test_f5_feature_selection(self) -> Dict[str, Any]:
        """
        测试2: F5 SHAP特征选择
        """
        print("\n" + "="*60)
        print("测试2: F5 SHAP特征选择")
        print("="*60)

        try:
            # 准备数据
            df = self._add_basic_features(self.data.copy())

            # 准备特征和标签
            feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'signal']]
            X = df[feature_cols].values
            y = (df['close'].shift(-1) > df['close']).astype(int).values  # 下一根涨跌

            # 去除NaN
            valid_idx = ~np.isnan(y)
            X = X[valid_idx]
            y = y[valid_idx]

            # 分割数据
            split = int(len(X) * 0.7)
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]

            print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")
            print(f"  原始特征数: {X.shape[1]}")

            # 训练简单模型
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            baseline_score = model.score(X_test, y_test)
            print(f"  原始特征准确率: {baseline_score:.4f}")

            # 使用SHAP选择特征
            selector = DynamicFeatureSelector(verbose=1)

            # 选择top 50%特征
            n_features = max(5, X.shape[1] // 2)
            print(f"\n  使用SHAP选择Top {n_features}个特征...")

            result = selector.select_with_shap(
                X=X_train,
                y=y_train,
                model=model,
                n_features=n_features,
                plot_summary=False  # 不绘图（加快速度）
            )

            # 使用选择的特征重新训练
            X_train_selected = X_train[:, result.selected_indices]
            X_test_selected = X_test[:, result.selected_indices]

            model_selected = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model_selected.fit(X_train_selected, y_train)

            selected_score = model_selected.score(X_test_selected, y_test)
            print(f"  选择特征准确率: {selected_score:.4f}")

            improvement = (selected_score - baseline_score) / baseline_score
            print(f"  性能提升: {improvement:+.2%}")

            result_dict = {
                'name': 'F5 SHAP特征选择',
                'original_features': X.shape[1],
                'selected_features': n_features,
                'reduction': 1 - n_features / X.shape[1],
                'baseline_accuracy': baseline_score,
                'selected_accuracy': selected_score,
                'improvement': improvement
            }

            self.results['f5_feature_selection'] = result_dict

            print("\n[F5结果]")
            print(f"  特征降维: {X.shape[1]} → {n_features} ({result_dict['reduction']:.1%})")
            print(f"  性能提升: {improvement:+.2%}")

            return result_dict

        except Exception as e:
            print(f"\n[警告] F5测试失败: {e}")
            print("  这可能是因为SHAP库未安装或数据不足")
            return {'name': 'F5 SHAP特征选择', 'status': 'failed', 'error': str(e)}

    def test_f6_kelly_risk(self) -> Dict[str, Any]:
        """
        测试3: F6 Kelly风险管理
        """
        print("\n" + "="*60)
        print("测试3: F6 Kelly风险管理")
        print("="*60)

        # 模拟历史交易统计
        historical_stats = {
            'win_rate': 0.55,
            'avg_win': 0.03,
            'avg_loss': 0.015,
            'n_trades': 50
        }

        # 创建风险控制器
        risk_controller = create_risk_controller(
            initial_capital=10000,
            strategy='kelly',
            verbose=1
        )

        print("\n  测试Kelly仓位计算...")
        position_result = risk_controller.calculate_position(
            signal_strength=0.8,
            historical_stats=historical_stats,
            volatility=0.25
        )

        print(f"  建议仓位: {position_result['position_size']:.2%}")
        print(f"  风险等级: {position_result['risk_level']}")
        print(f"  置信度: {position_result['confidence']:.2%}")

        print("\n  测试止损止盈计算...")
        stops = risk_controller.calculate_stops(
            entry_price=50000,
            position_type='long',
            atr=1000
        )

        print(f"  止损: ${stops['stop_loss']:.2f}")
        print(f"  止盈: ${stops['take_profit']:.2f}")
        print(f"  移动止损激活: ${stops['trailing_stop']:.2f}")

        # 对比固定仓位策略
        print("\n  对比Kelly vs 固定策略...")
        comparison = risk_controller.compare_strategies(
            signal_strength=0.8,
            historical_stats=historical_stats,
            volatility=0.25
        )

        print(f"  Kelly仓位: {comparison['kelly']['position_size']:.2%}")
        print(f"  固定仓位: {comparison['fixed']['position_size']:.2%}")
        print(f"  差异: {comparison['comparison']['position_difference']:.2%}")

        result_dict = {
            'name': 'F6 Kelly风险管理',
            'kelly_position': position_result['position_size'],
            'fixed_position': comparison['fixed']['position_size'],
            'position_difference': comparison['comparison']['position_difference'],
            'stop_loss': stops['stop_loss'],
            'take_profit': stops['take_profit'],
            'risk_level': position_result['risk_level']
        }

        self.results['f6_kelly_risk'] = result_dict

        return result_dict

    def test_f7_online_learning(self) -> Dict[str, Any]:
        """
        测试4: F7 在线学习+Walk-Forward验证
        """
        print("\n" + "="*60)
        print("测试4: F7 在线学习+Walk-Forward验证")
        print("="*60)

        try:
            # 准备数据
            df = self._add_basic_features(self.data.copy())

            # 准备特征和标签
            feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'signal']]
            X = df[feature_cols].values
            y = (df['close'].shift(-1) > df['close']).astype(int).values

            # 去除NaN
            valid_idx = ~np.isnan(y)
            X = X[valid_idx]
            y = y[valid_idx]

            print(f"  总样本数: {len(X)}")

            # 创建在线学习器（使用简单版本，不依赖River）
            learner = create_online_learner(
                learner_type='simple',  # SGDClassifier
                verbose=1
            )

            # Walk-Forward验证
            print("\n  执行Walk-Forward验证...")
            wf_result = learner.walk_forward_validation(
                X=X,
                y=y,
                train_window=1000,
                test_window=200,
                step_size=100
            )

            print(f"\n[F7结果]")
            print(f"  验证折数: {wf_result['n_folds']}")
            print(f"  平均准确率: {wf_result['avg_accuracy']:.4f}")
            print(f"  平均F1分数: {wf_result['avg_f1_score']:.4f}")

            # 获取统计信息
            stats = learner.get_statistics()
            print(f"  模型更新次数: {stats['n_updates']}")
            print(f"  漂移检测次数: {stats['n_drifts_detected']}")

            result_dict = {
                'name': 'F7 在线学习',
                'n_folds': wf_result['n_folds'],
                'avg_accuracy': wf_result['avg_accuracy'],
                'avg_f1_score': wf_result['avg_f1_score'],
                'n_updates': stats['n_updates'],
                'n_drifts': stats['n_drifts_detected']
            }

            self.results['f7_online_learning'] = result_dict

            return result_dict

        except Exception as e:
            print(f"\n[警告] F7测试失败: {e}")
            return {'name': 'F7 在线学习', 'status': 'failed', 'error': str(e)}

    def test_full_integration(self) -> Dict[str, Any]:
        """
        测试5: 完整集成测试
        F5特征选择 + F6风控 + F7在线学习
        """
        print("\n" + "="*60)
        print("测试5: Phase 3完整集成")
        print("="*60)

        try:
            # 准备数据
            df = self._add_basic_features(self.data.copy())

            # 1. F5: 特征选择
            print("\n[步骤1] F5特征选择...")
            feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            X = df[feature_cols].values
            y = (df['close'].shift(-1) > df['close']).astype(int).values

            valid_idx = ~np.isnan(y)
            X = X[valid_idx]
            y = y[valid_idx]

            split = int(len(X) * 0.7)
            X_train, y_train = X[:split], y[:split]

            # 训练模型
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            # SHAP特征选择
            selector = DynamicFeatureSelector(verbose=0)
            n_features = max(5, X.shape[1] // 2)

            try:
                selection_result = selector.select_with_shap(
                    X=X_train[:1000],  # 使用部分数据加速
                    y=y_train[:1000],
                    model=model,
                    n_features=n_features,
                    plot_summary=False
                )
                X_selected = X[:, selection_result.selected_indices]
                print(f"  特征选择: {X.shape[1]} → {n_features}")
            except:
                print("  [警告] SHAP选择失败，使用所有特征")
                X_selected = X

            # 2. F7: 在线学习
            print("\n[步骤2] F7在线学习...")
            learner = create_online_learner(learner_type='simple', verbose=0)

            wf_result = learner.walk_forward_validation(
                X=X_selected,
                y=y,
                train_window=1000,
                test_window=200,
                step_size=100
            )

            print(f"  Walk-Forward准确率: {wf_result['avg_accuracy']:.4f}")

            # 3. F6: Kelly风控（模拟）
            print("\n[步骤3] F6 Kelly风控...")
            risk_controller = create_risk_controller(strategy='kelly', verbose=0)

            # 模拟交易统计
            historical_stats = {
                'win_rate': wf_result['avg_accuracy'],
                'avg_win': 0.03,
                'avg_loss': 0.015,
                'n_trades': wf_result['n_folds'] * 10
            }

            position = risk_controller.calculate_position(
                signal_strength=0.8,
                historical_stats=historical_stats,
                volatility=0.25
            )

            print(f"  Kelly建议仓位: {position['position_size']:.2%}")

            # 综合结果
            result_dict = {
                'name': 'Phase 3完整集成',
                'feature_reduction': 1 - n_features / X.shape[1] if X_selected.shape[1] < X.shape[1] else 0,
                'ml_accuracy': wf_result['avg_accuracy'],
                'kelly_position': position['position_size'],
                'risk_level': position['risk_level'],
                'n_model_updates': learner.get_statistics()['n_updates']
            }

            self.results['full_integration'] = result_dict

            print("\n[完整集成结果]")
            print(f"  特征降维: {result_dict['feature_reduction']:.1%}")
            print(f"  模型准确率: {result_dict['ml_accuracy']:.4f}")
            print(f"  Kelly仓位: {result_dict['kelly_position']:.2%}")
            print(f"  模型更新: {result_dict['n_model_updates']}次")

            return result_dict

        except Exception as e:
            print(f"\n[警告] 完整集成测试失败: {e}")
            import traceback
            traceback.print_exc()
            return {'name': 'Phase 3完整集成', 'status': 'failed', 'error': str(e)}

    def _print_results(self, result: Dict[str, Any]):
        """打印测试结果"""
        print(f"\n[{result['name']}结果]")
        for key, value in result.items():
            if key == 'name':
                continue
            if isinstance(value, float):
                if 'return' in key or 'rate' in key:
                    print(f"  {key}: {value:.2%}")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    def save_results(self, filename: str = 'phase3_integration_results.json'):
        """保存测试结果"""
        output_dir = Path('F_intelligence/results')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename

        # 转换numpy类型为Python原生类型
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results_json = convert_types(self.results)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)

        print(f"\n[结果已保存] {output_path}")

    def generate_comparison_report(self):
        """生成对比报告"""
        print("\n" + "="*60)
        print("Phase 3集成测试 - 对比报告")
        print("="*60)

        if 'baseline' in self.results:
            baseline = self.results['baseline']
            print(f"\n[Phase 2 Baseline]")
            print(f"  总收益: {baseline.get('total_return', 0):.2%}")
            print(f"  Sharpe: {baseline.get('sharpe_ratio', 0):.2f}")
            print(f"  最大回撤: {baseline.get('max_drawdown', 0):.2%}")
            print(f"  胜率: {baseline.get('win_rate', 0):.2%}")
            print(f"  交易次数: {baseline.get('n_trades', 0)}")

        if 'f5_feature_selection' in self.results:
            f5 = self.results['f5_feature_selection']
            if 'status' not in f5:
                print(f"\n[F5 特征选择]")
                print(f"  特征降维: {f5.get('reduction', 0):.1%}")
                print(f"  性能提升: {f5.get('improvement', 0):+.2%}")

        if 'f6_kelly_risk' in self.results:
            f6 = self.results['f6_kelly_risk']
            print(f"\n[F6 Kelly风控]")
            print(f"  Kelly仓位: {f6.get('kelly_position', 0):.2%}")
            print(f"  固定仓位: {f6.get('fixed_position', 0):.2%}")
            print(f"  风险等级: {f6.get('risk_level', 'N/A')}")

        if 'f7_online_learning' in self.results:
            f7 = self.results['f7_online_learning']
            if 'status' not in f7:
                print(f"\n[F7 在线学习]")
                print(f"  平均准确率: {f7.get('avg_accuracy', 0):.4f}")
                print(f"  模型更新: {f7.get('n_updates', 0)}次")
                print(f"  漂移检测: {f7.get('n_drifts', 0)}次")

        if 'full_integration' in self.results:
            full = self.results['full_integration']
            if 'status' not in full:
                print(f"\n[完整集成]")
                print(f"  特征降维: {full.get('feature_reduction', 0):.1%}")
                print(f"  模型准确率: {full.get('ml_accuracy', 0):.4f}")
                print(f"  Kelly仓位: {full.get('kelly_position', 0):.2%}")

        print("\n" + "="*60)
        print("测试完成！")
        print("="*60)


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("Phase 3 集成测试")
    print("="*60)
    print("\n测试内容:")
    print("  1. Phase 2 Baseline（对比基准）")
    print("  2. F5 SHAP特征选择")
    print("  3. F6 Kelly风险管理")
    print("  4. F7 在线学习")
    print("  5. Phase 3完整集成")

    # 创建测试器
    tester = Phase3IntegrationTester()

    # 运行测试
    try:
        # 测试1: Baseline
        tester.test_baseline()

        # 测试2: F5特征选择
        tester.test_f5_feature_selection()

        # 测试3: F6 Kelly风控
        tester.test_f6_kelly_risk()

        # 测试4: F7在线学习
        tester.test_f7_online_learning()

        # 测试5: 完整集成
        tester.test_full_integration()

    except Exception as e:
        print(f"\n[错误] 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

    # 生成报告
    tester.generate_comparison_report()

    # 保存结果
    tester.save_results()

    print("\n✅ 所有测试完成！")


if __name__ == '__main__':
    main()
