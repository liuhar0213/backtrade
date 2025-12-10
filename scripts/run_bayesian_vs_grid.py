"""
Phase 3.1 测试：贝叶斯优化 vs 网格搜索

在BTC 15min数据上对比两种优化方法的性能
目标：验证贝叶斯优化能否以更少的评估次数找到更好的参数

Author: Phase 3 Team
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from core.feature_engine import compute_features
from core.strategy_pool_extended import generate_strategy_scores
from core.feature_mixer import FeatureMixer
from F_intelligence.F4_adaptive_optimizer import AdaptiveParameterOptimizer
from F_intelligence.base_optimizer import ParameterSpace


class SimplifiedBacktest:
    """简化的回测引擎（用于快速优化测试）"""

    def __init__(self, commission=0.0004, slippage=0.0001):
        self.commission = commission
        self.slippage = slippage
        self.total_cost = commission + slippage

    def backtest(self, df: pd.DataFrame, params: dict) -> dict:
        """
        快速回测

        Args:
            df: 包含价格和信号的数据
            params: 参数字典

        Returns:
            回测结果指标
        """
        equity = 10000
        position = 0
        entry_price = 0
        highest_equity = 0
        current_stop_loss = 0
        current_take_profit = 0

        trades = []
        equity_curve = [equity]

        # 提取参数
        bias_thr = params.get('bias_threshold', 0.35)
        atr_mult = params.get('atr_multiplier', 3.0)
        min_sl = params.get('min_stop_loss', 0.015)
        min_tp = params.get('min_take_profit', 0.03)
        trailing_activation = params.get('trailing_activation', 0.02)
        trailing_distance = params.get('trailing_distance', 0.005)

        # 简化回测逻辑
        for i in range(len(df)):
            row = df.iloc[i]
            price = row['close']
            atr = row.get('atr', price * 0.02)  # 默认ATR为价格的2%

            # 信号
            bias_signal = row.get('bias_score', 0.0)

            # 无持仓时检查入场
            if position == 0:
                # 多头信号
                if bias_signal > bias_thr:
                    position = 1
                    entry_price = price * (1 + self.total_cost)
                    # 设置止损止盈
                    stop_distance = max(atr * atr_mult / price, min_sl)
                    profit_distance = max(stop_distance * 2, min_tp)
                    current_stop_loss = entry_price * (1 - stop_distance)
                    current_take_profit = entry_price * (1 + profit_distance)
                    highest_equity = equity

                # 空头信号
                elif bias_signal < -bias_thr:
                    position = -1
                    entry_price = price * (1 - self.total_cost)
                    # 设置止损止盈
                    stop_distance = max(atr * atr_mult / price, min_sl)
                    profit_distance = max(stop_distance * 2, min_tp)
                    current_stop_loss = entry_price * (1 + stop_distance)
                    current_take_profit = entry_price * (1 - profit_distance)
                    highest_equity = equity

            # 有持仓时检查出场
            elif position == 1:  # 多头
                # 更新移动止损
                if equity > highest_equity * (1 + trailing_activation):
                    highest_equity = equity
                    new_stop = price * (1 - trailing_distance)
                    current_stop_loss = max(current_stop_loss, new_stop)

                # 检查止损
                if price <= current_stop_loss:
                    pnl = (current_stop_loss - entry_price) / entry_price * equity
                    equity += pnl
                    trades.append({'pnl': pnl, 'return': pnl / equity * 100})
                    position = 0

                # 检查止盈
                elif price >= current_take_profit:
                    pnl = (current_take_profit - entry_price) / entry_price * equity
                    equity += pnl
                    trades.append({'pnl': pnl, 'return': pnl / equity * 100})
                    position = 0

            elif position == -1:  # 空头
                # 更新移动止损
                if equity > highest_equity * (1 + trailing_activation):
                    highest_equity = equity
                    new_stop = price * (1 + trailing_distance)
                    current_stop_loss = min(current_stop_loss, new_stop)

                # 检查止损
                if price >= current_stop_loss:
                    pnl = (entry_price - current_stop_loss) / entry_price * equity
                    equity += pnl
                    trades.append({'pnl': pnl, 'return': pnl / equity * 100})
                    position = 0

                # 检查止盈
                elif price <= current_take_profit:
                    pnl = (entry_price - current_take_profit) / entry_price * equity
                    equity += pnl
                    trades.append({'pnl': pnl, 'return': pnl / equity * 100})
                    position = 0

            equity_curve.append(equity)

        # 计算指标
        if not trades:
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0
            }

        returns = [t['pnl'] for t in trades]
        total_return = (equity - 10000) / 10000

        # Sharpe
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # 最大回撤
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # 胜率
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = winning_trades / len(trades) if trades else 0

        # 盈亏比
        winning_pnl = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        losing_pnl = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 0

        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'profit_factor': profit_factor
        }


def load_btc_15min_data():
    """加载BTC 15分钟数据"""
    print("[1/4] 加载BTC 15min数据...")

    # 加载数据
    data_path = Path('data/BINANCE_BTCUSDT.P, 15.csv')
    if not data_path.exists():
        print(f"[错误] 数据文件不存在: {data_path}")
        return None

    df = pd.read_csv(data_path)
    # 根据实际列名调整
    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"  数据范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"  数据行数: {len(df)}")

    return df


def prepare_features(df: pd.DataFrame):
    """准备特征"""
    print("[2/4] 计算特征...")

    # 计算基础特征
    df_featured = compute_features(df.copy())

    # 生成策略得分
    df_featured = generate_strategy_scores(df_featured)

    # 特征混合
    mixer = FeatureMixer()
    df_featured = mixer.mix_features(df_featured)

    print(f"  特征列数: {len(df_featured.columns)}")

    return df_featured


def create_objective_function(df: pd.DataFrame, backtest_engine: SimplifiedBacktest):
    """创建目标函数"""

    def objective(params: dict):
        """
        目标函数：评估参数组合

        Args:
            params: 参数字典

        Returns:
            (score, metrics): Sharpe比率和详细指标
        """
        try:
            result = backtest_engine.backtest(df, params)

            # 使用复合得分作为优化目标
            sharpe = result['sharpe_ratio']
            total_return = result['total_return']
            max_dd = result['max_drawdown']
            win_rate = result['win_rate']

            # 复合得分（类似F∞）
            if max_dd >= 0:  # 避免除零
                composite_score = 0
            else:
                composite_score = (
                    sharpe * 0.4 +
                    total_return * 0.3 +
                    (-max_dd) * 0.2 +
                    win_rate * 0.1
                )

            metrics = result.copy()
            metrics['composite_score'] = composite_score

            return composite_score, metrics

        except Exception as e:
            print(f"[警告] 回测失败: {e}")
            return 0.0, {}

    return objective


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("Phase 3.1 测试: 贝叶斯优化 vs 网格搜索")
    print("="*60 + "\n")

    # 加载数据
    df = load_btc_15min_data()
    if df is None:
        return

    # 准备特征
    df = prepare_features(df)

    # 创建回测引擎
    backtest_engine = SimplifiedBacktest()

    # 创建目标函数
    print("[3/4] 创建优化目标函数...")
    objective_function = create_objective_function(df, backtest_engine)

    # 定义参数空间（与Phase 2的优化保持一致）
    parameter_space = [
        ParameterSpace('bias_threshold', 'real', (0.25, 0.50), default=0.35),
        ParameterSpace('atr_multiplier', 'real', (2.0, 4.0), default=3.0),
        ParameterSpace('min_stop_loss', 'real', (0.010, 0.025), default=0.015),
        ParameterSpace('min_take_profit', 'real', (0.020, 0.045), default=0.030),
        ParameterSpace('trailing_activation', 'real', (0.015, 0.035), default=0.020),
        ParameterSpace('trailing_distance', 'real', (0.004, 0.012), default=0.005)
    ]

    print(f"  参数空间: {len(parameter_space)}维")
    print()

    # 创建优化器
    optimizer = AdaptiveParameterOptimizer(verbose=1)

    # 对比测试
    print("[4/4] 执行优化对比...")
    print()

    results = optimizer.compare_methods(
        objective_function=objective_function,
        parameter_space=parameter_space,
        bayesian_n_calls=50,  # 贝叶斯：50次评估
        grid_n_points=3,      # 网格：每个参数3个点 = 3^6 = 729次评估
        maximize=True,
        random_state=42
    )

    # 详细结果分析
    print("\n" + "="*60)
    print("详细结果分析")
    print("="*60)

    for method_name, result in results.items():
        print(f"\n[{method_name.upper()}]")
        print(f"最优参数:")
        for param, value in result.best_params.items():
            print(f"  {param}: {value:.4f}")

        print(f"\n详细指标:")
        for metric, value in result.best_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
