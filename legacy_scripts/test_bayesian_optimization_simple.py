"""
Phase 3.1 简化测试：贝叶斯优化 vs 网格搜索

使用简化的回测和特征，专注于优化算法对比

Author: Phase 3 Team
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from F_intelligence.F4_adaptive_optimizer import AdaptiveParameterOptimizer
from F_intelligence.base_optimizer import ParameterSpace


class SimpleBacktest:
    """简化回测引擎"""

    def __init__(self, commission=0.0004, slippage=0.0001):
        self.commission = commission
        self.slippage = slippage
        self.total_cost = commission + slippage

    def add_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加简单技术指标"""
        df = df.copy()

        # ATR
        df['hl'] = df['high'] - df['low']
        df['hc'] = abs(df['high'] - df['close'].shift(1))
        df['lc'] = abs(df['low'] - df['close'].shift(1))
        df['atr'] = df[['hl', 'hc', 'lc']].max(axis=1).rolling(14).mean()

        # 移动平均
        df['ma_fast'] = df['close'].rolling(20).mean()
        df['ma_slow'] = df['close'].rolling(50).mean()

        # 简单信号：快线上穿慢线=多头，下穿=空头
        df['ma_diff'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow']

        # Bias score (简化版)
        df['bias_score'] = np.tanh(df['ma_diff'] * 10)  # 归一化到(-1, 1)

        return df.dropna()

    def backtest(self, df: pd.DataFrame, params: dict) -> dict:
        """
        执行回测

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

        # 回测循环
        for i in range(len(df)):
            row = df.iloc[i]
            price = row['close']
            atr = row.get('atr', price * 0.02)

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
                'profit_factor': 0.0,
                'composite_score': 0.0
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

        # 复合得分
        if max_drawdown >= 0:
            composite_score = 0
        else:
            composite_score = (
                sharpe_ratio * 0.4 +
                total_return * 0.3 +
                (-max_drawdown) * 0.2 +
                win_rate * 0.1
            )

        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'profit_factor': profit_factor,
            'composite_score': composite_score
        }


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("Phase 3.1 简化测试: 贝叶斯优化 vs 网格搜索")
    print("="*60 + "\n")

    # 加载数据
    print("[1/3] 加载BTC 15min数据...")
    data_path = Path('data/BINANCE_BTCUSDT.P, 15.csv')
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['time'])

    # 只保留需要的列
    df = df[['timestamp', 'open', 'high', 'low', 'close']].copy()

    print(f"  数据范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"  数据行数: {len(df)}")

    # 创建回测引擎并添加特征
    print("\n[2/3] 添加技术指标...")
    backtest_engine = SimpleBacktest()
    df = backtest_engine.add_simple_features(df)
    print(f"  处理后行数: {len(df)}")

    # 创建目标函数
    def objective(params: dict):
        """目标函数"""
        result = backtest_engine.backtest(df, params)
        score = result['composite_score']
        return score, result

    # 定义参数空间
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
    print("[3/3] 执行优化对比...")
    print()

    results = optimizer.compare_methods(
        objective_function=objective,
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

    # 性能加速比分析
    if 'bayesian_gp' in results and 'grid_search' in results:
        bayesian = results['bayesian_gp']
        grid = results['grid_search']

        print("\n" + "="*60)
        print("性能加速比分析")
        print("="*60)

        time_speedup = grid.optimization_time / bayesian.optimization_time
        eval_efficiency = bayesian.n_iterations / grid.n_iterations
        score_gap = abs(bayesian.best_score - grid.best_score)
        score_ratio = bayesian.best_score / grid.best_score if grid.best_score != 0 else 0

        print(f"\n[时间性能]")
        print(f"贝叶斯耗时: {bayesian.optimization_time:.2f}秒")
        print(f"网格耗时:   {grid.optimization_time:.2f}秒")
        print(f"时间加速比: {time_speedup:.2f}x")

        print(f"\n[评估效率]")
        print(f"贝叶斯评估: {bayesian.n_iterations}次")
        print(f"网格评估:   {grid.n_iterations}次")
        print(f"样本效率:   {eval_efficiency:.2%} (贝叶斯/网格)")

        print(f"\n[优化质量]")
        print(f"贝叶斯最优得分: {bayesian.best_score:.6f}")
        print(f"网格最优得分:   {grid.best_score:.6f}")
        print(f"得分差异:       {score_gap:.6f}")
        print(f"得分比率:       {score_ratio:.4f}")

        # 综合评价
        print(f"\n[综合评价]")
        if time_speedup > 10 and score_ratio > 0.95:
            print("贝叶斯优化表现优秀！")
            print(f"- 时间加速 {time_speedup:.1f}倍")
            print(f"- 仅用 {eval_efficiency:.1%} 的评估次数")
            print(f"- 找到了 {score_ratio*100:.1f}% 的最优解质量")
        elif time_speedup > 5:
            print("贝叶斯优化表现良好。")
        else:
            print("贝叶斯优化需要进一步调优。")

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
