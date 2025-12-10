#!/usr/bin/env python3
"""
基于案例分析洞察创建改进配置
结合三种配置的优点

改进要点:
1. 保持平衡型的紧止损 (1.5%)
2. 增加追踪止盈激活频率 (降低activation阈值)
3. 优化bias_threshold减少短期交易
4. 保持快速止盈 (3.0%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict

from core.feature_engine import compute_features
from core.strategy_pool_extended import generate_strategy_scores
from core.feature_mixer import FeatureMixer


class ImprovedBacktest:
    """改进配置回测引擎"""

    def __init__(self, commission=0.0004, slippage=0.0001):
        self.commission = commission
        self.slippage = slippage
        self.total_cost = commission + slippage

    def backtest(self, df: pd.DataFrame, params: Dict, config_name: str) -> Dict:
        """详细回测"""
        equity = 10000
        position = 0
        entry_price = 0
        entry_idx = 0
        entry_time = None
        highest_equity = 0
        current_stop_loss = 0
        current_take_profit = 0

        trades = []
        equity_curve = [equity]

        # 参数
        bias_thr = params.get('bias_threshold', 0.4)
        use_atr = params.get('use_atr', True)
        atr_mult = params.get('atr_multiplier', 2.5)
        min_sl = params.get('min_stop_loss', 0.02)
        min_tp = params.get('min_take_profit', 0.04)
        use_dynamic_trailing = params.get('dynamic_trailing', True)
        trailing_activation = params.get('trailing_activation', 0.02)
        trailing_distance = params.get('trailing_distance', 0.01)

        has_atr = 'atr' in df.columns

        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            current_time = row.get('datetime', row.get('timestamp', i))

            if position != 0:
                # 计算当前盈亏
                if position == 1:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                current_equity = equity + (current_price - entry_price) * position * (10000 / entry_price)

                # 更新最高权益
                if current_equity > highest_equity:
                    highest_equity = current_equity

                # 动态调整止损（ATR自适应）
                if use_atr and has_atr:
                    atr = row['atr']
                    dynamic_sl = atr / entry_price * atr_mult
                    current_stop_loss = max(dynamic_sl, min_sl)
                    current_take_profit = max(dynamic_sl * 2, min_tp)
                else:
                    current_stop_loss = min_sl
                    current_take_profit = min_tp

                # 检查止损
                if pnl_pct <= -current_stop_loss:
                    exit_price = current_price * (1 - self.slippage * position)
                    pnl = (exit_price - entry_price) * position * (10000 / entry_price)
                    pnl_after_cost = pnl - 10000 * self.total_cost * 2
                    equity += pnl_after_cost

                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'side': 'long' if position == 1 else 'short',
                        'pnl': pnl_after_cost,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'stop_loss',
                        'bars_held': i - entry_idx,
                        'config': config_name
                    })

                    position = 0
                    highest_equity = 0
                    equity_curve.append(equity)
                    continue

                # 检查止盈
                if pnl_pct >= current_take_profit:
                    exit_price = current_price * (1 - self.slippage * position)
                    pnl = (exit_price - entry_price) * position * (10000 / entry_price)
                    pnl_after_cost = pnl - 10000 * self.total_cost * 2
                    equity += pnl_after_cost

                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'side': 'long' if position == 1 else 'short',
                        'pnl': pnl_after_cost,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'take_profit',
                        'bars_held': i - entry_idx,
                        'config': config_name
                    })

                    position = 0
                    highest_equity = 0
                    equity_curve.append(equity)
                    continue

                # 动态追踪止盈
                if use_dynamic_trailing and pnl_pct > trailing_activation:
                    max_profit_pct = (highest_equity - equity) / equity if equity > 0 else 0

                    if pnl_pct < max_profit_pct - trailing_distance:
                        exit_price = current_price * (1 - self.slippage * position)
                        pnl = (exit_price - entry_price) * position * (10000 / entry_price)
                        pnl_after_cost = pnl - 10000 * self.total_cost * 2
                        equity += pnl_after_cost

                        trades.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'side': 'long' if position == 1 else 'short',
                            'pnl': pnl_after_cost,
                            'pnl_pct': pnl_pct,
                            'exit_reason': 'trailing_stop',
                            'bars_held': i - entry_idx,
                            'config': config_name
                        })

                        position = 0
                        highest_equity = 0
                        equity_curve.append(equity)
                        continue

            # 开仓逻辑
            if position == 0:
                bias = row.get('bias', 0)

                if abs(bias) > bias_thr:
                    direction = 1 if bias > 0 else -1
                    entry_price = current_price * (1 + self.slippage * direction)
                    entry_time = current_time
                    position = direction
                    entry_idx = i
                    highest_equity = equity

                    # 初始止损止盈
                    if use_atr and has_atr:
                        atr = row['atr']
                        current_stop_loss = max(atr / entry_price * atr_mult, min_sl)
                        current_take_profit = max(atr / entry_price * atr_mult * 2, min_tp)
                    else:
                        current_stop_loss = min_sl
                        current_take_profit = min_tp

            equity_curve.append(equity)

        # 统计
        if len(trades) == 0:
            return {
                'config_name': config_name,
                'sharpe': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'profit_factor': 0,
                'final_equity': equity,
                'params': params
            }

        trades_df = pd.DataFrame(trades)

        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]

        win_rate = len(winning_trades) / len(trades_df)
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

        total_return = (equity - 10000) / 10000

        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()

        # 出场原因统计
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

        return {
            'config_name': config_name,
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades_df),
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': equity,
            'params': params,
            'exit_reasons': exit_reasons
        }


def load_btc_15min_data():
    """加载BTC 15分钟数据"""
    data_dir = Path('data')
    files = list(data_dir.glob('*BTC*15*.csv'))

    if not files:
        print("错误: 找不到BTC 15分钟数据文件")
        return None

    file_path = files[0]
    print(f"加载数据: {file_path.name}")

    df = pd.read_csv(file_path)
    df.columns = [col.lower().strip() for col in df.columns]

    if 'volume' not in df.columns:
        df['volume'] = 0

    return df


def main():
    print("\n" + "="*80)
    print("改进配置测试 - 基于案例分析洞察")
    print("="*80)

    # 加载数据
    print("\n加载BTC 15分钟数据...")
    raw_df = load_btc_15min_data()
    if raw_df is None:
        return

    # 计算特征
    print("计算特征...")
    df = compute_features(raw_df)
    scores = generate_strategy_scores(df)
    df = pd.concat([df, scores], axis=1)

    # 使用最优权重
    best_weights = {'score_turtle': 0.4, 'score_ytc': 0.35, 'score_volume': 0.25}
    mixer = FeatureMixer(weights=best_weights)
    df = mixer.mix(df)

    print(f"数据准备完成: {len(df)} 行\n")

    # 定义配置
    configs = {
        '平衡型(原始)': {
            'bias_threshold': 0.35,
            'atr_multiplier': 3.0,
            'min_stop_loss': 0.015,
            'min_take_profit': 0.03,
            'trailing_activation': 0.02,
            'trailing_distance': 0.005,
            'use_atr': True,
            'dynamic_trailing': True
        },
        '改进型A': {
            'bias_threshold': 0.35,
            'atr_multiplier': 3.0,
            'min_stop_loss': 0.015,      # 保持紧止损
            'min_take_profit': 0.03,      # 保持快速止盈
            'trailing_activation': 0.015,  # 降低激活阈值 (从2.0%到1.5%)
            'trailing_distance': 0.005,   # 保持紧追踪
            'use_atr': True,
            'dynamic_trailing': True
        },
        '改进型B': {
            'bias_threshold': 0.40,        # 提高入场阈值减少短期交易
            'atr_multiplier': 3.0,
            'min_stop_loss': 0.015,
            'min_take_profit': 0.03,
            'trailing_activation': 0.015,  # 更积极的追踪
            'trailing_distance': 0.008,   # 稍微放宽追踪距离
            'use_atr': True,
            'dynamic_trailing': True
        },
        '改进型C': {
            'bias_threshold': 0.35,
            'atr_multiplier': 3.0,
            'min_stop_loss': 0.015,
            'min_take_profit': 0.035,      # 稍微提高止盈
            'trailing_activation': 0.015,
            'trailing_distance': 0.005,
            'use_atr': True,
            'dynamic_trailing': True
        }
    }

    # 运行回测
    print("开始测试改进配置...\n")
    engine = ImprovedBacktest()
    results = {}

    for config_name, params in configs.items():
        print(f"回测 {config_name}...")
        start = time.time()
        result = engine.backtest(df, params, config_name)
        elapsed = time.time() - start
        results[config_name] = result
        print(f"  完成 ({elapsed:.2f}秒)\n")

    # 生成对比报告
    print("\n" + "="*80)
    print("改进配置对比报告")
    print("="*80)

    # 核心指标对比表
    print("\n核心指标对比:")
    print(f"{'指标':<20} {'平衡型(原始)':<20} {'改进型A':<20} {'改进型B':<20} {'改进型C':<20}")
    print("-" * 100)

    metrics = [
        ('Sharpe比率', 'sharpe', '.4f'),
        ('总收益率', 'total_return', '.2%'),
        ('最大回撤', 'max_drawdown', '.2%'),
        ('胜率', 'win_rate', '.2%'),
        ('交易笔数', 'total_trades', 'd'),
        ('盈亏比', 'profit_factor', '.2f'),
        ('平均盈利', 'avg_win', '.2f'),
        ('平均亏损', 'avg_loss', '.2f'),
        ('最终资金', 'final_equity', '.2f')
    ]

    config_order = ['平衡型(原始)', '改进型A', '改进型B', '改进型C']

    for metric_name, metric_key, fmt in metrics:
        values = []
        for config_name in config_order:
            val = results[config_name][metric_key]
            if 'd' in fmt:
                values.append(f"{val:{fmt}}")
            else:
                values.append(f"{val:{fmt}}")
        print(f"{metric_name:<20} {values[0]:<20} {values[1]:<20} {values[2]:<20} {values[3]:<20}")

    # 出场原因分析
    print("\n" + "="*80)
    print("出场原因对比")
    print("="*80)

    for config_name in config_order:
        print(f"\n{config_name}:")
        exit_reasons = results[config_name]['exit_reasons']
        total = sum(exit_reasons.values())
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}次 ({count/total*100:.1f}%)")

    # 找出最佳配置
    print("\n" + "="*80)
    print("最佳配置评选")
    print("="*80)

    # 综合评分
    scores = {}
    for config_name in config_order:
        r = results[config_name]
        composite = (
            r['sharpe'] * 0.4 +
            r['total_return'] * 0.3 +
            (1 + r['max_drawdown']) * 0.2 +
            r['win_rate'] * 0.1
        )
        scores[config_name] = composite

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])

    print("\n综合评分排名:")
    for i, (config_name, score) in enumerate(sorted_scores, 1):
        r = results[config_name]
        marker = " [***]" if i == 1 else ""
        print(f"\n{i}. {config_name}{marker}")
        print(f"   综合评分: {score:.4f}")
        print(f"   Sharpe: {r['sharpe']:.4f} | 收益: {r['total_return']*100:.2f}% | 回撤: {r['max_drawdown']*100:.2f}% | 胜率: {r['win_rate']*100:.1f}%")

    # 改进分析
    print("\n" + "="*80)
    print("改进效果分析")
    print("="*80)

    baseline = results['平衡型(原始)']

    for config_name in ['改进型A', '改进型B', '改进型C']:
        improved = results[config_name]

        print(f"\n{config_name} vs 平衡型(原始):")
        print(f"  Sharpe: {improved['sharpe']:.4f} ({(improved['sharpe']-baseline['sharpe'])/baseline['sharpe']*100:+.1f}%)")
        print(f"  收益: {improved['total_return']*100:.2f}% ({(improved['total_return']-baseline['total_return'])*100:+.2f}个百分点)")
        print(f"  回撤: {improved['max_drawdown']*100:.2f}% ({(improved['max_drawdown']-baseline['max_drawdown'])*100:+.2f}个百分点)")
        print(f"  胜率: {improved['win_rate']*100:.1f}% ({(improved['win_rate']-baseline['win_rate'])*100:+.1f}个百分点)")
        print(f"  交易数: {improved['total_trades']} ({improved['total_trades']-baseline['total_trades']:+d}笔)")

        # 追踪止盈使用率对比
        baseline_trailing = baseline['exit_reasons'].get('trailing_stop', 0) / baseline['total_trades']
        improved_trailing = improved['exit_reasons'].get('trailing_stop', 0) / improved['total_trades']
        print(f"  追踪止盈使用率: {improved_trailing*100:.1f}% (原始{baseline_trailing*100:.1f}%, {(improved_trailing-baseline_trailing)*100:+.1f}个百分点)")

    # 推荐
    print("\n" + "="*80)
    print("最终推荐")
    print("="*80)

    best_config_name = sorted_scores[0][0]
    best_config = results[best_config_name]

    print(f"\n推荐配置: {best_config_name}")
    print(f"\n配置参数:")
    for key, value in best_config['params'].items():
        if isinstance(value, float) and 'threshold' not in key and 'multiplier' not in key:
            print(f"  {key}: {value*100:.1f}%")
        else:
            print(f"  {key}: {value}")

    print(f"\n预期表现:")
    print(f"  Sharpe比率: {best_config['sharpe']:.4f}")
    print(f"  总收益率: {best_config['total_return']*100:.2f}%")
    print(f"  最大回撤: {best_config['max_drawdown']*100:.2f}%")
    print(f"  胜率: {best_config['win_rate']*100:.1f}%")
    print(f"  交易笔数: {best_config['total_trades']}")
    print(f"  盈亏比: {best_config['profit_factor']:.2f}")

    # 保存最佳配置
    output_dir = Path('improved_results')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'best_improved_config.json', 'w') as f:
        json.dump({
            'config_name': best_config_name,
            'params': best_config['params'],
            'performance': {
                'sharpe': float(best_config['sharpe']),
                'total_return': float(best_config['total_return']),
                'max_drawdown': float(best_config['max_drawdown']),
                'win_rate': float(best_config['win_rate']),
                'total_trades': int(best_config['total_trades']),
                'profit_factor': float(best_config['profit_factor'])
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n最佳配置已保存到: {output_dir / 'best_improved_config.json'}")

    print("\n" + "="*80)
    print("改进配置测试完成!")
    print("="*80)


if __name__ == "__main__":
    main()
