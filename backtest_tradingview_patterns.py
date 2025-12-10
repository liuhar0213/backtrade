#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TradingView形态完整回测系统
包含动态止盈止损、仓位管理、详细统计
"""

import sys
import io
import pandas as pd
import numpy as np
from tradingview_patterns_optimized import TradingViewPatternDetectorOptimized

# Avoid reassigning stdout at import time (breaks pytest terminal I/O).
if __name__ == '__main__' and sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class TradingViewPatternBacktest:
    """TradingView形态回测引擎"""

    def __init__(self, df, fee_rate=0.0005):
        """
        初始化回测引擎

        Args:
            df: K线数据
            fee_rate: 手续费率（单边，默认0.05%）
        """
        self.df = df.copy()
        self.fee_rate = fee_rate

        # 计算ATR用于动态止损
        self.df['atr'] = self._calculate_atr(14)

    def _calculate_atr(self, period=14):
        """计算ATR"""
        df = self.df
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr

    def backtest_pattern(self, signals, pattern_name,
                         sl_atr_multiplier=2.0,
                         tp_atr_multiplier=3.0,
                         hold_periods=20):
        """
        回测单个形态

        Args:
            signals: 信号Series (True/False)
            pattern_name: 形态名称
            sl_atr_multiplier: 止损ATR倍数
            tp_atr_multiplier: 止盈ATR倍数
            hold_periods: 最大持仓周期

        Returns:
            dict: 回测结果
        """
        if signals.sum() == 0:
            return self._empty_result(pattern_name)

        # 判断形态方向
        direction = self._get_direction(pattern_name)
        if direction == 0:
            return self._empty_result(pattern_name)

        # 找到所有信号
        signal_indices = self.df[signals].index.tolist()

        trades = []
        for idx in signal_indices:
            if idx >= len(self.df) - 1:
                continue

            # 入场
            entry_price = self.df.loc[idx, 'close']
            atr = self.df.loc[idx, 'atr']

            if pd.isna(atr) or atr == 0:
                continue

            # 计算止损止盈
            if direction == 1:  # 做多
                stop_loss = entry_price - sl_atr_multiplier * atr
                take_profit = entry_price + tp_atr_multiplier * atr
            else:  # 做空
                stop_loss = entry_price + sl_atr_multiplier * atr
                take_profit = entry_price - tp_atr_multiplier * atr

            # 模拟持仓
            exit_idx = None
            exit_price = None
            exit_reason = 'max_hold'

            max_idx = min(idx + hold_periods, len(self.df) - 1)

            for i in range(idx + 1, max_idx + 1):
                high = self.df.loc[i, 'high']
                low = self.df.loc[i, 'low']
                close = self.df.loc[i, 'close']

                # 检查止损止盈
                if direction == 1:
                    if low <= stop_loss:
                        exit_idx = i
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                        break
                    elif high >= take_profit:
                        exit_idx = i
                        exit_price = take_profit
                        exit_reason = 'take_profit'
                        break
                else:
                    if high >= stop_loss:
                        exit_idx = i
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                        break
                    elif low <= take_profit:
                        exit_idx = i
                        exit_price = take_profit
                        exit_reason = 'take_profit'
                        break

            # 如果没触发止损止盈，按收盘价平仓
            if exit_idx is None:
                exit_idx = max_idx
                exit_price = self.df.loc[exit_idx, 'close']

            # 计算收益
            gross_return = direction * (exit_price - entry_price) / entry_price
            net_return = gross_return - 2 * self.fee_rate

            trades.append({
                'entry_idx': idx,
                'exit_idx': exit_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'gross_return': gross_return,
                'net_return': net_return,
                'exit_reason': exit_reason,
                'win': net_return > 0,
                'hold_periods': exit_idx - idx
            })

        if len(trades) == 0:
            return self._empty_result(pattern_name)

        # 统计分析
        trades_df = pd.DataFrame(trades)
        return self._calculate_stats(trades_df, pattern_name)

    def _get_direction(self, pattern_name):
        """获取形态方向（1=做多, -1=做空, 0=中性）"""
        bullish_keywords = [
            'Bullish', 'Hammer', 'Piercing', 'Morning', 'White_Soldiers',
            'Bottom', 'Rising', 'Upside', 'Dragonfly', 'Inverted_Hammer'
        ]
        bearish_keywords = [
            'Bearish', 'Shooting', 'Hanging', 'Dark_Cloud', 'Evening',
            'Black_Crows', 'Top', 'Falling', 'Downside', 'Gravestone'
        ]

        if any(kw in pattern_name for kw in bullish_keywords):
            return 1
        elif any(kw in pattern_name for kw in bearish_keywords):
            return -1
        else:
            return 0

    def _calculate_stats(self, trades_df, pattern_name):
        """计算统计指标"""
        total_trades = len(trades_df)
        winning_trades = trades_df['win'].sum()
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        total_return = trades_df['net_return'].sum() * 100
        avg_return = trades_df['net_return'].mean() * 100

        avg_win = trades_df[trades_df['win']]['net_return'].mean() * 100 if winning_trades > 0 else 0
        avg_loss = trades_df[~trades_df['win']]['net_return'].mean() * 100 if losing_trades > 0 else 0

        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')

        max_return = trades_df['net_return'].max() * 100
        min_return = trades_df['net_return'].min() * 100

        # 计算最大回撤
        cumulative_returns = (1 + trades_df['net_return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # 退出原因统计
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

        # 平均持仓周期
        avg_hold = trades_df['hold_periods'].mean()

        return {
            'pattern': pattern_name,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_return': max_return,
            'min_return': min_return,
            'max_drawdown': max_drawdown,
            'avg_hold_periods': avg_hold,
            'stop_loss_exits': exit_reasons.get('stop_loss', 0),
            'take_profit_exits': exit_reasons.get('take_profit', 0),
            'max_hold_exits': exit_reasons.get('max_hold', 0)
        }

    def _empty_result(self, pattern_name):
        """空结果"""
        return {
            'pattern': pattern_name,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'avg_return': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_return': 0,
            'min_return': 0,
            'max_drawdown': 0,
            'avg_hold_periods': 0,
            'stop_loss_exits': 0,
            'take_profit_exits': 0,
            'max_hold_exits': 0
        }


def main():
    print("=" * 100)
    print("TradingView形态完整回测系统")
    print("=" * 100)
    print()

    # 加载数据
    print("加载数据...")
    df = pd.read_csv('data/ETHUSDT_15.csv')
    print(f"数据: {len(df)} 根K线")
    print()

    # 创建检测器
    print("检测所有形态...")
    detector = TradingViewPatternDetectorOptimized(df, trend_rule='SMA50')
    patterns = detector.detect_all_patterns()
    print(f"检测完成: {len([c for c in patterns.columns if patterns[c].sum() > 0])} 种形态有信号")
    print()

    # 创建回测引擎
    print("初始化回测引擎...")
    backtest = TradingViewPatternBacktest(df, fee_rate=0.0005)
    print()

    # 回测所有形态
    print("=" * 100)
    print("开始回测...")
    print("=" * 100)
    print()

    all_results = []

    for col in patterns.columns:
        if patterns[col].sum() == 0:
            continue

        print(f"回测: {col}...", end=' ')

        result = backtest.backtest_pattern(
            patterns[col],
            col,
            sl_atr_multiplier=2.0,
            tp_atr_multiplier=3.0,
            hold_periods=20
        )

        all_results.append(result)
        print(f"完成 ({result['total_trades']} 笔交易)")

    print()
    print("=" * 100)
    print("回测完成！")
    print("=" * 100)
    print()

    # 生成结果DataFrame
    results_df = pd.DataFrame(all_results)

    # 保存结果
    results_df.to_csv('tradingview_backtest_results.csv', index=False, encoding='utf-8-sig')
    print("结果已保存: tradingview_backtest_results.csv")
    print()

    # 显示排名
    print("=" * 100)
    print("形态性能排名（按总收益）")
    print("=" * 100)
    print()

    # 过滤至少有10笔交易的形态
    valid_results = results_df[results_df['total_trades'] >= 10].copy()

    if len(valid_results) > 0:
        # 按总收益排序
        valid_results = valid_results.sort_values('total_return', ascending=False)

        print(f"{'排名':<5s} | {'形态名称':<35s} | {'交易':>6s} | {'胜率':>8s} | {'总收益':>10s} | {'盈亏比':>8s}")
        print("-" * 100)

        for rank, (_, row) in enumerate(valid_results.head(20).iterrows(), 1):
            print(f"{rank:<5d} | {row['pattern']:<35s} | {row['total_trades']:>6d} | "
                  f"{row['win_rate']:>7.1f}% | {row['total_return']:>9.2f}% | "
                  f"{row['profit_factor']:>8.2f}")

        print()
        print("=" * 100)
        print("详细统计")
        print("=" * 100)
        print()

        # Top 5形态详细信息
        print("【Top 5 最佳形态】")
        print("-" * 100)

        for rank, (_, row) in enumerate(valid_results.head(5).iterrows(), 1):
            print(f"\n{rank}. {row['pattern']}")
            print(f"   交易次数: {row['total_trades']}")
            print(f"   胜率: {row['win_rate']:.1f}% ({row['winning_trades']}胜/{row['losing_trades']}负)")
            print(f"   总收益: {row['total_return']:.2f}%")
            print(f"   平均收益: {row['avg_return']:.2f}%")
            print(f"   平均盈利: {row['avg_win']:.2f}%")
            print(f"   平均亏损: {row['avg_loss']:.2f}%")
            print(f"   盈亏比: {row['profit_factor']:.2f}")
            print(f"   最大回撤: {row['max_drawdown']:.2f}%")
            print(f"   平均持仓: {row['avg_hold_periods']:.1f} 根K线")
            print(f"   退出方式: 止盈{row['take_profit_exits']} | 止损{row['stop_loss_exits']} | 最大持仓{row['max_hold_exits']}")

        print()
        print("=" * 100)
        print("分类统计")
        print("=" * 100)
        print()

        # 分类统计
        bullish_patterns = valid_results[valid_results['pattern'].str.contains('Bullish|Hammer|Piercing|Morning|Bottom|Rising|Upside|Dragonfly|Inverted', case=False, regex=True)]
        bearish_patterns = valid_results[valid_results['pattern'].str.contains('Bearish|Shooting|Hanging|Dark|Evening|Top|Falling|Downside|Gravestone', case=False, regex=True)]

        print(f"看涨形态: {len(bullish_patterns)} 个")
        if len(bullish_patterns) > 0:
            print(f"  平均胜率: {bullish_patterns['win_rate'].mean():.1f}%")
            print(f"  平均总收益: {bullish_patterns['total_return'].mean():.2f}%")
            print(f"  最佳形态: {bullish_patterns.iloc[0]['pattern']} ({bullish_patterns.iloc[0]['total_return']:.2f}%)")

        print()
        print(f"看跌形态: {len(bearish_patterns)} 个")
        if len(bearish_patterns) > 0:
            print(f"  平均胜率: {bearish_patterns['win_rate'].mean():.1f}%")
            print(f"  平均总收益: {bearish_patterns['total_return'].mean():.2f}%")
            print(f"  最佳形态: {bearish_patterns.iloc[0]['pattern']} ({bearish_patterns.iloc[0]['total_return']:.2f}%)")

    else:
        print("没有足够的交易数据（至少10笔）用于分析")

    print()
    print("=" * 100)


if __name__ == '__main__':
    main()
