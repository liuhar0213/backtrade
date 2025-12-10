"""
包含手续费计算的优化版回测系统
手续费: 0.05% 单边 (0.1% 双边)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from find_correct_patterns import CorrectGoldenKDetector

class PatternBacktesterWithFees:
    """包含手续费的形态回测器"""

    def __init__(self, df, fee_rate=0.0005):
        """
        初始化回测器

        参数:
            df: DataFrame，包含 timestamp, open, high, low, close 列
            fee_rate: 单边手续费率，默认0.05% (Taker费率)
        """
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.fee_rate = fee_rate  # 单边手续费

        # 计算MA20和MA60
        self.df['ma20'] = self.df['close'].rolling(window=20).mean()
        self.df['ma60'] = self.df['close'].rolling(window=60).mean()

        # 初始化形态检测器
        detector_df = pd.DataFrame({
            'time': self.df['timestamp'],
            'open': self.df['open'],
            'high': self.df['high'],
            'low': self.df['low'],
            'close': self.df['close'],
            'volume': self.df.get('volume', [1] * len(self.df))
        })
        self.detector = CorrectGoldenKDetector(detector_df)

    def get_trend(self, i):
        """判断当前趋势"""
        if i < 60:
            return None

        ma20 = self.df.iloc[i]['ma20']
        ma60 = self.df.iloc[i]['ma60']
        price = self.df.iloc[i]['close']

        if pd.isna(ma20) or pd.isna(ma60):
            return None

        # 上涨趋势：MA20 > MA60 且 价格 > MA20
        if ma20 > ma60 and price > ma20:
            return 'uptrend'
        # 下跌趋势：MA20 < MA60 且 价格 < MA20
        elif ma20 < ma60 and price < ma20:
            return 'downtrend'
        else:
            return 'sideways'

    def simulate_trade(self, entry_idx, entry_price, stop_loss, direction,
                      pattern_name, entry_method):
        """
        模拟单笔交易（包含手续费）

        参数:
            entry_idx: 进场K线索引
            entry_price: 进场价格
            stop_loss: 止损价格
            direction: 'long' 或 'short'
            pattern_name: 形态名称
            entry_method: 进场方式

        返回:
            交易记录字典，如果无效返回None
        """
        if entry_idx >= len(self.df):
            return None

        risk = abs(entry_price - stop_loss)
        if risk == 0:
            return None

        # 2倍风险止盈
        if direction == 'long':
            take_profit = entry_price + risk * 2
        else:
            take_profit = entry_price - risk * 2

        entry_time = self.df.iloc[entry_idx]['timestamp']

        # 计算进场手续费
        entry_fee = entry_price * self.fee_rate

        # 模拟持仓，最多50根K线
        for j in range(entry_idx, min(entry_idx + 50, len(self.df))):
            candle = self.df.iloc[j]

            if direction == 'long':
                # 做多：先检查止损，再检查止盈
                if candle['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                elif candle['high'] >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                else:
                    continue

                # 计算出场手续费
                exit_fee = exit_price * self.fee_rate

                # 计算盈亏（扣除手续费）
                gross_pnl = exit_price - entry_price
                net_pnl = gross_pnl - entry_fee - exit_fee
                pnl_pct = net_pnl / entry_price * 100

            else:  # short
                # 做空：先检查止损，再检查止盈
                if candle['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                elif candle['low'] <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                else:
                    continue

                # 计算出场手续费
                exit_fee = exit_price * self.fee_rate

                # 计算盈亏（扣除手续费）
                gross_pnl = entry_price - exit_price
                net_pnl = gross_pnl - entry_fee - exit_fee
                pnl_pct = net_pnl / entry_price * 100

            return {
                'pattern': pattern_name,
                'entry_method': entry_method,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'direction': direction,
                'exit_time': candle['timestamp'],
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'gross_pnl_pct': (exit_price - entry_price) / entry_price * 100 if direction == 'long' else (entry_price - exit_price) / entry_price * 100,
                'fees_pct': (entry_fee + exit_fee) / entry_price * 100,
                'net_pnl_pct': pnl_pct,
                'holding_bars': j - entry_idx + 1
            }

        # 超时退出
        last_candle = self.df.iloc[min(entry_idx + 49, len(self.df) - 1)]
        exit_price = last_candle['close']

        # 计算出场手续费
        exit_fee = exit_price * self.fee_rate

        if direction == 'long':
            gross_pnl = exit_price - entry_price
            net_pnl = gross_pnl - entry_fee - exit_fee
            pnl_pct = net_pnl / entry_price * 100
            gross_pnl_pct = gross_pnl / entry_price * 100
        else:
            gross_pnl = entry_price - exit_price
            net_pnl = gross_pnl - entry_fee - exit_fee
            pnl_pct = net_pnl / entry_price * 100
            gross_pnl_pct = gross_pnl / entry_price * 100

        return {
            'pattern': pattern_name,
            'entry_method': entry_method,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'direction': direction,
            'exit_time': last_candle['timestamp'],
            'exit_price': exit_price,
            'exit_reason': 'timeout',
            'gross_pnl_pct': gross_pnl_pct,
            'fees_pct': (entry_fee + exit_fee) / entry_price * 100,
            'net_pnl_pct': pnl_pct,
            'holding_bars': 50
        }

    def calculate_fibonacci_levels(self, high, low, direction):
        """计算斐波那契回撤位"""
        if direction == 'long':
            return {
                '31.8%': low + (high - low) * 0.618,
                '50%': (high + low) / 2
            }
        else:
            return {
                '31.8%': high - (high - low) * 0.618,
                '50%': (high + low) / 2
            }

    # ========== 优化版形态回测方法（包含手续费）==========

    def backtest_evening_star_optimized(self):
        """回测黄昏之星（优化版 + 手续费）"""
        results = []

        for i in range(len(self.df)):
            if not self.detector.detect_evening_star(i):
                continue

            trend = self.get_trend(i)
            if trend != 'uptrend':
                continue

            k1 = self.df.iloc[i-2]
            k2 = self.df.iloc[i-1]
            k3 = self.df.iloc[i]

            pattern_candles = self.df.iloc[i-2:i+1]
            pattern_high = pattern_candles['high'].max()
            pattern_low = pattern_candles['low'].min()
            stop_loss = pattern_high

            # 突破进场
            if i + 1 < len(self.df):
                next_candle = self.df.iloc[i+1]
                if next_candle['open'] < k3['close']:
                    entry_price = next_candle['open']
                    trade = self.simulate_trade(i+1, entry_price, stop_loss, 'short',
                                                '黄昏之星', '突破进场')
                    if trade:
                        results.append(trade)

        return results

    def backtest_bull_cannon_optimized(self):
        """回测多方炮（优化版 + 手续费）"""
        results = []

        for i in range(len(self.df)):
            if not self.detector.detect_bullish_cannon(i):
                continue

            trend = self.get_trend(i)
            if trend == 'downtrend':
                continue

            k1 = self.df.iloc[i-2]
            k2 = self.df.iloc[i-1]
            k3 = self.df.iloc[i]

            pattern_candles = self.df.iloc[i-2:i+1]
            pattern_low = pattern_candles['low'].min()
            stop_loss = pattern_low

            # 突破进场
            if i + 1 < len(self.df):
                next_candle = self.df.iloc[i+1]
                if next_candle['open'] > k3['close']:
                    entry_price = next_candle['open']
                    trade = self.simulate_trade(i+1, entry_price, stop_loss, 'long',
                                                '多方炮', '突破进场')
                    if trade:
                        results.append(trade)

        return results

    def backtest_three_soldiers_optimized(self):
        """回测三兵（优化版 + 手续费）"""
        results = []

        for i in range(len(self.df)):
            if not self.detector.detect_three_soldiers(i):
                continue

            trend = self.get_trend(i)
            if trend == 'downtrend':
                continue

            pattern_candles = self.df.iloc[i-2:i+1]
            pattern_low = pattern_candles['low'].min()
            stop_loss = pattern_low

            k3 = self.df.iloc[i]

            # 突破进场
            if i + 1 < len(self.df):
                next_candle = self.df.iloc[i+1]
                if next_candle['open'] > k3['close']:
                    entry_price = next_candle['open']
                    trade = self.simulate_trade(i+1, entry_price, stop_loss, 'long',
                                                '三兵', '突破进场')
                    if trade:
                        results.append(trade)

        return results

    def backtest_hammer_optimized(self):
        """回测锤形线（优化版 + 手续费）"""
        results = []

        for i in range(len(self.df)):
            if not self.detector.detect_hammer(i):
                continue

            trend = self.get_trend(i)
            if trend == 'uptrend':
                continue

            curr = self.df.iloc[i]
            stop_loss = curr['low']

            # 突破进场
            if i + 1 < len(self.df):
                next_candle = self.df.iloc[i+1]
                if next_candle['open'] > curr['close']:
                    entry_price = next_candle['open']
                    trade = self.simulate_trade(i+1, entry_price, stop_loss, 'long',
                                                '锤形线', '突破进场')
                    if trade:
                        results.append(trade)

        return results

    def backtest_shooting_star_optimized(self):
        """回测射击之星（优化版 + 手续费）"""
        results = []

        for i in range(len(self.df)):
            if not self.detector.detect_shooting_star(i):
                continue

            trend = self.get_trend(i)
            if trend != 'uptrend':
                continue

            curr = self.df.iloc[i]
            stop_loss = curr['high']

            # 突破进场
            if i + 1 < len(self.df):
                next_candle = self.df.iloc[i+1]
                if next_candle['open'] < curr['close']:
                    entry_price = next_candle['open']
                    trade = self.simulate_trade(i+1, entry_price, stop_loss, 'short',
                                                '射击之星', '突破进场')
                    if trade:
                        results.append(trade)

        return results

    def backtest_piercing_optimized(self):
        """回测刺透（优化版 + 手续费）"""
        results = []

        for i in range(len(self.df)):
            if not self.detector.detect_piercing(i):
                continue

            trend = self.get_trend(i)
            if trend == 'uptrend':
                continue

            k1 = self.df.iloc[i-1]
            k2 = self.df.iloc[i]

            pattern_high = max(k1['high'], k2['high'])
            pattern_low = min(k1['low'], k2['low'])
            stop_loss = pattern_low

            # 回撤31.8%进场
            fib_levels = self.calculate_fibonacci_levels(pattern_high, pattern_low, 'long')

            for j in range(i+1, min(i+10, len(self.df))):
                candle = self.df.iloc[j]
                if candle['low'] <= fib_levels['31.8%']:
                    entry_price = fib_levels['31.8%']
                    trade = self.simulate_trade(j, entry_price, stop_loss, 'long',
                                                '刺透', '回撤31.8%')
                    if trade:
                        results.append(trade)
                    break

        return results

    def run_all_backtests(self):
        """运行所有优化版回测（包含手续费）"""

        print("\n" + "="*80)
        print("优化版回测系统（包含手续费）")
        print("="*80)
        print(f"\n数据范围: {self.df['timestamp'].min()} 至 {self.df['timestamp'].max()}")
        print(f"总K线数: {len(self.df)}")
        print(f"手续费率: {self.fee_rate * 100:.3f}% 单边 ({self.fee_rate * 200:.3f}% 双边)\n")

        # 定义优化版形态列表
        patterns = [
            ('黄昏之星', self.backtest_evening_star_optimized),
            ('多方炮', self.backtest_bull_cannon_optimized),
            ('三兵', self.backtest_three_soldiers_optimized),
            ('锤形线', self.backtest_hammer_optimized),
            ('射击之星', self.backtest_shooting_star_optimized),
            ('刺透', self.backtest_piercing_optimized),
        ]

        all_trades = []
        pattern_stats = []

        for pattern_name, backtest_func in patterns:
            print(f"正在回测: {pattern_name}...")
            trades = backtest_func()

            if len(trades) == 0:
                print(f"  未产生任何交易\n")
                continue

            all_trades.extend(trades)

            # 统计
            df_trades = pd.DataFrame(trades)
            win_trades = df_trades[df_trades['net_pnl_pct'] > 0]
            lose_trades = df_trades[df_trades['net_pnl_pct'] < 0]

            total = len(df_trades)
            wins = len(win_trades)
            losses = len(lose_trades)
            win_rate = wins / total * 100 if total > 0 else 0

            avg_gross_pnl = df_trades['gross_pnl_pct'].mean()
            avg_fees = df_trades['fees_pct'].mean()
            avg_net_pnl = df_trades['net_pnl_pct'].mean()
            total_net_pnl = df_trades['net_pnl_pct'].sum()
            total_fees = df_trades['fees_pct'].sum()

            print(f"  交易次数: {total}, 胜率: {win_rate:.2f}%")
            print(f"  平均毛利: {avg_gross_pnl:.2f}%, 平均手续费: {avg_fees:.2f}%, 平均净利: {avg_net_pnl:.2f}%")
            print(f"  总净利: {total_net_pnl:.2f}%, 总手续费: {total_fees:.2f}%")
            print()

            pattern_stats.append({
                'pattern': pattern_name,
                'total_trades': total,
                'win_rate': win_rate,
                'avg_net_pnl': avg_net_pnl,
                'total_net_pnl': total_net_pnl,
                'total_fees': total_fees
            })

        # 整体统计
        print("\n" + "="*80)
        print("整体统计（包含手续费）")
        print("="*80)

        if len(all_trades) > 0:
            df_all = pd.DataFrame(all_trades)
            total_trades = len(df_all)
            total_wins = len(df_all[df_all['net_pnl_pct'] > 0])
            overall_win_rate = total_wins / total_trades * 100
            overall_avg_net_pnl = df_all['net_pnl_pct'].mean()
            overall_total_net_pnl = df_all['net_pnl_pct'].sum()
            overall_total_fees = df_all['fees_pct'].sum()

            print(f"\n总交易次数: {total_trades}")
            print(f"总胜率: {overall_win_rate:.2f}%")
            print(f"平均净利: {overall_avg_net_pnl:.2f}%")
            print(f"累计净利: {overall_total_net_pnl:.2f}%")
            print(f"累计手续费: {overall_total_fees:.2f}%")

            return all_trades, {
                'total_trades': total_trades,
                'win_rate': overall_win_rate,
                'avg_net_pnl': overall_avg_net_pnl,
                'total_net_pnl': overall_total_net_pnl,
                'total_fees': overall_total_fees
            }
        else:
            print("\n未产生任何交易！")
            return [], {}


def main():
    """主函数"""

    # 读取数据
    print("正在加载数据: data/ETHUSDT_15.csv")
    df = pd.read_csv('data/ETHUSDT_15.csv')

    # 转换时间列
    df['timestamp'] = pd.to_datetime(df['time'])

    # 筛选2025-05-01之后的数据
    df = df[df['timestamp'] >= '2025-05-01'].reset_index(drop=True)
    print(f"筛选后数据: {len(df)} 根K线\n")

    # 运行包含手续费的回测
    backtester = PatternBacktesterWithFees(df, fee_rate=0.0005)  # 0.05% 单边
    trades, summary = backtester.run_all_backtests()

    # 保存结果
    if len(trades) > 0:
        df_trades = pd.DataFrame(trades)
        df_trades.to_csv('backtest_results_with_fees.csv', index=False, encoding='utf-8-sig')
        print(f"\n交易记录已保存到: backtest_results_with_fees.csv")

        # 对比
        print("\n" + "="*80)
        print("对比分析：包含手续费 vs 不含手续费")
        print("="*80)

        # 读取原始优化版数据（不含手续费）
        try:
            original_df = pd.read_csv('optimized_backtest_results.csv')

            print(f"\n不含手续费版本:")
            print(f"  总交易: {len(original_df)}次")
            print(f"  总胜率: {(original_df['pnl_pct'] > 0).sum() / len(original_df) * 100:.2f}%")
            print(f"  累计收益: {original_df['pnl_pct'].sum():.2f}%")

            print(f"\n包含手续费版本:")
            print(f"  总交易: {summary['total_trades']}次")
            print(f"  总胜率: {summary['win_rate']:.2f}%")
            print(f"  累计净利: {summary['total_net_pnl']:.2f}%")
            print(f"  累计手续费: {summary['total_fees']:.2f}%")

            impact = summary['total_net_pnl'] - original_df['pnl_pct'].sum()
            print(f"\n手续费影响: {impact:.2f}% ({impact / original_df['pnl_pct'].sum() * 100:.1f}%)")

        except:
            pass

    print("\n回测完成！")


if __name__ == '__main__':
    main()


def run_backtests(df: pd.DataFrame = None, csv_path: str = 'data/ETHUSDT_15.csv', fee_rate: float = 0.0005, save_csv: bool = True):
    """Programmatic entry point for the backtest-with-fees workflow.

    If `df` is None the CSV at `csv_path` will be loaded and preprocessed
    the same way `main()` does. Returns `(trades, summary)`.
    """
    if df is None:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['time'])
        df = df[df['timestamp'] >= '2025-05-01'].reset_index(drop=True)

    backtester = PatternBacktesterWithFees(df, fee_rate=fee_rate)
    trades, summary = backtester.run_all_backtests()

    if save_csv and len(trades) > 0:
        df_trades = pd.DataFrame(trades)
        try:
            df_trades.to_csv('backtest_results_with_fees.csv', index=False, encoding='utf-8-sig')
        except PermissionError:
            pass

    return trades, summary
