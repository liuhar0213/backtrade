#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume + Supertrend + Sessions 策略回测
基于 Volume_Supertrend_Sessions_Strategy.pine
"""

import pandas as pd
import numpy as np
import sys
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Avoid reassigning stdout at import time (breaks pytest terminal I/O).
if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class VolumeSupertrendSessionsStrategy:
    """
    Volume + Supertrend + Sessions 融合策略
    """

    def __init__(self, data,
                 atr_period=10,
                 st_factor=2.5,
                 vol_length=50,
                 vol_multiplier=1.0,
                 require_vol_confirm=False,
                 min_vol_ratio=1.3,
                 only_strong_signals=False,
                 use_session_filter=True,
                 stop_loss_type='ATR',
                 stop_loss_atr=1.8,
                 stop_loss_percent=2.0,
                 take_profit_rr=2.0):

        self.data = data.copy()
        self.atr_period = atr_period
        self.st_factor = st_factor
        self.vol_length = vol_length
        self.vol_multiplier = vol_multiplier
        self.require_vol_confirm = require_vol_confirm
        self.min_vol_ratio = min_vol_ratio
        self.only_strong_signals = only_strong_signals
        self.use_session_filter = use_session_filter
        self.stop_loss_type = stop_loss_type
        self.stop_loss_atr = stop_loss_atr
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_rr = take_profit_rr

        # 初始化
        self.initial_capital = 10000
        self.position_size_pct = 0.5  # 50%仓位
        self.commission = 0.001  # 0.1%
        self.slippage = 0.0002  # 0.02%

    def calculate_supertrend(self):
        """计算Supertrend指标"""
        df = self.data

        # 计算ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()

        # 计算基础带
        df['upper_band'] = (df['high'] + df['low']) / 2 + self.st_factor * df['atr']
        df['lower_band'] = (df['high'] + df['low']) / 2 - self.st_factor * df['atr']

        # 计算Supertrend
        df['supertrend'] = 0.0
        df['direction'] = 1

        for i in range(1, len(df)):
            # 更新upper band
            if df['close'].iloc[i-1] <= df['upper_band'].iloc[i-1]:
                df.loc[df.index[i], 'upper_band'] = min(df['upper_band'].iloc[i], df['upper_band'].iloc[i-1])

            # 更新lower band
            if df['close'].iloc[i-1] >= df['lower_band'].iloc[i-1]:
                df.loc[df.index[i], 'lower_band'] = max(df['lower_band'].iloc[i], df['lower_band'].iloc[i-1])

            # 确定方向
            if df['close'].iloc[i] > df['upper_band'].iloc[i-1]:
                df.loc[df.index[i], 'direction'] = -1  # 上升趋势
            elif df['close'].iloc[i] < df['lower_band'].iloc[i-1]:
                df.loc[df.index[i], 'direction'] = 1   # 下降趋势
            else:
                df.loc[df.index[i], 'direction'] = df['direction'].iloc[i-1]

            # 设置Supertrend值
            if df['direction'].iloc[i] == -1:
                df.loc[df.index[i], 'supertrend'] = df['lower_band'].iloc[i]
            else:
                df.loc[df.index[i], 'supertrend'] = df['upper_band'].iloc[i]

        return df

    def calculate_volume_surprise(self):
        """计算Volume Surprise（简化版，基于小时分组）"""
        df = self.data

        # 提取小时作为时段标识
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour

        # 计算每个小时的预期成交量（历史平均）
        df['expected_volume'] = df.groupby('hour')['volume'].transform(
            lambda x: x.rolling(window=min(self.vol_length, len(x)), min_periods=1).mean()
        )

        # 计算成交量比率
        df['vol_ratio'] = df['volume'] / df['expected_volume']
        df['vol_ratio'] = df['vol_ratio'].fillna(1.0)

        # 成交量确认
        df['is_high_volume'] = df['vol_ratio'] >= self.vol_multiplier
        df['is_strong_volume'] = df['vol_ratio'] >= self.min_vol_ratio

        return df

    def identify_sessions(self):
        """识别交易时段（UTC时间）"""
        df = self.data
        dt = pd.to_datetime(df['datetime'])
        hour = dt.dt.hour

        # 定义时段 (UTC时间)
        # 纽约: 13:00-22:00 UTC
        # 伦敦: 07:00-16:00 UTC
        # 东京: 00:00-09:00 UTC
        # 悉尼: 21:00-06:00 UTC (跨天)

        df['session_ny'] = (hour >= 13) & (hour < 22)
        df['session_london'] = (hour >= 7) & (hour < 16)
        df['session_tokyo'] = (hour >= 0) & (hour < 9)
        df['session_sydney'] = (hour >= 21) | (hour < 6)

        # 默认开启纽约+伦敦
        if self.use_session_filter:
            df['in_session'] = df['session_ny'] | df['session_london']
        else:
            df['in_session'] = True

        return df

    def generate_signals(self):
        """生成交易信号"""
        df = self.data

        # Supertrend趋势判断
        df['st_uptrend'] = df['direction'] < 0
        df['st_downtrend'] = df['direction'] > 0

        # Supertrend反转信号
        df['st_to_uptrend'] = (df['direction'] < 0) & (df['direction'].shift(1) > 0)
        df['st_to_downtrend'] = (df['direction'] > 0) & (df['direction'].shift(1) < 0)

        # 成交量确认
        if self.require_vol_confirm:
            df['volume_confirmed'] = df['is_high_volume']
        else:
            df['volume_confirmed'] = True

        # 基础信号
        df['long_condition'] = df['st_to_uptrend'] & df['volume_confirmed'] & df['in_session']
        df['short_condition'] = df['st_to_downtrend'] & df['volume_confirmed'] & df['in_session']

        # 强信号
        df['strong_long'] = df['st_to_uptrend'] & df['is_strong_volume'] & df['in_session']
        df['strong_short'] = df['st_to_downtrend'] & df['is_strong_volume'] & df['in_session']

        # 如果只交易强信号
        if self.only_strong_signals:
            df['long_condition'] = df['strong_long']
            df['short_condition'] = df['strong_short']

        return df

    def calculate_stop_take(self, entry_price, is_long, atr_value):
        """计算止损止盈"""
        if self.stop_loss_type == 'ATR':
            if is_long:
                stop = entry_price - (atr_value * self.stop_loss_atr)
            else:
                stop = entry_price + (atr_value * self.stop_loss_atr)
        elif self.stop_loss_type == 'Percent':
            if is_long:
                stop = entry_price * (1 - self.stop_loss_percent / 100)
            else:
                stop = entry_price * (1 + self.stop_loss_percent / 100)

        # 计算止盈
        risk = abs(entry_price - stop)
        reward = risk * self.take_profit_rr

        if is_long:
            take = entry_price + reward
        else:
            take = entry_price - reward

        return stop, take

    def backtest(self):
        """执行回测"""
        print("[*] 开始回测...")
        print(f"初始资金: ${self.initial_capital:,.2f}")
        print(f"数据范围: {self.data['datetime'].iloc[0]} 至 {self.data['datetime'].iloc[-1]}")
        print(f"总K线数: {len(self.data)}")
        print("="*60)

        # 计算指标
        self.data = self.calculate_supertrend()
        self.data = self.calculate_volume_surprise()
        self.data = self.identify_sessions()
        self.data = self.generate_signals()

        # 初始化账户
        cash = self.initial_capital
        position = 0  # 0: 无仓位, 1: 多头, -1: 空头
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        shares = 0

        trades = []
        equity_curve = [cash]

        for i in range(1, len(self.data)):
            row = self.data.iloc[i]
            prev_row = self.data.iloc[i-1]

            current_price = row['close']
            high = row['high']
            low = row['low']
            atr = row['atr']

            # 检查出场条件
            if position != 0:
                # 检查止损止盈
                if position == 1:  # 多头
                    if low <= stop_loss:
                        # 止损
                        exit_price = stop_loss * (1 - self.slippage)
                        pnl = (exit_price - entry_price) * shares - abs(exit_price * shares * self.commission)
                        cash += exit_price * shares

                        trades.append({
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': row['datetime'],
                            'exit_price': exit_price,
                            'type': 'LONG',
                            'exit_reason': 'STOP LOSS',
                            'pnl': pnl,
                            'pnl_pct': (exit_price / entry_price - 1) * 100,
                            'shares': shares
                        })

                        position = 0
                        shares = 0

                    elif high >= take_profit:
                        # 止盈
                        exit_price = take_profit * (1 - self.slippage)
                        pnl = (exit_price - entry_price) * shares - abs(exit_price * shares * self.commission)
                        cash += exit_price * shares

                        trades.append({
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': row['datetime'],
                            'exit_price': exit_price,
                            'type': 'LONG',
                            'exit_reason': 'TAKE PROFIT',
                            'pnl': pnl,
                            'pnl_pct': (exit_price / entry_price - 1) * 100,
                            'shares': shares
                        })

                        position = 0
                        shares = 0

                elif position == -1:  # 空头
                    if high >= stop_loss:
                        # 止损
                        exit_price = stop_loss * (1 + self.slippage)
                        pnl = (entry_price - exit_price) * shares - abs(exit_price * shares * self.commission)
                        cash += pnl + (entry_price * shares)

                        trades.append({
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': row['datetime'],
                            'exit_price': exit_price,
                            'type': 'SHORT',
                            'exit_reason': 'STOP LOSS',
                            'pnl': pnl,
                            'pnl_pct': (entry_price / exit_price - 1) * 100,
                            'shares': shares
                        })

                        position = 0
                        shares = 0

                    elif low <= take_profit:
                        # 止盈
                        exit_price = take_profit * (1 + self.slippage)
                        pnl = (entry_price - exit_price) * shares - abs(exit_price * shares * self.commission)
                        cash += pnl + (entry_price * shares)

                        trades.append({
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': row['datetime'],
                            'exit_price': exit_price,
                            'type': 'SHORT',
                            'exit_reason': 'TAKE PROFIT',
                            'pnl': pnl,
                            'pnl_pct': (entry_price / exit_price - 1) * 100,
                            'shares': shares
                        })

                        position = 0
                        shares = 0

            # 检查入场条件
            if position == 0:
                if row['long_condition']:
                    # 开多
                    position = 1
                    entry_price = current_price * (1 + self.slippage)
                    entry_time = row['datetime']
                    shares = (cash * self.position_size_pct) / entry_price
                    cash -= entry_price * shares * (1 + self.commission)

                    stop_loss, take_profit = self.calculate_stop_take(entry_price, True, atr)

                elif row['short_condition']:
                    # 开空
                    position = -1
                    entry_price = current_price * (1 - self.slippage)
                    entry_time = row['datetime']
                    shares = (cash * self.position_size_pct) / entry_price
                    cash -= entry_price * shares * self.commission

                    stop_loss, take_profit = self.calculate_stop_take(entry_price, False, atr)

            # 记录权益
            if position == 1:
                equity = cash + current_price * shares
            elif position == -1:
                equity = cash + (entry_price - current_price) * shares
            else:
                equity = cash

            equity_curve.append(equity)

        # 回测结束，平掉剩余仓位
        if position != 0:
            final_price = self.data.iloc[-1]['close']
            if position == 1:
                exit_price = final_price * (1 - self.slippage)
                pnl = (exit_price - entry_price) * shares - abs(exit_price * shares * self.commission)
                cash += exit_price * shares
            else:
                exit_price = final_price * (1 + self.slippage)
                pnl = (entry_price - exit_price) * shares - abs(exit_price * shares * self.commission)
                cash += pnl + (entry_price * shares)

            trades.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': self.data.iloc[-1]['datetime'],
                'exit_price': exit_price,
                'type': 'LONG' if position == 1 else 'SHORT',
                'exit_reason': 'CLOSE',
                'pnl': pnl,
                'pnl_pct': ((exit_price / entry_price - 1) if position == 1 else (entry_price / exit_price - 1)) * 100,
                'shares': shares
            })

        return trades, equity_curve

    def analyze_results(self, trades, equity_curve):
        """分析回测结果"""
        if not trades:
            print("[!] 没有交易记录!")
            return

        trades_df = pd.DataFrame(trades)

        # 基础统计
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')

        final_equity = equity_curve[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        # 最大回撤
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        # 输出结果
        print("\n" + "="*60)
        print("[SUCCESS] 回测完成!")
        print("="*60)
        print(f"初始资金:      ${self.initial_capital:,.2f}")
        print(f"最终权益:      ${final_equity:,.2f}")
        print(f"总收益:        ${total_pnl:,.2f} ({total_return:.2f}%)")
        print(f"最大回撤:      {max_drawdown:.2f}%")
        print("="*60)
        print(f"总交易次数:    {total_trades}")
        print(f"盈利次数:      {winning_trades}")
        print(f"亏损次数:      {losing_trades}")
        print(f"胜率:          {win_rate:.2f}%")
        print(f"盈亏比:        {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "盈亏比:        N/A")
        print(f"利润因子:      {profit_factor:.2f}")
        print("="*60)
        print(f"平均盈利:      ${avg_win:,.2f}")
        print(f"平均亏损:      ${avg_loss:,.2f}")
        print(f"平均交易盈亏:  ${avg_pnl:,.2f}")
        print("="*60)

        # 保存详细交易记录
        trades_df.to_csv('backtest_trades.csv', index=False)
        print(f"\n[+] 详细交易记录已保存到: backtest_trades.csv")

        # 显示最近10笔交易
        print(f"\n最近10笔交易:")
        print(trades_df.tail(10).to_string())

        return trades_df

if __name__ == "__main__":
    # 加载数据
    print("[*] 加载数据...")
    df = pd.read_csv('LINKUSDT_8h_2020-01-01_to_2025-11-09.csv')
    print(f"[+] 数据加载成功! 共 {len(df)} 根K线\n")

    # 创建策略实例
    strategy = VolumeSupertrendSessionsStrategy(
        data=df,
        atr_period=10,
        st_factor=2.5,
        vol_length=50,
        vol_multiplier=1.0,
        require_vol_confirm=False,
        min_vol_ratio=1.3,
        only_strong_signals=False,
        use_session_filter=True,
        stop_loss_type='ATR',
        stop_loss_atr=1.8,
        take_profit_rr=2.0
    )

    # 执行回测
    trades, equity_curve = strategy.backtest()

    # 分析结果
    strategy.analyze_results(trades, equity_curve)
