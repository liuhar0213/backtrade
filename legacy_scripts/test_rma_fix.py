#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试使用RMA的修正版本是否能匹配TradingView
"""

import pandas as pd
import sys
sys.path.insert(0, 'strategies')
from supertrend_strategy_fixed import SupertrendStrategyFixed
from supertrend_strategy import SupertrendStrategy

def main():
    # 读取数据
    df = pd.read_csv('data/BINANCE_ETHUSDT.P, 60.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    # 使用旧版本（SMA）
    strategy_old = SupertrendStrategy(atr_period=14, factor=3.0)
    df_old = strategy_old.calculate_supertrend(df.copy())

    # 使用新版本（RMA）
    strategy_new = SupertrendStrategyFixed(atr_period=14, factor=3.0)
    df_new = strategy_new.calculate_supertrend(df.copy())

    # 对比12-05 15:00的数值
    target_time = pd.Timestamp('2023-12-05 15:00:00', tz=df.index.tz)

    print("\n" + "="*120)
    print("12-05 15:00 数值对比")
    print("="*120)

    row_old = df_old.loc[target_time]
    row_new = df_new.loc[target_time]

    print(f"\n{'项目':<20} {'旧版(SMA)':<15} {'新版(RMA)':<15} {'TradingView':<15} {'差异(RMA vs TV)'}")
    print("-"*120)

    print(f"{'收盘价':<20} ${row_old['close']:<14.2f} ${row_new['close']:<14.2f} ${'2200.19':<14} -")
    print(f"{'ATR':<20} ${row_old['atr']:<14.2f} ${row_new['atr']:<14.2f} ${'??':<14} ??")
    print(f"{'下轨 (Final LB)':<20} ${row_old['final_lb']:<14.2f} ${row_new['final_lb']:<14.2f} ${'~2204':<14} ${row_new['final_lb'] - 2204:.2f}")
    print(f"{'上轨 (Final UB)':<20} ${row_old['final_ub']:<14.2f} ${row_new['final_ub']:<14.2f} ${'??':<14} ??")

    direction_old_str = "上升" if row_old['direction'] == 1 else "下降"
    direction_new_str = "上升" if row_new['direction'] == 1 else "下降"
    print(f"{'方向':<20} {direction_old_str:<15} {direction_new_str:<15} {'下降':<15} {'-' if direction_new_str == '下降' else '不匹配！'}")

    print("\n" + "="*120)
    print("结论")
    print("="*120)

    if abs(row_new['final_lb'] - 2204) < 1.0:
        print(f"\n✓ 使用RMA后，final_lb = ${row_new['final_lb']:.2f}，接近TradingView的 ~$2204！")
        print(f"  差异只有 ${abs(row_new['final_lb'] - 2204):.2f}")
        print(f"\n  问题解决！ATR应该使用RMA（Wilder's Smoothing）而不是SMA！")
    else:
        print(f"\n✗ 使用RMA后，final_lb = ${row_new['final_lb']:.2f}，仍与TradingView的 ~$2204 不匹配")
        print(f"  差异: ${row_new['final_lb'] - 2204:.2f}")
        print(f"\n  可能还有其他问题...")

    # 对比12-01到12-05的反转点
    print("\n" + "="*120)
    print("对比12-01到12-05期间的方向")
    print("="*120)

    start_time = pd.Timestamp('2023-12-01 09:00:00', tz=df.index.tz)
    end_time = pd.Timestamp('2023-12-05 15:00:00', tz=df.index.tz)

    mask = (df_new.index >= start_time) & (df_new.index <= end_time)

    # 统计方向
    period_new = df_new[mask]
    up_count = len(period_new[period_new['direction'] == 1])
    down_count = len(period_new[period_new['direction'] == -1])

    print(f"\n使用RMA的新版本:")
    print(f"  上升趋势: {up_count} 根")
    print(f"  下降趋势: {down_count} 根")

    # 找出所有反转点
    reversals_new = []
    for i in range(1, len(period_new)):
        curr_t = period_new.index[i]
        prev_t = period_new.index[i-1]

        if period_new.loc[prev_t, 'direction'] != period_new.loc[curr_t, 'direction']:
            from_dir = "上升" if period_new.loc[prev_t, 'direction'] == 1 else "下降"
            to_dir = "上升" if period_new.loc[curr_t, 'direction'] == 1 else "下降"
            reversals_new.append((curr_t, from_dir, to_dir))

    if reversals_new:
        print(f"\n  发现 {len(reversals_new)} 个反转点:")
        for t, from_d, to_d in reversals_new:
            print(f"    {t}: {from_d} → {to_d}")
    else:
        print(f"\n  没有反转点（一直保持同一方向）")

    print(f"\nTradingView显示:")
    print(f"  12-05之前: 下降趋势")
    print(f"  12-05 15:00: 反转为上升趋势")

    if down_count > 0:
        print(f"\n✓ 使用RMA后，出现了下降趋势！可能匹配TradingView了！")
    else:
        print(f"\n✗ 使用RMA后，仍然全是上升趋势，不匹配TradingView")

if __name__ == '__main__':
    main()
