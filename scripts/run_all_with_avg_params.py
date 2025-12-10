"""
使用平均优化参数测试所有币种和周期

基于三个最佳策略的平均参数：
- SOLUSDT_30: confidence=0.45, bias=0.20, stop_loss=1.5%, take_profit=4.0%
- BNBUSDT_60: confidence=0.45, bias=0.15, stop_loss=2.0%, take_profit=4.0%
- SOLUSDT_60: confidence=0.40, bias=0.25, stop_loss=2.0%, take_profit=4.0%

平均参数：
- entry_confidence_threshold: 0.433
- entry_bias_threshold: 0.200
- stop_loss_pct: 0.0183 (1.83%)
- take_profit_pct: 0.040 (4.0%)
"""

import pandas as pd
import os
from weighted_feature_strategy import WeightedFeatureStrategy, WeightedFeatureBacktest

# 平均优化参数
AVG_CONFIG = {
    'entry_confidence_threshold': 0.433,
    'entry_bias_threshold': 0.200,
    'stop_loss_pct': 0.0183,
    'take_profit_pct': 0.040,
    'taker_fee': 0.00033
}

# 所有12个数据集
ALL_SYMBOLS = [
    'BTCUSDT_15', 'BTCUSDT_30', 'BTCUSDT_60',
    'ETHUSDT_15', 'ETHUSDT_30', 'ETHUSDT_60',
    'BNBUSDT_15', 'BNBUSDT_30', 'BNBUSDT_60',
    'SOLUSDT_15', 'SOLUSDT_30', 'SOLUSDT_60',
]

def load_data_with_v5(symbol):
    """加载OHLCV + v5特征数据"""
    ohlcv_path = f'data/{symbol}.csv'
    v5_path = f'results/v5_stage_abc/{symbol}/D_hat_stage_abc.csv'

    if not os.path.exists(ohlcv_path) or not os.path.exists(v5_path):
        return None

    try:
        ohlcv = pd.read_csv(ohlcv_path)
        v5 = pd.read_csv(v5_path)

        min_len = min(len(ohlcv), len(v5))
        ohlcv = ohlcv.iloc[:min_len].reset_index(drop=True)
        v5 = v5.iloc[:min_len].reset_index(drop=True)

        ohlc_cols = [c for c in ['open', 'high', 'low', 'close'] if c in ohlcv.columns]
        data = pd.concat([ohlcv[ohlc_cols], v5], axis=1)

        return data
    except Exception as e:
        print(f'  ERROR loading {symbol}: {e}')
        return None

def main():
    print('='*120)
    print('使用平均优化参数测试所有币种和周期')
    print('='*120)
    print()
    print('平均参数配置:')
    print(f"  入场置信度阈值: {AVG_CONFIG['entry_confidence_threshold']:.3f}")
    print(f"  入场偏差阈值: {AVG_CONFIG['entry_bias_threshold']:.3f}")
    print(f"  止损: {AVG_CONFIG['stop_loss_pct']*100:.2f}%")
    print(f"  止盈: {AVG_CONFIG['take_profit_pct']*100:.2f}%")
    print()

    all_results = []
    all_trades = []

    for i, symbol in enumerate(ALL_SYMBOLS, 1):
        print(f'\n{"="*120}')
        print(f'[{i}/{len(ALL_SYMBOLS)}] 回测: {symbol}')
        print(f'{"="*120}')

        # 加载数据
        print(f'加载数据...')
        data = load_data_with_v5(symbol)

        if data is None:
            print(f'  [SKIP] 数据不可用')
            continue

        print(f'  数据行数: {len(data)}')

        # 运行回测
        print(f'运行回测...')
        try:
            strategy = WeightedFeatureStrategy(AVG_CONFIG)
            backtest = WeightedFeatureBacktest(strategy)
            result = backtest.run_backtest(data, symbol)

            stats = result['stats']
            trades = result.get('trades', [])

            # 保存统计结果
            result_summary = {
                'symbol': symbol,
                'coin': symbol.split('_')[0].replace('USDT', ''),
                'timeframe': symbol.split('_')[1],
                'total_trades': stats['total_trades'],
                'win_rate': stats['win_rate'],
                'total_return': stats['total_return'],
                'sharpe': stats['sharpe'],
                'max_drawdown': stats['max_drawdown'],
                'profit_factor': stats['profit_factor'],
                'final_capital': stats['final_capital']
            }
            all_results.append(result_summary)

            print(f'\n回测完成！')
            print(f'  交易次数: {stats["total_trades"]}')
            print(f'  胜率: {stats["win_rate"]*100:.2f}%')
            print(f'  总收益: {stats["total_return"]*100:+.2f}%')
            print(f'  Sharpe: {stats["sharpe"]:.3f}')
            print(f'  最大回撤: {stats["max_drawdown"]*100:.2f}%')

            # 保存交易明细
            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                trades_file = f'results/trades_avg_{symbol}.csv'
                trades_df.to_csv(trades_file, index=False)
                print(f'  交易明细已保存: {trades_file}')

                # 添加到总交易列表
                for trade in trades:
                    trade['symbol'] = symbol
                    all_trades.append(trade)

        except Exception as e:
            print(f'  [ERROR] 回测失败: {e}')
            import traceback
            traceback.print_exc()

    # 保存汇总结果
    if len(all_results) > 0:
        print(f'\n{"="*120}')
        print('保存汇总结果')
        print(f'{"="*120}')

        results_df = pd.DataFrame(all_results)
        summary_file = 'results/avg_params_test_summary.csv'
        results_df.to_csv(summary_file, index=False)
        print(f'\n汇总结果已保存: {summary_file}')

        # 保存所有交易
        if len(all_trades) > 0:
            all_trades_df = pd.DataFrame(all_trades)
            all_trades_file = 'results/avg_params_all_trades.csv'
            all_trades_df.to_csv(all_trades_file, index=False)
            print(f'所有交易明细已保存: {all_trades_file}')
            print(f'总交易笔数: {len(all_trades)}')

        # 生成报告
        print(f'\n{"="*120}')
        print('测试报告')
        print(f'{"="*120}')

        print(f'\n总体统计:')
        print(f'  测试币种数: {len(ALL_SYMBOLS)}')
        print(f'  成功测试: {len(results_df)}')
        print(f'  平均交易次数: {results_df["total_trades"].mean():.0f}')
        print(f'  平均胜率: {results_df["win_rate"].mean()*100:.1f}%')
        print(f'  平均收益: {results_df["total_return"].mean()*100:+.2f}%')
        print(f'  平均Sharpe: {results_df["sharpe"].mean():.3f}')

        # 按币种统计
        print(f'\n按币种统计:')
        print(f'{"币种":<10} {"数据集":<8} {"平均收益":<12} {"平均Sharpe":<12} {"平均胜率":<10}')
        print(f'{"-"*60}')

        for coin in ['BTC', 'ETH', 'BNB', 'SOL']:
            coin_data = results_df[results_df['coin'] == coin]
            if len(coin_data) > 0:
                avg_return = coin_data['total_return'].mean()
                avg_sharpe = coin_data['sharpe'].mean()
                avg_win_rate = coin_data['win_rate'].mean()
                print(f'{coin:<10} {len(coin_data):<8} {avg_return*100:>+10.2f}%  {avg_sharpe:>10.3f}  {avg_win_rate*100:>8.1f}%')

        # 按周期统计
        print(f'\n按周期统计:')
        print(f'{"周期":<10} {"数据集":<8} {"平均收益":<12} {"平均Sharpe":<12} {"平均胜率":<10}')
        print(f'{"-"*60}')

        timeframe_map = {'15': '15分钟', '30': '30分钟', '60': '1小时'}
        for tf in ['15', '30', '60']:
            tf_data = results_df[results_df['timeframe'] == tf]
            if len(tf_data) > 0:
                avg_return = tf_data['total_return'].mean()
                avg_sharpe = tf_data['sharpe'].mean()
                avg_win_rate = tf_data['win_rate'].mean()
                print(f'{timeframe_map[tf]:<10} {len(tf_data):<8} {avg_return*100:>+10.2f}%  {avg_sharpe:>10.3f}  {avg_win_rate*100:>8.1f}%')

        # Top 5 表现
        print(f'\nTop 5 最佳表现 (按Sharpe):')
        print(f'{"排名":<6} {"标的":<15} {"交易数":<10} {"胜率":<10} {"收益":<12} {"Sharpe":<10}')
        print(f'{"-"*70}')

        top5 = results_df.nlargest(5, 'sharpe')
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            print(f'{idx:<6} {row["symbol"]:<15} {row["total_trades"]:<10} '
                  f'{row["win_rate"]*100:>7.1f}%  {row["total_return"]*100:>+9.2f}%  {row["sharpe"]:>8.3f}')

        # Bottom 5 表现
        print(f'\nBottom 5 表现 (按Sharpe):')
        print(f'{"排名":<6} {"标的":<15} {"交易数":<10} {"胜率":<10} {"收益":<12} {"Sharpe":<10}')
        print(f'{"-"*70}')

        bottom5 = results_df.nsmallest(5, 'sharpe')
        for idx, (_, row) in enumerate(bottom5.iterrows(), 1):
            print(f'{idx:<6} {row["symbol"]:<15} {row["total_trades"]:<10} '
                  f'{row["win_rate"]*100:>7.1f}%  {row["total_return"]*100:>+9.2f}%  {row["sharpe"]:>8.3f}')

    print(f'\n{"="*120}')
    print('测试完成！')
    print(f'{"="*120}')
    print('\n生成的文件:')
    print('  1. results/avg_params_test_summary.csv - 汇总统计')
    print('  2. results/avg_params_all_trades.csv - 所有交易明细')
    print('  3. results/trades_avg_[SYMBOL].csv - 各标的交易明细 (12个文件)')

if __name__ == "__main__":
    main()
