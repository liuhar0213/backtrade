"""
多币种验证测试

目的：
验证Phase 3.6策略在多个币种上的表现，并与原有系统对比

测试币种：
1. BTC (比特币)
2. ETH (以太坊)
3. SOL (Solana)
4. BNB (币安币)
5. ADA (卡尔达诺)

手续费设置：
- Maker: 0.012% (币安40%返佣后)
- Taker: 0.033% (币安40%返佣后)

Author: Backtrade Team
Date: 2025-10-30
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入Phase 3.6模块
from F_intelligence.trend_phase_detector import create_trend_phase_detector
from F_intelligence.retrace_complexity_analyzer import create_retrace_complexity_analyzer
from F_intelligence.multi_timeframe_resonance import create_multi_timeframe_resonance_engine


class MultiSymbolValidator:
    """多币种验证器"""

    def __init__(self):
        """初始化验证器"""
        # 币种配置
        self.symbols = {
            'BTC': 'data/BINANCE_BTCUSDT.P, 15.csv',
            'ETH': 'data/BINANCE_ETHUSDT.P, 15.csv',
            'SOL': 'data/BINANCE_SOLUSDT.P, 15.csv',
            'BNB': 'data/BINANCE_BNBUSDT.P, 15.csv',
            'ADA': 'data/BINANCE_ADAUSDT.P, 15.csv'
        }

        # 手续费设置（币安40%返佣后）
        self.maker_fee = 0.00012  # 0.012%
        self.taker_fee = 0.00033  # 0.033%
        self.slippage = 0.0005    # 0.05%

        # 初始化Phase 3.6模块
        self.phase_detector = create_trend_phase_detector()
        self.retrace_analyzer = create_retrace_complexity_analyzer()
        self.resonance_engine = create_multi_timeframe_resonance_engine()

        # 存储结果
        self.results = []

    def load_and_prepare_data(self, symbol: str, data_path: str) -> pd.DataFrame:
        """
        加载并准备数据

        Args:
            symbol: 币种名称
            data_path: 数据路径

        Returns:
            准备好的数据
        """
        print(f"\n加载 {symbol} 数据: {data_path}")

        # 加载数据
        data = pd.read_csv(data_path)

        # 标准化列名
        if 'time' in data.columns:
            data = data.rename(columns={'time': 'timestamp'})
        if 'Volume' in data.columns:
            data = data.rename(columns={'Volume': 'volume'})

        # 添加volume列（如果缺失）
        if 'volume' not in data.columns:
            data['volume'] = 1000000  # 模拟volume

        # 检查必要的列
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"{symbol} 缺少必要列: {col}")

        print(f"  数据点数: {len(data)}")
        if 'timestamp' in data.columns:
            print(f"  时间范围: {data['timestamp'].iloc[0]} 至 {data['timestamp'].iloc[-1]}")

        # 计算技术指标
        self._compute_indicators(data)

        # 计算Phase 3.6信号
        self._compute_phase36_signals(data)

        return data

    def _compute_indicators(self, data: pd.DataFrame):
        """计算技术指标"""
        close = data['close'].values

        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean()
        ema26 = pd.Series(close).ewm(span=26).mean()
        data['macd'] = ema12 - ema26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']

        # RSI
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        data['bb_middle'] = pd.Series(close).rolling(window=20).mean()
        bb_std = pd.Series(close).rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)

        # EMA
        data['ema_fast'] = pd.Series(close).ewm(span=12).mean()
        data['ema_slow'] = pd.Series(close).ewm(span=26).mean()

        # ATR
        high = data['high'].values
        low = data['low'].values
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        data['atr'] = pd.Series(tr).rolling(window=14).mean()

        # 成交量均线
        data['avg_volume'] = data['volume'].rolling(window=20).mean()

        # 综合偏向度 (bias)
        data['bias'] = (
            0.3 * (data['macd'] / data['close']) +
            0.2 * ((data['rsi'] - 50) / 50) +
            0.2 * ((data['close'] - data['bb_middle']) / data['bb_middle']) +
            0.15 * ((data['ema_fast'] - data['ema_slow']) / data['close']) +
            0.15 * ((data['volume'] - data['avg_volume']) / data['avg_volume'])
        )

    def _compute_phase36_signals(self, data: pd.DataFrame):
        """计算Phase 3.6信号"""
        phase36_signals = []

        for i in range(100, len(data)):
            # 获取窗口数据
            window_data = data.iloc[i-100:i]

            prices = window_data['close'].values
            volume = window_data['volume'].values
            macd_hist = window_data['macd_hist'].values
            macd_line = window_data['macd'].values

            # 趋势阶段检测
            phase_result = self.phase_detector.detect_phase(
                prices, volume, macd_hist, macd_line
            )

            # 回调复杂度分析
            retrace_result = self.retrace_analyzer.analyze_retrace(prices)

            # 简化的多时间框架共振
            resonance_score = 1.0 - retrace_result['complexity_score']
            if phase_result['phase'] == 'startup':
                resonance_score *= 1.2
            elif phase_result['phase'] == 'exhaustion':
                resonance_score *= 0.8

            resonance_score = np.clip(resonance_score, 0, 1)

            phase36_signals.append({
                'trend_phase': phase_result['phase'],
                'phase_score': phase_result['phase_score'],
                'complexity_score': retrace_result['complexity_score'],
                'resonance_score': resonance_score,
                'macd_divergence': phase_result.get('macd_divergence', False)
            })

        # 补齐前100个数据点
        for _ in range(100):
            phase36_signals.insert(0, {
                'trend_phase': 'neutral',
                'phase_score': 0.0,
                'complexity_score': 0.5,
                'resonance_score': 0.0,
                'macd_divergence': False
            })

        # 添加到数据框
        for key in phase36_signals[0].keys():
            data[f'phase36_{key}'] = [s[key] for s in phase36_signals]

    def backtest_original_system(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        回测原有系统

        Args:
            symbol: 币种名称
            data: 数据

        Returns:
            回测结果
        """
        print(f"\n回测原有系统 ({symbol})...")

        capital = 10000
        position = None
        trades = []

        for i in range(100, len(data)):
            current_bar = data.iloc[i]

            # 检查出场
            if position is not None:
                should_exit = self._exit_fixed_sl_tp(position, current_bar)

                if should_exit:
                    # 平仓
                    exit_price = current_bar['close']
                    pnl = (exit_price - position['entry_price']) * position['direction']

                    # 扣除手续费（开仓+平仓，使用Taker费率）
                    entry_cost = position['entry_price'] * self.taker_fee
                    exit_cost = exit_price * self.taker_fee
                    total_fees = entry_cost + exit_cost
                    pnl = pnl - total_fees

                    pnl_pct = pnl / position['entry_price']
                    capital += capital * pnl_pct * 0.1  # 10%仓位

                    trades.append({
                        'entry_bar': position['entry_bar'],
                        'exit_bar': i,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'direction': 'LONG' if position['direction'] == 1 else 'SHORT',
                        'pnl_pct': pnl_pct,
                        'holding_bars': i - position['entry_bar']
                    })

                    position = None

            # 检查入场
            if position is None:
                signal = self._entry_bias_threshold(current_bar)

                if signal in ['LONG', 'SHORT']:
                    position = {
                        'entry_bar': i,
                        'entry_price': current_bar['close'],
                        'direction': 1 if signal == 'LONG' else -1,
                        'highest_price': current_bar['close'],
                        'lowest_price': current_bar['close']
                    }

            # 更新持仓最高/最低价
            if position is not None:
                position['highest_price'] = max(position['highest_price'], current_bar['high'])
                position['lowest_price'] = min(position['lowest_price'], current_bar['low'])

        # 计算指标
        return self._calculate_metrics(symbol, "Original System", trades, capital)

    def backtest_phase36_strategy(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        回测Phase 3.6策略

        Args:
            symbol: 币种名称
            data: 数据

        Returns:
            回测结果
        """
        print(f"\n回测Phase 3.6策略 ({symbol})...")

        capital = 10000
        position = None
        trades = []

        for i in range(100, len(data)):
            current_bar = data.iloc[i]

            # 检查出场
            if position is not None:
                should_exit = self._exit_phase36_exhaustion(position, current_bar)

                if should_exit:
                    # 平仓
                    exit_price = current_bar['close']
                    pnl = (exit_price - position['entry_price']) * position['direction']

                    # 扣除手续费（开仓+平仓，使用Taker费率）
                    entry_cost = position['entry_price'] * self.taker_fee
                    exit_cost = exit_price * self.taker_fee
                    total_fees = entry_cost + exit_cost
                    pnl = pnl - total_fees

                    pnl_pct = pnl / position['entry_price']
                    capital += capital * pnl_pct * 0.1  # 10%仓位

                    trades.append({
                        'entry_bar': position['entry_bar'],
                        'exit_bar': i,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'direction': 'LONG' if position['direction'] == 1 else 'SHORT',
                        'pnl_pct': pnl_pct,
                        'holding_bars': i - position['entry_bar']
                    })

                    position = None

            # 检查入场
            if position is None:
                signal = self._entry_phase36_resonance(current_bar)

                if signal in ['LONG', 'SHORT']:
                    position = {
                        'entry_bar': i,
                        'entry_price': current_bar['close'],
                        'direction': 1 if signal == 'LONG' else -1,
                        'highest_price': current_bar['close'],
                        'lowest_price': current_bar['close']
                    }

            # 更新持仓最高/最低价
            if position is not None:
                position['highest_price'] = max(position['highest_price'], current_bar['high'])
                position['lowest_price'] = min(position['lowest_price'], current_bar['low'])

        # 计算指标
        return self._calculate_metrics(symbol, "Phase 3.6 Smart", trades, capital)

    def _calculate_metrics(self, symbol: str, strategy: str, trades: List[Dict], capital: float) -> Dict:
        """计算回测指标"""
        if len(trades) == 0:
            return {
                'symbol': symbol,
                'strategy': strategy,
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'avg_trade': 0.0,
                'final_capital': capital
            }

        trades_df = pd.DataFrame(trades)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        win_rate = winning_trades / len(trades_df)
        total_return = (capital - 10000) / 10000

        # Sharpe (简化)
        returns = trades_df['pnl_pct'].values
        sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)

        # 最大回撤
        cumulative = (1 + trades_df['pnl_pct']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'symbol': symbol,
            'strategy': strategy,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'avg_trade': np.mean(returns),
            'final_capital': capital
        }

    # ===== 入场逻辑 =====

    def _entry_bias_threshold(self, bar) -> str:
        """原有的bias_threshold逻辑"""
        bias = bar['bias']
        if np.isnan(bias):
            return "NEUTRAL"

        if bias > 0.35:  # 使用优化后的阈值
            return "LONG"
        elif bias < -0.35:
            return "SHORT"
        return "NEUTRAL"

    def _entry_phase36_resonance(self, bar) -> str:
        """Phase 3.6共振逻辑"""
        resonance_score = bar['phase36_resonance_score']
        trend_phase = bar['phase36_trend_phase']
        complexity_score = bar['phase36_complexity_score']

        # 高共振 + 启动期/加速期 + 简单回调
        if (resonance_score > 0.6 and
            trend_phase in ['startup', 'acceleration'] and
            complexity_score < 0.4):

            # 根据其他指标判断方向
            if bar['macd'] > 0 and bar['rsi'] < 70:
                return "LONG"
            elif bar['macd'] < 0 and bar['rsi'] > 30:
                return "SHORT"

        return "NEUTRAL"

    # ===== 出场逻辑 =====

    def _exit_fixed_sl_tp(self, position, bar) -> bool:
        """固定止损止盈"""
        entry_price = position['entry_price']
        current_price = bar['close']
        direction = position['direction']

        if direction == 1:  # 做多
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # 做空
            pnl_pct = (entry_price - current_price) / entry_price

        # 止损1.5% 或 止盈3%（使用优化后的参数）
        if pnl_pct <= -0.015 or pnl_pct >= 0.03:
            return True

        return False

    def _exit_phase36_exhaustion(self, position, bar) -> bool:
        """Phase 3.6衰竭期出场"""
        trend_phase = bar['phase36_trend_phase']
        complexity_score = bar['phase36_complexity_score']
        macd_divergence = bar.get('phase36_macd_divergence', False)

        # 趋势衰竭 + 复杂回调
        if trend_phase == 'exhaustion' and complexity_score > 0.6:
            return True

        # MACD背离
        if macd_divergence:
            return True

        # 基本止损
        entry_price = position['entry_price']
        current_price = bar['close']
        direction = position['direction']

        if direction == 1:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        if pnl_pct <= -0.015:  # 1.5%止损
            return True

        return False

    def run_validation(self):
        """运行多币种验证"""
        print("\n" + "=" * 80)
        print("多币种验证测试")
        print("=" * 80)
        print(f"\n手续费设置:")
        print(f"  Maker: {self.maker_fee * 100:.3f}%")
        print(f"  Taker: {self.taker_fee * 100:.3f}%")
        print(f"  滑点:  {self.slippage * 100:.3f}%")

        # 遍历所有币种
        for symbol, data_path in self.symbols.items():
            print(f"\n{'=' * 80}")
            print(f"测试币种: {symbol}")
            print(f"{'=' * 80}")

            # 加载数据
            data = self.load_and_prepare_data(symbol, data_path)

            # 回测原有系统
            result_original = self.backtest_original_system(symbol, data)
            self.results.append(result_original)
            self._print_result(result_original)

            # 回测Phase 3.6策略
            result_phase36 = self.backtest_phase36_strategy(symbol, data)
            self.results.append(result_phase36)
            self._print_result(result_phase36)

        # 生成汇总报告
        self._generate_summary_report()

    def _print_result(self, result: Dict):
        """打印单个结果"""
        print(f"\n  策略: {result['strategy']}")
        print(f"  交易次数: {result['total_trades']}")
        print(f"  胜率: {result['win_rate']*100:.1f}%")
        print(f"  总收益: {result['total_return']*100:+.2f}%")
        print(f"  Sharpe: {result['sharpe']:.3f}")
        print(f"  最大回撤: {result['max_drawdown']*100:.2f}%")
        print(f"  最终资金: ${result['final_capital']:,.2f}")

    def _generate_summary_report(self):
        """生成汇总报告"""
        print("\n" + "=" * 80)
        print("多币种验证汇总报告")
        print("=" * 80)

        # 转换为DataFrame
        results_df = pd.DataFrame(self.results)

        # 按币种和策略分组
        print("\n详细结果:")
        print("-" * 80)
        print(f"{'币种':<8} {'策略':<20} {'交易数':<8} {'胜率':<10} {'收益率':<12} {'Sharpe':<10}")
        print("-" * 80)

        for _, row in results_df.iterrows():
            print(f"{row['symbol']:<8} {row['strategy']:<20} {row['total_trades']:<8} "
                  f"{row['win_rate']*100:>6.1f}%   {row['total_return']*100:>+8.2f}%   "
                  f"{row['sharpe']:>8.3f}")

        print("-" * 80)

        # 策略对比
        print("\n策略性能对比:")
        print("-" * 80)

        for strategy in ['Original System', 'Phase 3.6 Smart']:
            strategy_results = results_df[results_df['strategy'] == strategy]

            avg_sharpe = strategy_results['sharpe'].mean()
            avg_return = strategy_results['total_return'].mean()
            avg_winrate = strategy_results['win_rate'].mean()
            profitable_symbols = len(strategy_results[strategy_results['total_return'] > 0])

            print(f"\n{strategy}:")
            print(f"  平均Sharpe: {avg_sharpe:.3f}")
            print(f"  平均收益: {avg_return*100:+.2f}%")
            print(f"  平均胜率: {avg_winrate*100:.1f}%")
            print(f"  盈利币种: {profitable_symbols}/{len(self.symbols)}")

        # 关键发现
        print("\n" + "=" * 80)
        print("[关键发现]")

        phase36_results = results_df[results_df['strategy'] == 'Phase 3.6 Smart']
        original_results = results_df[results_df['strategy'] == 'Original System']

        phase36_avg_sharpe = phase36_results['sharpe'].mean()
        original_avg_sharpe = original_results['sharpe'].mean()

        if phase36_avg_sharpe > original_avg_sharpe:
            improvement = (phase36_avg_sharpe - original_avg_sharpe) / abs(original_avg_sharpe) * 100
            print(f"Phase 3.6策略平均Sharpe优于原有系统 {improvement:+.1f}%")
        else:
            decline = (original_avg_sharpe - phase36_avg_sharpe) / abs(original_avg_sharpe) * 100
            print(f"Phase 3.6策略平均Sharpe低于原有系统 {decline:.1f}%")
            print("建议进一步优化Phase 3.6参数")

        # 最佳币种
        best_symbol = results_df.loc[results_df['sharpe'].idxmax()]
        print(f"\n最佳表现:")
        print(f"  币种: {best_symbol['symbol']}")
        print(f"  策略: {best_symbol['strategy']}")
        print(f"  Sharpe: {best_symbol['sharpe']:.3f}")

        print("=" * 80)

        # 保存结果
        results_df.to_csv('results/multi_symbol_validation_results.csv', index=False)
        print(f"\n结果已保存至: results/multi_symbol_validation_results.csv")


def main():
    """主测试入口"""
    # 创建验证器
    validator = MultiSymbolValidator()

    # 运行验证
    validator.run_validation()

    print("\n[完成] 多币种验证测试已完成")


if __name__ == "__main__":
    main()
