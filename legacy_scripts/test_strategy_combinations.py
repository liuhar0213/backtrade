"""
策略组合对比测试

目的：
展示不同入场+出场策略组合的回测效果，验证Phase 3.6策略的实际价值

对比策略：
1. 原有系统 (bias_threshold + fixed_sl_tp)
2. Phase 3.6智能组合 (phase36_resonance + phase36_exhaustion)
3. 经典技术分析 (macd_rsi_combo + atr_trailing)
4. 突破跟随 (bb_breakout + opposite_signal)
5. 均线交叉 (ema_crossover + time_based)

Author: Phase 3.6+ Team
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


class StrategyTester:
    """策略组合测试器"""

    def __init__(self, data_path: str):
        """
        初始化测试器

        Args:
            data_path: BTC 15分钟数据路径
        """
        self.data_path = data_path
        self.data = None

        # 初始化Phase 3.6模块
        self.phase_detector = create_trend_phase_detector()
        self.retrace_analyzer = create_retrace_complexity_analyzer()
        self.resonance_engine = create_multi_timeframe_resonance_engine()

    def load_data(self):
        """加载数据"""
        print(f"[1/6] 加载数据: {self.data_path}")
        self.data = pd.read_csv(self.data_path)

        # 标准化列名
        if 'time' in self.data.columns:
            self.data = self.data.rename(columns={'time': 'timestamp'})
        if 'Volume' in self.data.columns:
            self.data = self.data.rename(columns={'Volume': 'volume'})

        # 添加volume列（如果缺失）
        if 'volume' not in self.data.columns:
            self.data['volume'] = 1000000  # 模拟volume

        # 计算基础指标
        self._compute_indicators()

        print(f"  数据点数: {len(self.data)}")
        if 'timestamp' in self.data.columns:
            print(f"  时间范围: {self.data['timestamp'].iloc[0]} 至 {self.data['timestamp'].iloc[-1]}")

    def _compute_indicators(self):
        """计算技术指标"""
        print("[2/6] 计算技术指标...")

        # 确保数据列存在
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"缺少必要列: {col}")

        # 简化的技术指标计算
        close = self.data['close'].values

        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean()
        ema26 = pd.Series(close).ewm(span=26).mean()
        self.data['macd'] = ema12 - ema26
        self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
        self.data['macd_hist'] = self.data['macd'] - self.data['macd_signal']

        # RSI
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        self.data['bb_middle'] = pd.Series(close).rolling(window=20).mean()
        bb_std = pd.Series(close).rolling(window=20).std()
        self.data['bb_upper'] = self.data['bb_middle'] + (bb_std * 2)
        self.data['bb_lower'] = self.data['bb_middle'] - (bb_std * 2)

        # EMA
        self.data['ema_fast'] = pd.Series(close).ewm(span=12).mean()
        self.data['ema_slow'] = pd.Series(close).ewm(span=26).mean()

        # ATR
        high = self.data['high'].values
        low = self.data['low'].values
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        self.data['atr'] = pd.Series(tr).rolling(window=14).mean()

        # 成交量均线
        self.data['avg_volume'] = self.data['volume'].rolling(window=20).mean()

        # 综合偏向度 (bias)
        self.data['bias'] = (
            0.3 * (self.data['macd'] / self.data['close']) +
            0.2 * ((self.data['rsi'] - 50) / 50) +
            0.2 * ((self.data['close'] - self.data['bb_middle']) / self.data['bb_middle']) +
            0.15 * ((self.data['ema_fast'] - self.data['ema_slow']) / self.data['close']) +
            0.15 * ((self.data['volume'] - self.data['avg_volume']) / self.data['avg_volume'])
        )

        print(f"  指标计算完成")

    def compute_phase36_signals(self):
        """计算Phase 3.6信号"""
        print("[3/6] 计算Phase 3.6市场结构信号...")

        phase36_signals = []

        for i in range(100, len(self.data)):
            # 获取窗口数据
            window_data = self.data.iloc[i-100:i]

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

            # 简化的多时间框架共振（使用当前数据模拟）
            # 实际应该从多个时间框架获取数据
            resonance_score = 1.0 - retrace_result['complexity_score']  # 简化计算
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
            self.data[f'phase36_{key}'] = [s[key] for s in phase36_signals]

        print(f"  Phase 3.6信号计算完成")

    def backtest_strategy(self, strategy_name: str, entry_logic, exit_logic) -> Dict:
        """
        回测策略组合

        Args:
            strategy_name: 策略名称
            entry_logic: 入场逻辑函数
            exit_logic: 出场逻辑函数

        Returns:
            回测结果
        """
        capital = 10000
        position = None
        trades = []

        for i in range(100, len(self.data)):
            current_bar = self.data.iloc[i]

            # 检查出场
            if position is not None:
                should_exit = exit_logic(position, current_bar, i)

                if should_exit:
                    # 平仓
                    exit_price = current_bar['close']
                    pnl = (exit_price - position['entry_price']) * position['direction']

                    # 扣除手续费（开仓+平仓，使用Taker费率 0.033%）
                    entry_cost = position['entry_price'] * 0.00033  # 开仓手续费
                    exit_cost = exit_price * 0.00033  # 平仓手续费
                    total_fees = entry_cost + exit_cost
                    pnl = pnl - total_fees  # 扣除手续费后的净利润

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
                signal = entry_logic(current_bar, i)

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
        if len(trades) == 0:
            return {
                'strategy': strategy_name,
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'avg_trade': 0.0
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
            'strategy': strategy_name,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'avg_trade': np.mean(returns),
            'final_capital': capital
        }

    def run_all_tests(self):
        """运行所有策略对比测试"""
        print("\n" + "=" * 80)
        print("策略组合对比测试")
        print("=" * 80)

        results = []

        # 策略1: 原有系统 (bias_threshold + fixed_sl_tp)
        print("\n[4/6] 测试策略1: 原有系统 (bias_threshold + fixed_sl_tp)")
        result1 = self.backtest_strategy(
            "Original System",
            entry_logic=self._entry_bias_threshold,
            exit_logic=self._exit_fixed_sl_tp
        )
        results.append(result1)
        self._print_result(result1)

        # 策略2: Phase 3.6智能组合
        print("\n[4/6] 测试策略2: Phase 3.6智能组合 (resonance + exhaustion)")
        result2 = self.backtest_strategy(
            "Phase 3.6 Smart",
            entry_logic=self._entry_phase36_resonance,
            exit_logic=self._exit_phase36_exhaustion
        )
        results.append(result2)
        self._print_result(result2)

        # 策略3: MACD + RSI组合
        print("\n[5/6] 测试策略3: 经典技术分析 (macd_rsi + atr_trailing)")
        result3 = self.backtest_strategy(
            "MACD+RSI Classic",
            entry_logic=self._entry_macd_rsi,
            exit_logic=self._exit_atr_trailing
        )
        results.append(result3)
        self._print_result(result3)

        # 策略4: 布林带突破
        print("\n[5/6] 测试策略4: 突破跟随 (bb_breakout + opposite_signal)")
        result4 = self.backtest_strategy(
            "BB Breakout",
            entry_logic=self._entry_bb_breakout,
            exit_logic=self._exit_opposite_signal
        )
        results.append(result4)
        self._print_result(result4)

        # 策略5: EMA均线交叉
        print("\n[6/6] 测试策略5: 均线交叉 (ema_crossover + time_based)")
        result5 = self.backtest_strategy(
            "EMA Crossover",
            entry_logic=self._entry_ema_crossover,
            exit_logic=self._exit_time_based
        )
        results.append(result5)
        self._print_result(result5)

        # 生成对比报告
        self._generate_comparison_report(results)

    def _print_result(self, result: Dict):
        """打印单个策略结果"""
        print(f"  交易次数: {result['total_trades']}")
        print(f"  胜率: {result['win_rate']*100:.1f}%")
        print(f"  总收益: {result['total_return']*100:+.2f}%")
        print(f"  Sharpe: {result['sharpe']:.3f}")
        print(f"  最大回撤: {result['max_drawdown']*100:.2f}%")

    def _generate_comparison_report(self, results: List[Dict]):
        """生成策略对比报告"""
        print("\n" + "=" * 80)
        print("策略对比报告")
        print("=" * 80)

        # 按Sharpe排序
        sorted_results = sorted(results, key=lambda x: x['sharpe'], reverse=True)

        print("\n排名 (按Sharpe比率):")
        print("-" * 80)
        print(f"{'排名':<6} {'策略名称':<25} {'交易数':<10} {'胜率':<10} {'收益率':<12} {'Sharpe':<10}")
        print("-" * 80)

        for rank, result in enumerate(sorted_results, 1):
            marker = " [BEST]" if rank == 1 else ""
            print(f"{rank:<6} {result['strategy']:<25} {result['total_trades']:<10} "
                  f"{result['win_rate']*100:>6.1f}%   {result['total_return']*100:>+8.2f}%   "
                  f"{result['sharpe']:>8.3f}{marker}")

        print("-" * 80)

        # 关键发现
        best = sorted_results[0]
        print(f"\n[关键发现]")
        print(f"最佳策略: {best['strategy']}")
        print(f"  Sharpe比率: {best['sharpe']:.3f}")
        print(f"  年化收益: {best['total_return']*100:+.2f}%")
        print(f"  风险调整后收益最优")

        # Phase 3.6表现
        phase36_result = next((r for r in results if 'Phase 3.6' in r['strategy']), None)
        if phase36_result:
            rank = sorted_results.index(phase36_result) + 1
            print(f"\nPhase 3.6智能策略排名: #{rank}")
            if rank == 1:
                print("  Phase 3.6策略表现最佳！")
            elif rank <= 2:
                print("  Phase 3.6策略表现优秀")
            else:
                print("  Phase 3.6策略有改进空间")

        print("=" * 80)

    # ===== 入场逻辑实现 =====

    def _entry_bias_threshold(self, bar, i) -> str:
        """原有的bias_threshold逻辑"""
        bias = bar['bias']
        if np.isnan(bias):
            return "NEUTRAL"

        if bias > 0.4:
            return "LONG"
        elif bias < -0.4:
            return "SHORT"
        return "NEUTRAL"

    def _entry_phase36_resonance(self, bar, i) -> str:
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

    def _entry_macd_rsi(self, bar, i) -> str:
        """MACD + RSI组合逻辑"""
        macd = bar['macd']
        macd_signal = bar['macd_signal']
        rsi = bar['rsi']

        if np.isnan(macd) or np.isnan(rsi):
            return "NEUTRAL"

        # 做多：MACD金叉 + RSI < 40
        if macd > macd_signal and rsi < 40:
            return "LONG"

        # 做空：MACD死叉 + RSI > 60
        if macd < macd_signal and rsi > 60:
            return "SHORT"

        return "NEUTRAL"

    def _entry_bb_breakout(self, bar, i) -> str:
        """布林带突破逻辑"""
        close = bar['close']
        bb_upper = bar['bb_upper']
        bb_lower = bar['bb_lower']
        volume = bar['volume']
        avg_volume = bar['avg_volume']

        if np.isnan(bb_upper) or np.isnan(volume):
            return "NEUTRAL"

        # 向上突破 + 成交量放大
        if close > bb_upper and volume > avg_volume * 1.5:
            return "LONG"

        # 向下突破 + 成交量放大
        if close < bb_lower and volume > avg_volume * 1.5:
            return "SHORT"

        return "NEUTRAL"

    def _entry_ema_crossover(self, bar, i) -> str:
        """EMA均线交叉逻辑"""
        ema_fast = bar['ema_fast']
        ema_slow = bar['ema_slow']

        if i < 1:
            return "NEUTRAL"

        prev_bar = self.data.iloc[i-1]
        ema_fast_prev = prev_bar['ema_fast']
        ema_slow_prev = prev_bar['ema_slow']

        if np.isnan(ema_fast) or np.isnan(ema_slow):
            return "NEUTRAL"

        # 金叉
        if ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow:
            return "LONG"

        # 死叉
        if ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow:
            return "SHORT"

        return "NEUTRAL"

    # ===== 出场逻辑实现 =====

    def _exit_fixed_sl_tp(self, position, bar, i) -> bool:
        """固定止损止盈"""
        entry_price = position['entry_price']
        current_price = bar['close']
        direction = position['direction']

        if direction == 1:  # 做多
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # 做空
            pnl_pct = (entry_price - current_price) / entry_price

        # 止损2% 或 止盈4%
        if pnl_pct <= -0.02 or pnl_pct >= 0.04:
            return True

        return False

    def _exit_phase36_exhaustion(self, position, bar, i) -> bool:
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

        if pnl_pct <= -0.03:  # 3%止损
            return True

        return False

    def _exit_atr_trailing(self, position, bar, i) -> bool:
        """ATR追踪止损"""
        entry_price = position['entry_price']
        current_price = bar['close']
        atr = bar['atr']
        direction = position['direction']

        if direction == 1:
            pnl_pct = (current_price - entry_price) / entry_price
            # 盈利>2%后启动追踪
            if pnl_pct > 0.02:
                trailing_stop = position['highest_price'] - atr * 2.5
                if current_price < trailing_stop:
                    return True
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            if pnl_pct > 0.02:
                trailing_stop = position['lowest_price'] + atr * 2.5
                if current_price > trailing_stop:
                    return True

        # 基本止损
        if pnl_pct <= -0.02:
            return True

        return False

    def _exit_opposite_signal(self, position, bar, i) -> bool:
        """反向信号出场"""
        entry_direction = 'LONG' if position['direction'] == 1 else 'SHORT'
        current_signal = self._entry_bias_threshold(bar, i)

        # 持有多单时出现空头信号
        if entry_direction == 'LONG' and current_signal == 'SHORT':
            return True

        # 持有空单时出现多头信号
        if entry_direction == 'SHORT' and current_signal == 'LONG':
            return True

        # 基本止损
        entry_price = position['entry_price']
        current_price = bar['close']
        direction = position['direction']

        if direction == 1:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        if pnl_pct <= -0.025:
            return True

        return False

    def _exit_time_based(self, position, bar, i) -> bool:
        """时间止损"""
        holding_bars = i - position['entry_bar']

        # 持仓超过50周期
        if holding_bars > 50:
            return True

        # 基本止损
        entry_price = position['entry_price']
        current_price = bar['close']
        direction = position['direction']

        if direction == 1:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        if pnl_pct <= -0.02:
            return True

        return False


def main():
    """主测试入口"""
    # 数据路径
    data_path = "data/BINANCE_BTCUSDT.P, 15.csv"  # 注意有空格

    # 创建测试器
    tester = StrategyTester(data_path)

    # 加载数据
    tester.load_data()

    # 计算Phase 3.6信号
    tester.compute_phase36_signals()

    # 运行所有测试
    tester.run_all_tests()

    print("\n[完成] 策略组合对比测试已完成")


if __name__ == "__main__":
    main()
