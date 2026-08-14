# Backtrade 📈

A fast, flexible backtesting framework for cryptocurrency trading strategies.

## 🎯 Features

- **Multi-exchange support** — Binance, BYBIT, OKX historical data
- **Strategy library** — Built-in: long_wick, engulfing, cross, gap, l1_l2 quality
- **Performance metrics** — Sharpe ratio, max drawdown, win rate, profit factor
- **Visualization** — Trade charts with entry/exit points
- **Optimization** — Parameter grid search for strategy tuning

## 🚀 Quick Start

```python
from backtrade import Backtester, Strategy

class MyStrategy(Strategy):
    def on_candle(self, candle):
        if candle.close > candle.open * 1.02:
            self.buy(amount=0.01)

bt = Backtester(symbol='BTC/USDT', timeframe='1h')
bt.run(MyStrategy(), start='2024-01-01', end='2024-12-31')
bt.report()
```

## 📊 Example Output

```
═══════════════════════════════════
Backtest Results
═══════════════════════════════════
Total Return:    +15.23%
Max Drawdown:   -8.45%
Sharpe Ratio:   1.87
Win Rate:       62.5%
Total Trades:   48
═══════════════════════════════════
```

## 📝 License

MIT
