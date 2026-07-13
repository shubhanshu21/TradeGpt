"""Turns a trades DataFrame (as produced by BacktestEngine.run) into the
numbers that actually matter for deciding whether a strategy is worth
paper trading: total return, win rate, profit factor, drawdown, Sharpe.
"""
import numpy as np
import pandas as pd


def compute_metrics(trades: pd.DataFrame, initial_capital: float) -> dict:
    if trades.empty:
        return {
            "total_trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0,
            "total_net_pnl": 0.0, "return_pct": 0.0, "avg_pnl_per_trade": 0.0,
            "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "largest_win": 0.0, "largest_loss": 0.0,
        }

    trades = trades.sort_values("exit_time").reset_index(drop=True)
    wins = trades[trades["net_pnl"] > 0]["net_pnl"]
    losses = trades[trades["net_pnl"] <= 0]["net_pnl"]

    total_net_pnl = trades["net_pnl"].sum()
    win_rate = len(wins) / len(trades) * 100
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf")

    equity_curve = initial_capital + trades["net_pnl"].cumsum()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100

    daily_pnl = trades.groupby("day")["net_pnl"].sum()
    daily_returns = daily_pnl / initial_capital
    sharpe = 0.0
    if daily_returns.std(ddof=0) > 0:
        sharpe = (daily_returns.mean() / daily_returns.std(ddof=0)) * np.sqrt(252)

    return {
        "total_trades": len(trades),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else float("inf"),
        "total_net_pnl": round(total_net_pnl, 2),
        "return_pct": round(total_net_pnl / initial_capital * 100, 2),
        "avg_pnl_per_trade": round(trades["net_pnl"].mean(), 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "avg_win": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss": round(losses.mean(), 2) if len(losses) else 0.0,
        "largest_win": round(wins.max(), 2) if len(wins) else 0.0,
        "largest_loss": round(losses.min(), 2) if len(losses) else 0.0,
    }
