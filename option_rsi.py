import pandas as pd
from math import sqrt
from typing import Tuple
from Portfolio import max_drawdown


class OptionRSI:
    @staticmethod
    def calculate_metrics(equity_curve: pd.DataFrame, rf_daily: pd.Series) -> Tuple[float, float, float, float]:
        """
        Calculate strategy metrics based on the equity curve.

        Args:
            equity_curve (pd.DataFrame): DataFrame with 'Equity' column representing the equity curve.
            rf_daily (pd.Series): Daily risk-free rate series.

        Returns:
            Tuple[float, float, float, float]: end equity, CAGR, max drawdown, Sharpe ratio.
        """
        equity_curve = equity_curve.asfreq("B", method="ffill")
        equity = equity_curve["Equity"]
        equity_curve["daily_ret"] = equity.pct_change()

        start_eq = float(equity.iloc[0])
        end_eq = float(equity.iloc[-1])
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        cagr = (end_eq / start_eq) ** (1 / years) - 1 if years > 0 else float("nan")
        mdd = max_drawdown(equity)

        # Sharpe ratio (annualized, excess returns over the backtest's risk-free series)
        daily_ret = equity_curve["daily_ret"].reindex(rf_daily.index)
        daily_excess = (daily_ret - rf_daily).dropna()
        sharpe = float(daily_excess.mean() / daily_excess.std() * sqrt(252)) if len(daily_excess) > 1 and daily_excess.std() > 0 else float("nan")

        return end_eq, cagr, mdd, sharpe
