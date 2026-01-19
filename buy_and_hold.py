import pandas as pd
from math import sqrt
from Portfolio import Portfolio,max_drawdown

class BuyAndHold:
    def __init__(self, starting_cash: float):
        self.starting_cash = starting_cash

    # Buy-and-hold underlying QQQ comparison (uses fractional shares for simplicity)
    def calculate_metrics(self, data: pd.DataFrame, rf_daily: pd.Series) -> dict:
        close_on_eq_index = data["Close"].reindex(rf_daily.index).ffill()
        first_close = close_on_eq_index.iloc[0] if len(close_on_eq_index) > 0 else float("nan")

        if pd.notna(first_close) and first_close > 0:
            bh_shares = float(self.starting_cash) / float(first_close)
            buy_and_hold = close_on_eq_index * bh_shares
        else:
            buy_and_hold = pd.Series(0.0, index=rf_daily.index, name="BuyAndHold")

        if len(buy_and_hold) > 0 and buy_and_hold.dropna().sum() > 0:
            bh_start = float(buy_and_hold.iloc[0])
            bh_end = float(buy_and_hold.iloc[-1])
            years = (buy_and_hold.index[-1] - buy_and_hold.index[0]).days / 365.25
            bh_cagr = (bh_end / bh_start) ** (1 / years) - 1 if years > 0 else float("nan")
            bh_mdd = max_drawdown(buy_and_hold)
            bh_daily_ret = buy_and_hold.pct_change()
            bh_daily_excess = (bh_daily_ret.reindex(rf_daily.index) - rf_daily).dropna()
            bh_sharpe = float(bh_daily_excess.mean() / bh_daily_excess.std() * sqrt(252)) if len(
                bh_daily_excess) > 1 and bh_daily_excess.std() > 0 else float("nan")
        else:
            bh_end = float("nan")
            bh_cagr = float("nan")
            bh_mdd = float("nan")
            bh_sharpe = float("nan")

        return buy_and_hold, bh_end, bh_cagr, bh_mdd, bh_sharpe

