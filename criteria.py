import math

import pandas as pd
from Portfolio import Portfolio
from data_classes import OptionTrade
from dataclasses import dataclass, field


class EntryExit:
    def __init__(self, name):
        self.name = name
        self.entry_rsi_low: float = 30.0
        self.entry_rsi_high: float = 50.0
        self.hold_days: int = 180
        self.profit_take: float = 0.50
        self.portfolio: Portfolio = Portfolio()
        self.report: List[Dict[str, object]] = []

    def setRsiHoldProfit(self, entry_rsi_low: float = 30.0, entry_rsi_high: float = 50.0, hold_days: int = 180, profit_take: float = 0.50):
        self.entry_rsi_low = entry_rsi_low
        self.entry_rsi_high = entry_rsi_high
        self.hold_days = hold_days
        self.profit_take = profit_take

    def check_entry_conditions(self, d, end_ts, rsi: float, sigma: float) -> bool:
        """
        Check if the entry criteria are met.
        :param d: Current date.
        :param end_ts: End timestamp of the backtest.
        :param rsi: RSI value.
        :param sigma: Volatility.
        :return: True if entry criteria are met, False otherwise.
        """
        # Ensure we can hold 180 days within the backtest window
        if d + pd.Timedelta(days=self.hold_days) > end_ts:
            return False

        if pd.isna(rsi) or not (self.entry_rsi_low <= float(rsi) <= self.entry_rsi_high):
            return False

        if not (math.isfinite(sigma) and sigma > 0):
            return False
        return True

    def check_exit_conditions(self, holding: int, px: float, open_trade: OptionTrade) -> tuple[bool, bool]:
        """
        Check if the exit criteria are met.
        :param holding: Number of holding days.
        :param px: Current price.
        :param open_trade: The open trade to evaluate.
        :return: True if exit criteria are met, False otherwise.
        """
        hit_profit = px >= open_trade.target_price
        hit_time = holding >= self.hold_days
        return hit_profit, hit_profit or hit_time
