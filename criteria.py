# BaseEntryExit
#     |
#     +-- OptionEntryExit
#     |
#     +-- UnderlyingEntryExit

import math

import pandas as pd
from data_classes import OptionTrade, OptionLeg, TradePortfolio, UnderlyingTrade, UnderlyingLeg
from typing import List, Dict


class BaseEntryExit:
    def __init__(self, name):
        self.name = name
        self.portfolio: TradePortfolio = TradePortfolio()
        self.report: List[Dict[str, object]] = []


class BuyAndHoldEntryExit(BaseEntryExit):

    @staticmethod
    def check_entry_conditions(d, idx, expiry_mode, S, bs_model, q, r, end_ts, rsi: float, sigma: float, strike_round, target_delta, min_days_out=365) -> [bool, OptionLeg, OptionTrade]:
        # underlying entry rule (for now always true)
        leg = UnderlyingLeg()
        # btcp will set qty and cost_basis
        trade = UnderlyingTrade(qty=0, entry_price=S, entry_time=d, cost_basis=0.0)
        return True, leg, trade

    def check_exit_conditions(self, holding: int, px: float, open_trade: OptionTrade) -> tuple[bool, bool]:
        # underlying exit rule (for now: time-based only)
        hit_time = False
        return False, hit_time


class OptionEntryExit(BaseEntryExit):
    def __init__(self, name,
                 entry_rsi_low=30.0,
                 entry_rsi_high=50.0,
                 hold_days=180,
                 profit_take=0.50):
        super().__init__(name)
        self.entry_rsi_low = entry_rsi_low
        self.entry_rsi_high = entry_rsi_high
        self.hold_days = hold_days
        self.profit_take = profit_take

    def setRsiHoldProfit(self, entry_rsi_low: float = 30.0, entry_rsi_high: float = 50.0, hold_days: int = 180, profit_take: float = 0.50):
        self.entry_rsi_low = entry_rsi_low
        self.entry_rsi_high = entry_rsi_high
        self.hold_days = hold_days
        self.profit_take = profit_take

    def check_entry_conditions(self, d, idx, expiry_mode, S, q, r, end_ts, rsi: float, sigma: float, strike_round, target_delta, min_days_out=365) -> [bool, OptionLeg, OptionTrade]:
        """
        Check if the entry criteria are met.
        :param d: Current date.
        :param idx: Current index in the data.
        :param expiry_mode: Mode for selecting option expiry.
        :param S: Current stock price.
        :param q: Dividend yield.
        :param r: Risk-free rate.
        :param end_ts: End timestamp of the backtest.
        :param rsi: RSI value.
        :param sigma: Volatility.
        :param strike_round: Rounding increment for strike price.
        :param target_delta: Target delta for option selection.
        :param min_days_out: Minimum days to option expiry.
        :return: True if entry criteria are met, False otherwise.
        """
        # Ensure we can hold 180 days within the backtest window
        if d + pd.Timedelta(days=self.hold_days) > end_ts:
            return False, None, None

        if pd.isna(rsi) or not (self.entry_rsi_low <= float(rsi) <= self.entry_rsi_high):
            return False, None, None

        if not (math.isfinite(sigma) and sigma > 0):
            return False, None, None

        ok, expiry, K, entry_px = OptionLeg.check_entry_px(d=d, idx=idx, expiry_mode=expiry_mode, S=S, q=q, r=r, sigma=sigma, strike_round=strike_round, target_delta=target_delta, min_days_out=min_days_out)
        leg = OptionLeg(strike=K, expiry=expiry)
        # btcp will set qty and cost_basis
        trade = OptionTrade(qty=0, entry_price=entry_px, entry_time=d, profit_take=self.profit_take, cost_basis=0.0)
        return ok, leg, trade

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
