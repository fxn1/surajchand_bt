# backtest
# trade_rules (list)
#   entry_cutoff
#   exit_cutoff
#       trade_portfolios (list)
#         TradePortfolio
#           cash
#           optionLeg (list)
#             OptionLeg
#               option_type
#               strike
#               expiry
#               trades (list)
#                   OptionTrade
#                       qty
#                       entry_price
#                       entry_time
#                       exit_price
#                       exit_time

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import pandas as pd
import math
from calendar import monthcalendar, FRIDAY
from black_scholes import BlackScholesModel


@dataclass
class BaseTrade:
    entry_price: float = 0.0
    entry_time: pd.Timestamp = pd.Timestamp.min
    exit_price: float = 0.0
    exit_time: pd.Timestamp = pd.Timestamp.min
    cost_basis: float = 0.0
    qty: int = 0
    multiplier: int = 1
    pnl: float = 0.0
    ret: float = 0.0
    reason: str = ""

    def __init__(self, entry_price: float, entry_time: pd.Timestamp, cost_basis: float, qty: int, multiplier: int):
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.exit_price = 0.0
        self.exit_time = pd.Timestamp.min
        self.cost_basis = cost_basis
        self.qty = qty
        self.multiplier = multiplier

    @property
    def is_open(self) -> bool:
        return self.exit_time == pd.Timestamp.min

    def holding(self, d: pd.Timestamp) -> int:
        exit_day = d if self.is_open else self.exit_time
        return (exit_day - self.entry_time).days

    def posn_value(self, current_price: float) -> float:
        px = current_price if self.is_open else self.exit_price
        return px * self.qty * self.multiplier

    def unrealised_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.qty * self.multiplier

    # TODO: commission_per_contract
    def realised_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty * self.multiplier

    # TODO: commission_per_contract
    def total_pnl(self, current_price: float) -> float:
        px = current_price if self.is_open else self.exit_price
        return (px - self.entry_price) * self.qty * self.multiplier

    def total_ret(self, current_price: float) -> float:
        if self.is_open:
            return ((current_price / self.entry_price) - 1) * 100
        return ((self.exit_price / self.entry_price) - 1) * 100


class UnderlyingTrade(BaseTrade):

    def __init__(self, qty: int, entry_price: float, entry_time: pd.Timestamp, cost_basis: float):
        BaseTrade.__init__(self, entry_price=entry_price, entry_time=entry_time, cost_basis=cost_basis, qty=qty, multiplier=1)


@dataclass
class OptionTrade(BaseTrade):
    target_price: float = 0.0

    def __init__(self, qty: int, entry_price: float, entry_time: pd.Timestamp, profit_take: float, cost_basis: float):
        BaseTrade.__init__(self, entry_price=entry_price, entry_time=entry_time, cost_basis=cost_basis, qty=qty, multiplier=100)
        self.target_price = entry_price * (1 + profit_take)


@dataclass
class BaseLeg:
    trades: List[BaseTrade] = field(default_factory=list)

    def get_price(self, S, r, q, sigma, d):
        raise NotImplementedError

    def first_open_trade(self) -> Optional[BaseTrade]:
        for trade in self.trades:
            if trade.is_open:
                return trade
        return None

    def num_trades(self) -> int:
        return len(self.trades)

    def pnl(self, current_price: float) -> float:
        return sum(trade.total_pnl(current_price) for trade in self.trades)


@dataclass
class UnderlyingLeg(BaseLeg):
    def get_price(self, S, r, q, sigma, d):
        """
        Compute the mark-to-market price - UnderlyingLeg: price = S
        """
        return S


@dataclass
class OptionLeg(BaseLeg):
    # option_type: str  # 'call' or 'put'
    strike: float = 0.0
    expiry: pd.Timestamp = pd.Timestamp.min
    bs_model: BlackScholesModel = BlackScholesModel(q=0.00)

    def get_price(self, S, r, q, sigma, d):
        """
        Compute the mark-to-market price = Black-Scholes call
        """
        T = max((self.expiry - d).days / 365.0, 1/365.0)
        px, _ = self.bs_model.bs_call_price_delta(S, self.strike, T, r, q, sigma)
        return px

    # -----------------------------
    # Expiration rules
    # -----------------------------

    @staticmethod
    def third_friday(year: int, month: int) -> pd.Timestamp:
        cal = monthcalendar(year, month)
        fridays = [week[FRIDAY] for week in cal if week[FRIDAY] != 0]
        return pd.Timestamp(year=year, month=month, day=fridays[2])  # 3rd Friday

    @staticmethod
    def adjust_to_prev_trading_day(ts: pd.Timestamp, trading_index: pd.DatetimeIndex) -> pd.Timestamp:
        """
        If ts isn't a trading day in the dataset, walk backward until it is.
        (Handles holidays like Good Friday without needing a full exchange calendar.)
        """
        t = ts
        while t not in trading_index:
            t -= pd.Timedelta(days=1)
            if (ts - t).days > 10:
                # Safety valve
                break
        return t

    @staticmethod
    def pick_expiry(entry: pd.Timestamp,
                    trading_index: pd.DatetimeIndex,
                    mode: str = "monthly_3rd_friday",
                    min_days_out: int = 365) -> pd.Timestamp:
        """
        mode:
          - monthly_3rd_friday: first monthly 3rd Friday >= entry + min_days_out
          - leaps_jan_3rd_friday: first Jan 3rd Friday >= entry + min_days_out
        """
        target = entry + pd.Timedelta(days=min_days_out)

        if mode == "monthly_3rd_friday":
            y, m = target.year, target.month
            exp = OptionLeg.third_friday(y, m)
            if exp < target:
                # go to next month
                if m == 12:
                    y, m = y + 1, 1
                else:
                    m += 1
                exp = OptionLeg.third_friday(y, m)
            return OptionLeg.adjust_to_prev_trading_day(exp, trading_index)

        if mode == "leaps_jan_3rd_friday":
            y = entry.year + 1
            exp = OptionLeg.third_friday(y, 1)
            if exp < target:
                exp = OptionLeg.third_friday(y + 1, 1)
            return OptionLeg.adjust_to_prev_trading_day(exp, trading_index)

        raise ValueError(f"Unknown expiry mode: {mode}")

    @staticmethod
    def get_entry_px(S, d, expiry, q, r, sigma, strike_round, target_delta):
        T0 = (expiry - d).days / 365.0
        K = OptionLeg.bs_model.strike_for_delta_call(S, target_delta, T0, r, q, sigma, strike_round=strike_round)
        entry_px, _ = OptionLeg.bs_model.bs_call_price_delta(S, K, T0, r, q, sigma)
        return K, entry_px

    @staticmethod
    def check_entry_px(d, idx, expiry_mode, S, q, r, sigma, strike_round, target_delta, min_days_out=365):
        expiry = OptionLeg.pick_expiry(d, idx, mode=expiry_mode, min_days_out=min_days_out)
        if expiry <= d:
            return False, None, None, None

        K, entry_px = OptionLeg.get_entry_px(S, d, expiry, q, r, sigma, strike_round, target_delta)

        if not (math.isfinite(entry_px) and entry_px > 0):
            return False, None, None, None

        return True, expiry, K, entry_px


@dataclass
class TradePortfolio:
    cash: float = 0.0
    legs: List[BaseLeg] = field(default_factory=list)
    equity_curve: List[Dict[str, object]] = field(default_factory=list)

    def add_optionLeg(self, strike: float, expiry: pd.Timestamp, contracts: int, entry_time: pd.Timestamp, entry_price: float, profit_take: float, cost: float) -> tuple[OptionTrade, OptionLeg]:
        optionLeg = OptionLeg(strike=strike, expiry=expiry)
        trade = OptionTrade(qty=contracts, entry_price=entry_price, entry_time=entry_time, profit_take=profit_take, cost_basis=cost)

        optionLeg.trades.append(trade)
        self.legs.append(optionLeg)
        return trade, optionLeg

    def add_underlyingLeg(self, qty, entry_time: pd.Timestamp, entry_price: float, cost: float):
        leg = UnderlyingLeg()
        trade = UnderlyingTrade(qty=qty, entry_price=entry_price, entry_time=entry_time, cost_basis=cost)
        leg.trades.append(trade)
        self.legs.append(leg)
        return trade, leg

    def num_portfolio_trades(self) -> int:
        return sum(leg.num_trades() for leg in self.legs)

    def first_open_trade(self) -> Union[tuple[BaseTrade, BaseLeg], tuple[None, None]]:
        for leg in self.legs:
            trade = leg.first_open_trade()
            if trade is not None:
                return trade, leg
        return None, None

    def pnl(self, current_price: float) -> float:
        return sum(leg.pnl(current_price) for leg in self.legs)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def trade_stats(trades):
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        wins = int((trades_df["pnl_$"] > 0).sum())
        losses = int((trades_df["pnl_$"] <= 0).sum())
        win_rate = wins / len(trades_df)
        profit_factor = (
            trades_df.loc[trades_df["pnl_$"] > 0, "pnl_$"].sum()
            / abs(trades_df.loc[trades_df["pnl_$"] <= 0, "pnl_$"].sum())
            if losses > 0 else float("inf")
        )
    else:
        wins = losses = 0
        win_rate = float("nan")
        profit_factor = float("nan")
    return losses, profit_factor, win_rate, wins
