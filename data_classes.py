# backtest
# trade_rules (list)
#   entry_cutoff
#   exit_cutoff
#       trade_portfolios (list)
#         TradePortfolio
#           cash
#           positions (list)
#             OptionPosition
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


@dataclass
class OptionTrade:
    contracts: int = 0
    entry_price: float = 0.0
    entry_time: pd.Timestamp = pd.Timestamp.min
    exit_price: float = 0.0
    exit_time: pd.Timestamp = pd.Timestamp.min
    target_price: float = 0.0
    cost_basis: float = 0.0
    pnl: float = 0.0
    ret: float = 0.0
    reason: str = ""

    def __init__(self, contracts: int, entry_price: float, entry_time: pd.Timestamp, profit_take: float, cost_basis: float):
        self.contracts = contracts
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.target_price = entry_price * (1 + profit_take)
        self.cost_basis = cost_basis

    @property
    def is_open(self) -> bool:
        return self.exit_time == pd.Timestamp.min

    def holding(self, d: pd.Timestamp) -> int:
        exit_day = d if self.is_open else self.exit_time
        return (exit_day - self.entry_time).days

    def unrealised_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.contracts * 100

    # TODO: commission_per_contract
    def realised_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.contracts * 100

    # TODO: commission_per_contract
    def total_pnl(self, current_price: float) -> float:
        if self.is_open:
            return (current_price - self.entry_price) * self.contracts * 100
        return (self.exit_price - self.entry_price) * self.contracts * 100

    def total_ret(self, current_price: float) -> float:
        if self.is_open:
            return ((current_price / self.entry_price) - 1) * 100
        return ((self.exit_price / self.entry_price) - 1) * 100


@dataclass
class OptionPosition:
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: pd.Timestamp = pd.Timestamp.min
    trades: List[OptionTrade] = field(default_factory=list)

    def first_open_trade(self) -> Optional[OptionTrade]:
        for trade in self.trades:
            if trade.is_open:
                return trade
        return None

    def num_trades(self) -> int:
        return len(self.trades)

    def posn_contracts(self) -> int:
        return sum(trade.contracts for trade in self.trades if not trade.is_open)

    def pnl(self, current_price: float) -> float:
        return sum(trade.total_pnl(current_price) for trade in self.trades)


@dataclass
class TradePortfolio:
    cash: float = 0.0
    positions: List[OptionPosition] = field(default_factory=list)
    equity_curve: List[Dict[str, object]] = field(default_factory=list)

    def add_position(self, option_type: str, strike: float, expiry: pd.Timestamp, contracts: int, entry_time: pd.Timestamp, entry_price: float, profit_take: float, cost: float) -> tuple[OptionTrade, OptionPosition]:
        position = OptionPosition(option_type=option_type, strike=strike, expiry=expiry)
        trade = OptionTrade(
            contracts=contracts,
            entry_price=entry_price,
            entry_time=entry_time,
            profit_take=profit_take,
            cost_basis=cost
        )

        position.trades.append(trade)
        self.positions.append(position)
        return trade, position

    def num_portfolio_trades(self) -> int:
        return sum(position.num_trades() for position in self.positions)

    def first_open_trade(self) -> Union[tuple[OptionTrade, OptionPosition], tuple[None, None]]:
        for position in self.positions:
            trade = position.first_open_trade()
            if trade is not None:
                return trade, position
        return None, None

    def pnl(self, current_price: float) -> float:
        return sum(position.pnl(current_price) for position in self.positions)


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


@dataclass
class TradeRule:
    entry_cutoff: pd.Timestamp = pd.Timestamp.min
    exit_cutoff: pd.Timestamp = pd.Timestamp.min
    trade_portfolios: List[TradePortfolio] = field(default_factory=list)


@dataclass
class Backtest:
    name: str
    start_date: pd.Timestamp = pd.Timestamp.min
    end_date: pd.Timestamp = pd.Timestamp.min
    trade_rules: List[TradeRule] = field(default_factory=list)
