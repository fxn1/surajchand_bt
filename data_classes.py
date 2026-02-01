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
from typing import List, Optional
from datetime import datetime

@dataclass
class OptionTrade:
    qty: int
    entry_price: float
    entry_time: datetime
    exit_price: float = None
    exit_time: datetime = None

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def unrealised_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.qty * 100

    def realised_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty * 100

    def pnl(self, current_price: float) -> float:
        if self.is_open:
            return (current_price - self.entry_price) * self.qty * 100
        return (self.exit_price - self.entry_price) * self.qty * 100

@dataclass
class OptionPosition:
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: datetime
    trades: List[OptionTrade] = field(default_factory=list)

    def pnl(self, current_price: float) -> float:
        total_pnl = 0.0
        for trade in self.trades:
            total_pnl += trade.pnl(current_price)
        return total_pnl

@dataclass
class TradePortfolio:
    cash: float
    positions: List[OptionPosition] = field(default_factory=list)

    def pnl(self, current_price: float) -> float:
        total_pnl = 0.0
        for position in self.positions:
            total_pnl += position.pnl(current_price)
        return total_pnl

@dataclass
class TradeRule:
    entry_cutoff: datetime
    exit_cutoff: datetime
    trade_portfolios: List[TradePortfolio] = field(default_factory=list)

@dataclass
class Backtest:
    name: str
    start_date: datetime
    end_date: datetime
    trade_rules: List[TradeRule] = field(default_factory=list)
