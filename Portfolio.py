import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Position:
    entry_date: pd.Timestamp = pd.Timestamp.min
    expiry: pd.Timestamp = pd.Timestamp.min
    K: float = 0.0
    contracts: int = 0
    entry_price: float = 0.0
    target_price: float = 0.0
    cost_basis: float = 0.0
    trades: List[Dict[str, object]] = field(default_factory=list)

    def updposn(self, d: pd.Timestamp, expiry: pd.Timestamp, K: float, contracts: int, entry_px: float,
                profit_take: float, cost: float):

        self.entry_date = d
        self.expiry = expiry
        self.K = K
        self.contracts = contracts
        self.entry_price = entry_px
        self.target_price = entry_px * (1 + profit_take)
        self.cost_basis = cost


@dataclass
class Portfolio:
    cash: float = 0.0
    posn : Position = field(default_factory=Position)
    equity_curve: List[Dict[str, object]] = field(default_factory=list)

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

