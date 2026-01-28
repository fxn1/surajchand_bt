import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Position:
    entry_date: pd.Timestamp
    expiry: pd.Timestamp
    K: float
    contracts: int
    entry_price: float
    target_price: float
    cost_basis: float


@dataclass
class Portfolio:
    cash: float
    pos: Optional[Position]
    trades: List[Dict[str, object]]
    equity_curve: List[Dict[str, object]]

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

