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
