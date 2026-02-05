import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Union
from data_classes import OptionTrade, OptionPosition


@dataclass
class Portfolio:
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

    # def update_equity_curve(self, current_price: float):
    #     equity = self.cash + sum(position.pnl(current_price) for position in self.positions)
    #     self.equity_curve.append({"Date": pd.Timestamp.now(), "Equity": equity})


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
