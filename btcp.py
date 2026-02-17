"""
QQQ "LEAP" Call Strategy Backtest (Model-Based)
Period: 2022-01-01 to 2025-12-26

Rules
- Entry: RSI(14) between 30 and 50 (inclusive)
- Buy: ~1-year call targeting delta ~0.80
- Exit: after 180 calendar days OR earlier if option hits +50% price gain
- Vol input: VXN implied vol index (default). Fallback to realized vol if missing.
- Expiry rule (choose one):
    * "monthly_3rd_friday": first 3rd-Friday monthly expiry >= 365 days out
    * "leaps_jan_3rd_friday": first Jan 3rd-Friday >= 365 days out (LEAPS-style)

Data sources (downloaded at runtime):
- QQQ daily: Stooq CSV endpoint
- VXN (VXNCLS): FRED CSV endpoint
- 1Y rate (DGS1): FRED CSV endpoint (optional)

Notes
- This is NOT real option quotes; it's Black–Scholes approximation.
- Uses European call formula; QQQ options are American-style (approx is usually close for calls w/ low div).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from criteria import OptionEntryExit
from buy_and_hold import BuyAndHold
from option_rsi import OptionRSI
from data_classes import trade_stats

##############

############
START_DATE = "1995-01-01"
END_DATE = "2025-12-29"
STOCK_SYMBOL = "QQQ.US"


# -----------------------------
# Data download helpers
# -----------------------------

STOOQ_CSV = "https://stooq.com/q/d/l/?i=d&s={symbol}"          # e.g. qqq.us


def download_stooq(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(STOOQ_CSV.format(symbol=symbol.lower()), parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    # Columns: Open, High, Low, Close, Volume
    return df


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"  # e.g. VXNCLS, DGS1


def download_fred(series_id: str) -> pd.Series:
    """
    Robust FRED CSV loader:
    - Handles UTF-8 BOM in headers (common cause of missing 'DATE')
    - Doesn’t assume the date column is exactly named 'DATE'
    - Detects HTML/error responses and raises a helpful error
    """
    url = FRED_CSV.format(series_id=series_id)

    # utf-8-sig strips BOM if present
    df = pd.read_csv(url, encoding="utf-8-sig")

    # Clean header whitespace/BOM oddities
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    if df.shape[1] < 2:
        # Often indicates an HTML error page got parsed weirdly
        head = df.head(3).to_string(index=False)
        raise RuntimeError(
            f"Unexpected response from FRED for {series_id}. "
            f"Columns={df.columns.tolist()}\nFirst rows:\n{head}"
        )

    date_col = df.columns[0]
    value_col = series_id if series_id in df.columns else df.columns[1]

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)

    # Parse numeric values (FRED sometimes uses '.' for missing)
    s = pd.to_numeric(df[value_col], errors="coerce")
    s.name = series_id
    return s


# -----------------------------
# Indicators: RSI (Wilder)
# -----------------------------

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# -----------------------------
# Backtest engine
# -----------------------------


def backtest(
    entryExit_df: pd.DataFrame,
    symbol: str = STOCK_SYMBOL,
    start: str = START_DATE,
    end: str = END_DATE,
    target_delta: float = 0.90,
    profit_take: float = 0.50,
    expiry_mode: str = "monthly_3rd_friday",     # or "leaps_jan_3rd_friday"
    vol_source: str = "vxn",                     # "vxn" or "hv" or "vxn_then_hv"
    hv_window: int = 30,
    q: float = 0.00,
    use_fred_rate: bool = True,                  # if False, uses constant_r
    constant_r: float = 0.04,
    strike_round: float = 1.0,
    starting_cash: float = 8_000.0,
    position_size_pct: float = 0.95,
    commission_per_contract: float = 0.65,
) -> [pd.DataFrame, pd.DataFrame]:

    # Load underlying
    raw = download_stooq(symbol)
    data = raw.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()
    idx = data.index

    # RSI
    data["rsi14"] = rsi_wilder(data["Close"], 14)

    # Realized vol (fallback)
    data["logret"] = np.log(data["Close"]).diff()
    data["hv"] = data["logret"].rolling(hv_window).std() * math.sqrt(252)
    data["hv"] = data["hv"].clip(lower=0.05, upper=1.50)

    # VXN (implied vol proxy), convert % to decimal
    vxn = download_fred("VXNCLS")
    vxn = vxn.reindex(idx).ffill() / 100.0
    vxn = vxn.clip(lower=0.05, upper=1.50)
    data["vxn_iv"] = vxn

    # Risk-free rate (DGS1) optional, convert % to decimal
    if use_fred_rate:
        dgs1 = download_fred("DGS1")
        r_series = (dgs1.reindex(idx).ffill() / 100.0).clip(lower=0.0, upper=0.20)
        data["r"] = r_series
    else:
        data["r"] = float(constant_r)

    def choose_sigma(srow) -> float:
        if vol_source == "vxn":
            return float(srow["vxn_iv"]) if pd.notna(srow["vxn_iv"]) else float("nan")
        if vol_source == "hv":
            return float(srow["hv"]) if pd.notna(srow["hv"]) else float("nan")
        if vol_source == "vxn_then_hv":
            if pd.notna(srow["vxn_iv"]):
                return float(srow["vxn_iv"])
            return float(srow["hv"])
        raise ValueError(f"Unknown vol_source: {vol_source}")

    # Initialize portfolios
    for row in entryExit_df.itertuples():
        entryExit = row.entryExit
        entryExit.portfolio.cash = starting_cash

    end_ts = pd.Timestamp(end)

    for d in idx:
        row = data.loc[d]
        S = float(row["Close"])
        r = float(row["r"])
        rsi = row["rsi14"]
        sigma = choose_sigma(row)

        for row in entryExit_df.itertuples():
            entryExit = row.entryExit
            exit_true = False
            # Mark-to-market
            open_trade, base_leg = entryExit.portfolio.first_open_trade()
            if open_trade and open_trade.qty > 0:
                px = base_leg.get_price(S, r, q, sigma, d)
                pos_value = open_trade.posn_value(px)
                equity = entryExit.portfolio.cash + pos_value

                holding = open_trade.holding(d)
                hit_profit, exit_true = entryExit.check_exit_conditions(holding, px, open_trade)
                if exit_true:
                    # print(f"{entryExit.name} | {d.date()} Exit: S={S} K={base_leg.strike} exp={base_leg.expiry.date()} hit_profit={hit_profit} Tpx=${open_trade.target_price:.2f} px=${px:.2f}  qty={open_trade.qty} pos_value=${pos_value:,.2f} equity=${equity:,.2f} rsi={rsi} sigma={sigma} holding_days={holding}")
                    entryExit.portfolio.cash += pos_value
                    entryExit.portfolio.cash -= commission_per_contract * open_trade.qty
                    open_trade.exit_price = px
                    open_trade.exit_time = d  # open_trade is now closed

                    entryExit.report.append({
                        "entry_date": open_trade.entry_time.date(),
                        "exit_date": d.date(),
                        "contracts": open_trade.qty,
                        "entry_price": open_trade.entry_price,
                        "exit_price": px,
                        "holding_days": holding,
                        "reason": "profit_target" if hit_profit else "time_exit",
                        "pnl_$": pos_value - open_trade.cost_basis,
                        "return_%": open_trade.total_ret(px),
                    })
                    equity = entryExit.portfolio.cash
            else:
                equity = entryExit.portfolio.cash

            entryExit.portfolio.equity_curve.append({"Date": d, "Equity": equity})

            # Entries
            if exit_true or not open_trade:
                ok, leg, open_trade = entryExit.check_entry_conditions(d, idx, expiry_mode, S, q, r, end_ts, rsi, sigma, strike_round, target_delta, min_days_out=365)
                if not ok:
                    continue
                mult = open_trade.multiplier
                alloc = equity * position_size_pct
                qty = int(alloc // (open_trade.entry_price * mult))
                if qty < 1:
                    qty = 1 if entryExit.portfolio.cash >= open_trade.entry_price * mult else 0
                if qty == 0:
                    continue

                cost = open_trade.entry_price * mult * qty + commission_per_contract * qty
                if cost > entryExit.portfolio.cash:
                    qty = int(entryExit.portfolio.cash // (open_trade.entry_price * mult))
                    if qty < 1:
                        continue
                    cost = open_trade.entry_price * mult * qty + commission_per_contract * qty
                    if cost > entryExit.portfolio.cash:
                        continue

                entryExit.portfolio.cash -= cost
                open_trade.qty=qty
                open_trade.cost_basis=cost
                leg.trades.append(open_trade)
                entryExit.portfolio.legs.append(leg) # Add leg to portfolio
                # print(f"{entryExit.name} | {d.date()} ENTRY: S={S} K={leg.strike} exp={leg.expiry.date()} Tpx=${open_trade.target_price:.2f} px=${open_trade.entry_price:.2f} qty={qty} cost=${cost:,.2f} rsi={rsi} sigma={sigma}")

    stratDf = pd.DataFrame(columns=["name", "end_eq", "cagr", "mdd", "sharpe", "losses", "profit_factor", "win_rate", "wins", "trades"])
    plotDf = pd.DataFrame(columns=["name", "index", "values"])
    for row in entryExit_df.itertuples():
        entryExit = row.entryExit
        eq = pd.DataFrame(entryExit.portfolio.equity_curve).set_index("Date")
        rf_daily = data["r"].reindex(eq.index).ffill() / 252.0

        losses, profit_factor, win_rate, wins = trade_stats(entryExit.report)
        end_eq, cagr, mdd, sharpe = OptionRSI.calculate_metrics(eq, rf_daily)
        stratDf.loc[len(stratDf)] = [entryExit.name, end_eq, cagr*100, mdd*100, sharpe, losses, profit_factor, win_rate*100, wins, entryExit.portfolio.num_portfolio_trades()]
        plotDf.loc[len(plotDf)] = [entryExit.name, eq.index, eq["Equity"]]

    # Buy-and-hold underlying QQQ comparison (uses fractional shares for simplicity)
    bah = BuyAndHold(starting_cash=starting_cash)
    buy_and_hold, bh_end, bh_cagr, bh_mdd, bh_sharpe = bah.calculate_metrics(data, rf_daily)
    stratDf.loc[len(stratDf)] = ["buy_and_hold", bh_end, bh_cagr*100, bh_mdd*100, bh_sharpe, 0, float("nan"), float("nan"), 0, 0]
    plotDf.loc[len(plotDf)] = ["buy_and_hold", buy_and_hold.index, buy_and_hold.values]

    return stratDf, plotDf


def main():
    # Change these if you want:
    entryExit_df = pd.DataFrame(columns=["entryExit"])
    entryExit_df.loc[len(entryExit_df)] = [OptionEntryExit("entry_rsi_30", entry_rsi_low=30.0, entry_rsi_high=50.0, hold_days=180, profit_take=0.50)]
    entryExit_df.loc[len(entryExit_df)] = [OptionEntryExit("entry_rsi_20", entry_rsi_low=20.0, entry_rsi_high=70.0, hold_days=180, profit_take=0.50)]

    stratDf, plotDf = backtest(
        entryExit_df=entryExit_df,
        symbol=STOCK_SYMBOL,
        start=START_DATE,
        end=END_DATE,
        expiry_mode="monthly_3rd_friday",   # try "leaps_jan_3rd_friday"
        vol_source="vxn_then_hv",          # safer than pure "vxn"
        use_fred_rate=True,
        position_size_pct=0.10,
        commission_per_contract=0.65,
    )

    print_summary(stratDf)
    # showTrades(entryExit.trades) ## TODO
    plot_plotdf(plotDf, title="Equity Curve: Strategies QQQ", xlablel="Date", ylabel="Equity ($)")


def plot_plotdf(plotDf, title, xlablel, ylabel):
    plt.figure()
    for row in plotDf.itertuples():
        plt.plot(row.index, row.values, label=row.name)
    plt.title(title)
    plt.xlabel(xlablel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_summary(stratDf):
    print("\n=== SUMMARY ===")
    print(f"symbol  : {STOCK_SYMBOL}")
    print(f"start   : {START_DATE}")
    print(f"end     : {END_DATE}")
    print(f"expiry_mode : monthly_3rd_friday")
    print(f"vol_source  : vxn_then_hv")
    with pd.option_context('display.float_format', '{:,.2f}'.format,
                           'display.max_columns', None, 'display.max_rows', None):
        print("\n=== STRATEGY METRICS ===")
        print(stratDf.to_string(index=False))


def showTrades(trades):
    trades_df = pd.DataFrame(trades)
    print("\n=== TRADES ===")
    if len(trades_df) == 0:
        print("No trades.")
    else:
        cols = ["entry_date", "exit_date", "expiry", "K", "contracts",
                "entry_price", "exit_price", "holding_days", "reason", "return_%", "pnl_$"]
        out = trades_df[cols].copy()
        out["K"] = out["K"].round(0).astype(int)
        out["entry_price"] = out["entry_price"].round(2)
        out["exit_price"] = out["exit_price"].round(2)
        out["return_%"] = out["return_%"].round(2)
        out["pnl_$"] = out["pnl_$"].round(2)
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
