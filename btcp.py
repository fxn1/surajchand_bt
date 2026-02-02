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
from calendar import monthcalendar, FRIDAY
from typing import Optional, Dict, List

import pandas as pd
import matplotlib.pyplot as plt

from black_scholes import BlackScholesModel
from criteria import EntryExit
from buy_and_hold import BuyAndHold
from option_rsi import OptionRSI
from Portfolio import Position, trade_stats

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


import pandas as pd
import numpy as np

FRED_CSV  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"  # e.g. VXNCLS, DGS1

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
# Expiration rules
# -----------------------------

def third_friday(year: int, month: int) -> pd.Timestamp:
    cal = monthcalendar(year, month)
    fridays = [week[FRIDAY] for week in cal if week[FRIDAY] != 0]
    return pd.Timestamp(year=year, month=month, day=fridays[2])  # 3rd Friday


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
        exp = third_friday(y, m)
        if exp < target:
            # go to next month
            if m == 12:
                y, m = y + 1, 1
            else:
                m += 1
            exp = third_friday(y, m)
        return adjust_to_prev_trading_day(exp, trading_index)

    if mode == "leaps_jan_3rd_friday":
        y = entry.year + 1
        exp = third_friday(y, 1)
        if exp < target:
            exp = third_friday(y + 1, 1)
        return adjust_to_prev_trading_day(exp, trading_index)

    raise ValueError(f"Unknown expiry mode: {mode}")

# -----------------------------
# Backtest engine
# -----------------------------

def backtest(
    bs_model: BlackScholesModel,
    entryExit_df:pd.DataFrame,
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
) -> Dict[str, object]:

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

    def choose_sigma(row) -> float:
        if vol_source == "vxn":
            return float(row["vxn_iv"]) if pd.notna(row["vxn_iv"]) else float("nan")
        if vol_source == "hv":
            return float(row["hv"]) if pd.notna(row["hv"]) else float("nan")
        if vol_source == "vxn_then_hv":
            if pd.notna(row["vxn_iv"]):
                return float(row["vxn_iv"])
            return float(row["hv"])
        raise ValueError(f"Unknown vol_source: {vol_source}")

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
            # Mark-to-market
            if entryExit.portfolio.posn.contracts > 0:
                T = max((entryExit.portfolio.posn.expiry - d).days / 365.0, 1 / 365.0)
                px, _ = bs_model.bs_call_price_delta(S, entryExit.portfolio.posn.strike, T, r, q, sigma)
                pos_value = px * 100 * entryExit.portfolio.posn.contracts
                equity = entryExit.portfolio.cash + pos_value

                holding = (d - entryExit.portfolio.posn.entry_date).days
                hit_profit, exit_true = entryExit.check_exit_conditions(holding, px)
                if exit_true:
                    entryExit.portfolio.cash += pos_value
                    entryExit.portfolio.cash -= commission_per_contract * entryExit.portfolio.posn.contracts

                    entryExit.portfolio.posn.trades.append({
                        "entry_date": entryExit.portfolio.posn.entry_date.date(),
                        "exit_date": d.date(),
                        "expiry": entryExit.portfolio.posn.expiry.date(),
                        "K": entryExit.portfolio.posn.strike,
                        "contracts": entryExit.portfolio.posn.contracts,
                        "entry_price": entryExit.portfolio.posn.entry_price,
                        "exit_price": px,
                        "holding_days": holding,
                        "reason": "profit_target" if hit_profit else "time_exit",
                        "pnl_$": pos_value - entryExit.portfolio.posn.cost_basis,
                        "return_%": (px - entryExit.portfolio.posn.entry_price) / entryExit.portfolio.posn.entry_price * 100.0,
                    })

                    entryExit.portfolio.posn.contracts = 0
                    equity = entryExit.portfolio.cash

            else:
                equity = entryExit.portfolio.cash

            entryExit.portfolio.equity_curve.append({"Date": d, "Equity": equity})

            # Entries
            if entryExit.portfolio.posn.contracts <= 0:
                if not entryExit.check_entry_conditions(d, end_ts, rsi, sigma):
                    continue

                expiry = pick_expiry(d, idx, mode=expiry_mode, min_days_out=365)
                if expiry <= d:
                    continue

                T0 = (expiry - d).days / 365.0
                K = bs_model.strike_for_delta_call(S, target_delta, T0, r, q, sigma, strike_round=strike_round)
                entry_px, _ = bs_model.bs_call_price_delta(S, K, T0, r, q, sigma)
                if not (math.isfinite(entry_px) and entry_px > 0):
                    continue

                alloc = equity * position_size_pct
                contracts = int(alloc // (entry_px * 100))
                if contracts < 1:
                    contracts = 1 if entryExit.portfolio.cash >= entry_px * 100 else 0
                if contracts == 0:
                    continue

                cost = entry_px * 100 * contracts + commission_per_contract * contracts
                if cost > entryExit.portfolio.cash:
                    contracts = int(entryExit.portfolio.cash // (entry_px * 100))
                    if contracts < 1:
                        continue
                    cost = entry_px * 100 * contracts + commission_per_contract * contracts
                    if cost > entryExit.portfolio.cash:
                        continue

                entryExit.portfolio.cash -= cost
                entryExit.portfolio.posn.updposn(d, expiry, K, contracts, entry_px, profit_take, cost)

    stratDf = pd.DataFrame(columns=["name", "end_eq", "cagr", "mdd", "sharpe", "losses", "profit_factor", "win_rate", "wins", "trades"])
    plotDf = pd.DataFrame(columns=["name", "index", "values"])
    for row in entryExit_df.itertuples():
        entryExit = row.entryExit
        eq = pd.DataFrame(entryExit.portfolio.equity_curve).set_index("Date")
        rf_daily = data["r"].reindex(eq.index).ffill() / 252.0

        losses, profit_factor, win_rate, wins = trade_stats(entryExit.portfolio.posn.trades)
        end_eq, cagr, mdd, sharpe = OptionRSI.calculate_metrics(eq, rf_daily)
        stratDf.loc[len(stratDf)] = [entryExit.name, end_eq, cagr*100, mdd*100, sharpe, losses, profit_factor, win_rate*100, wins, len(entryExit.portfolio.posn.trades)]
        plotDf.loc[len(plotDf)] = [entryExit.name, eq.index, eq["Equity"]]

    # Buy-and-hold underlying QQQ comparison (uses fractional shares for simplicity)
    bah = BuyAndHold(starting_cash=starting_cash)
    buy_and_hold, bh_end, bh_cagr, bh_mdd, bh_sharpe = bah.calculate_metrics(data, rf_daily)
    stratDf.loc[len(stratDf)] = ["buy_and_hold", bh_end, bh_cagr*100, bh_mdd*100, bh_sharpe, 0, float("nan"), float("nan"), 0, 0]
    plotDf.loc[len(plotDf)] = ["buy_and_hold", buy_and_hold.index, buy_and_hold.values]

    return stratDf, plotDf

def main():
    # Change these if you want:
    entryExit = EntryExit("entry_rsi_30")
    entryExit.setRsiHoldProfit(entry_rsi_low=30.0, entry_rsi_high=50.0, hold_days=180, profit_take=0.50)
    entryExit1 = EntryExit("entry_rsi_20")
    entryExit1.setRsiHoldProfit(entry_rsi_low=20.0, entry_rsi_high=70.0, hold_days=180, profit_take=0.50)
    entryExit_df = pd.DataFrame(columns=["entryExit"])
    entryExit_df.loc[len(entryExit_df)] = [entryExit]
    entryExit_df.loc[len(entryExit_df)] = [entryExit1]

    stratDf, plotDf = backtest(
        bs_model=BlackScholesModel(q=0.00),
        entryExit_df=entryExit_df,
        symbol= STOCK_SYMBOL,
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