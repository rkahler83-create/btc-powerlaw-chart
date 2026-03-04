import datetime as dt
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yfinance as yf

OUTFILE = "docs/powerlaw.png"

START_DATE = "2011-01-01"
END_YEAR = 2040
GENESIS = dt.date(2009, 1, 3)

# Power Law Parameter (log-log)
B = 5.68
A = 10 ** (-16.493)

SUPPORT_MULT = 0.35
RESIST_MULT = 2.20

COL_PRICE = "#F7931A"
COL_REG = "#9BE26D"
COL_SUP = "#E44E3A"
COL_RES = "#8E5BFF"

BG = "#0B0D10"

def days_since_genesis(d):
    return max(1, (d - GENESIS).days)

def powerlaw_price(days):
    return A * np.power(days, B)

def fetch_data():
    df = yf.download(
        "BTC-USD",
        start=START_DATE,
        interval="1d",
        progress=False
    )
    s = df["Close"].dropna()
    dates = s.index.date
    prices = s.to_numpy(dtype=float)
    x = np.array([float(days_since_genesis(d)) for d in dates])
    return x, prices

def main():
    x_days, y_close = fetch_data()

    x_min = float(days_since_genesis(dt.date(2011, 1, 1)))
    x_max = float(days_since_genesis(dt.date(END_YEAR, 12, 31)))

    x_line = np.logspace(math.log10(x_min), math.log10(x_max), 900)
    y_reg = powerlaw_price(x_line)

    # Regression an aktuellen Kurs anpassen
    last_x = x_days[-1]
    last_price = y_close[-1]
    reg_last = powerlaw_price(np.array([last_x]))[0]
    scale = last_price / reg_last
    y_reg = y_reg * scale

    y_sup = y_reg * SUPPORT_MULT
    y_res = y_reg * RESIST_MULT

    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor=BG)
    ax = fig.add_subplot(111, facecolor=BG)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(x_line, y_res, color=COL_RES, lw=1.6)
    ax.plot(x_line, y_reg, color=COL_REG, lw=1.6)
    ax.plot(x_line, y_sup, color=COL_SUP, lw=1.6)
    ax.plot(x_days, y_close, color=COL_PRICE, lw=1.6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(100, 10_000_000)

    ax.grid(True, which="major", alpha=0.15)
    ax.grid(True, which="minor", alpha=0.07)

    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, pos: f"{int(v):,}")
    )

    years = list(range(2011, END_YEAR + 1))
    xticks = [days_since_genesis(dt.date(y, 1, 1)) for y in years]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(y) for y in years], rotation=25)

    ax.set_ylabel("USD")

    ax.text(
        0.02, 0.93,
        "Quelle Kursdaten: Yahoo Finance (Ticker BTC-USD)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
        bbox=dict(
            boxstyle="square,pad=0.35",
            facecolor="none",
            edgecolor="white",
            linewidth=1.0,
        ),
    )

    plt.tight_layout()
    fig.savefig(OUTFILE, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
