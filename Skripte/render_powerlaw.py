import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

OUTFILE = "docs/powerlaw.png"

GENESIS = dt.date(2009, 1, 3)
START_DATE = dt.date(2011, 1, 1)
END_DATE = dt.date(2040, 12, 31)

TICKER = "BTC-USD"

COLOR_PRICE = "#F7931A"
COLOR_REG   = "#9BD36A"
COLOR_SUP   = "#E04B3F"
COLOR_RES   = "#8C4BD6"

Y_MIN = 1
Y_MAX = 10_000_000

def days_since_genesis(d):
    return max(1, (d - GENESIS).days)

def fmt_plain(x, pos):
    if x >= 1:
        return f"{int(x):,}"
    return ""

def load_data():
    df = yf.download(
        TICKER,
        start=GENESIS.isoformat(),
        progress=False
    )
    close = df["Close"].dropna().astype(float)
    close.index = close.index.tz_localize(None)
    close = close[close > 0]
    return close

def main():
    close_all = load_data()
    close = close_all[close_all.index.date >= START_DATE]

    x_days = np.array([days_since_genesis(d.date()) for d in close.index])
    y_price = close.values

    lx = np.log10(x_days)
    ly = np.log10(y_price)

    b, a = np.polyfit(lx, ly, 1)

    resid = ly - (a + b * lx)
    off_low = np.quantile(resid, 0.10)
    off_high = np.quantile(resid, 0.90)

    x_start = days_since_genesis(START_DATE)
    x_end = days_since_genesis(END_DATE)

    x_grid = np.logspace(np.log10(x_start), np.log10(x_end), 1500)

    y_reg = 10 ** (a + b * np.log10(x_grid))
    y_sup = 10 ** (a + b * np.log10(x_grid) + off_low)
    y_res = 10 ** (a + b * np.log10(x_grid) + off_high)

    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)

    fig.patch.set_facecolor("#0B0C10")
    ax.set_facecolor("#0B0C10")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(x_grid, y_res, color=COLOR_RES, linewidth=2.2, label="Widerstand")
    ax.plot(x_grid, y_reg, color=COLOR_REG, linewidth=2.2, label="Regression")
    ax.plot(x_grid, y_sup, color=COLOR_SUP, linewidth=2.2, label="Unterstützung")
    ax.plot(x_days, y_price, color=COLOR_PRICE, linewidth=2.0, label="Preis (Tagesschluss)")

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_plain))

    year_ticks = []
    year_labels = []
    for y in range(2011, 2041):
        year_ticks.append(days_since_genesis(dt.date(y, 1, 1)))
        year_labels.append(str(y))

    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels, rotation=30, ha="right")

    ax.grid(True, which="major", alpha=0.35, linewidth=1.0)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.6)

    ax.set_ylabel("USD", color="#CCCCCC")
    ax.tick_params(axis="both", colors="#DDDDDD")

    leg = ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="#0F0F0F",
        edgecolor="#3A3A3A",
        framealpha=0.9
    )
    for t in leg.get_texts():
        t.set_color("#E6E6E6")

    ax.text(
        days_since_genesis(dt.date(2011, 6, 1)),
        2_000_000,
        "Quelle Kursdaten: Yahoo Finance (Ticker BTC-USD)",
        color="white",
        fontsize=11,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="square,pad=0.45",
            facecolor="none",
            edgecolor="white",
            linewidth=1.0
        ),
    )

    for spine in ax.spines.values():
        spine.set_color("#22252A")

    plt.tight_layout()
    fig.savefig(OUTFILE, facecolor=fig.get_facecolor(), bbox_inches="tight")

if __name__ == "__main__":
    main()
