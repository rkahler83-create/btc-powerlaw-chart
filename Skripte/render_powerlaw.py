# Skripte/render_powerlaw.py

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTFILE = "docs/powerlaw.png"

START_DATE = dt.date(2011, 1, 1)
END_DATE = dt.date(2040, 12, 31)
GENESIS = dt.date(2009, 1, 3)

Y_MAX = 10_000_000

COL_PRICE = "#F7930A"
COL_REG   = "#7FC64B"
COL_SUP   = "#D04B3F"
COL_RES   = "#7E3FBF"

BG = "#0B0C10"

SOURCE_TEXT = "Quelldaten: Yahoo Finance (Ticker BTC-USD)"


def days_since_genesis(d):
    return max(1, (d - GENESIS).days)


def fetch_data():

    df = yf.download(
        "BTC-USD",
        start=START_DATE.isoformat(),
        progress=False
    )

    close = df["Close"]

    close.index = pd.to_datetime(close.index)

    return close


def fit_powerlaw(x, y):

    lx = np.log10(x)
    ly = np.log10(y)

    b, a = np.polyfit(lx, ly, 1)

    return a, b


def eval_powerlaw(a, b, x):

    return 10 ** (a + b * np.log10(x))


def main():

    close = fetch_data()

    dates = close.index.date

    x_hist = np.array([days_since_genesis(d) for d in dates])
    y_hist = close.values

    a_mid, b_mid = fit_powerlaw(x_hist, y_hist)

    x_line = np.geomspace(x_hist.min(), days_since_genesis(END_DATE), 1200)

    y_reg = eval_powerlaw(a_mid, b_mid, x_line)

    y_sup = y_reg * 0.35
    y_res = y_reg * 2.8

    plt.close("all")

    fig = plt.figure(figsize=(16,9), dpi=180)
    ax = fig.add_subplot(111)

    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(x_hist, y_hist, color=COL_PRICE, linewidth=2.2, label="Preis (Tagesschluss)")
    ax.plot(x_line, y_reg, color=COL_REG, linewidth=2.2, label="Lineare Regression")
    ax.plot(x_line, y_sup, color=COL_SUP, linewidth=2.2, label="Unterstützung")
    ax.plot(x_line, y_res, color=COL_RES, linewidth=2.2, label="Widerstand")

    ax.set_ylim(0.1, Y_MAX)

    # ⭐ DEINE Y-ACHSE
    ticks = [
        0.1,
        1,
        5,
        10,
        50,
        100,
        500,
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        10000000
    ]

    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else v))

    years = list(range(2011,2041))
    xticks = [days_since_genesis(dt.date(y,1,1)) for y in years]

    ax.set_xticks(xticks)
    ax.set_xticklabels(years, rotation=60)

    ax.legend(loc="lower right")

    ax.text(
        0.10,
        0.93,
        SOURCE_TEXT,
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        bbox=dict(
            facecolor=(0,0,0,0),
            edgecolor="white",
            linewidth=0.8
        )
    )

    fig.savefig(
        OUTFILE,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
